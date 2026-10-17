"""Offline, cost-conscious BigQuery DDL/view generator for Smart Factory.

This module does not import cloud SDKs, create datasets, submit queries or
load any user data. An offline SQL plan is NOT a BigQuery server-side dry run.
"""

from __future__ import annotations

import argparse
import json
import re

PROJECT_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
DATASET_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,255}$")
LOCATION_PATTERN = re.compile(r"^(?:US|EU|[a-z][a-z0-9-]{1,40})$")
MAXIMUM_BYTES_BILLED = 1024 * 1024 * 1024  # 1 GiB per future query job
SOURCE_LABEL = "OFFLINE PLAN ONLY: no BigQuery job, dataset, cost or permissions verified"


class BigQueryPlanError(ValueError):
    """Invalid identifier or attempted unsafe BigQuery configuration."""


def _config(project: str, prefix: str, location: str) -> tuple[str, str, str]:
    if not isinstance(project, str) or not PROJECT_PATTERN.fullmatch(project):
        raise BigQueryPlanError("Invalid GCP project ID")
    if not isinstance(prefix, str) or not DATASET_PATTERN.fullmatch(prefix):
        raise BigQueryPlanError("Invalid BigQuery dataset prefix")
    if len(prefix) > 237:
        raise BigQueryPlanError("Dataset prefix too long")
    if not isinstance(location, str) or not LOCATION_PATTERN.fullmatch(location):
        raise BigQueryPlanError("Invalid BigQuery location")
    return project, prefix, location


def build_sql_plan(project: str, prefix: str = "smart_factory", location: str = "EU") -> dict:
    """Return SQL statements for human review; never connect to Google Cloud."""
    project, prefix, location = _config(project, prefix, location)
    raw_dataset = prefix + "_raw"
    mart_dataset = prefix + "_mart"

    def table(dataset: str, name: str) -> str:
        return "`" + project + "." + dataset + "." + name + "`"

    def raw(name: str) -> str:
        return table(raw_dataset, name)

    def mart(name: str) -> str:
        return table(mart_dataset, name)

    source_fields = """
  source_sha256 STRING NOT NULL,
  ingestion_run_id STRING NOT NULL"""
    engine_sensors = ",\n".join(
        "  sensor_" + str(i) + " FLOAT64" for i in range(1, 22)
    )
    tables = [
        (
            "ai4i_records",
            source_fields
            + """,
  udi INT64 NOT NULL,
  product_id STRING,
  type STRING,
  air_temperature_k FLOAT64,
  process_temperature_k FLOAT64,
  rotational_speed_rpm FLOAT64,
  torque_nm FLOAT64,
  tool_wear_min FLOAT64,
  machine_failure INT64,
  twf INT64,
  hdf INT64,
  pwf INT64,
  osf INT64,
  rnf INT64,
  ingested_at TIMESTAMP NOT NULL""",
            raw_dataset,
        ),
        (
            "fd001_cycles",
            source_fields
            + """,
  split STRING NOT NULL,
  unit_number INT64 NOT NULL,
  time_cycles INT64 NOT NULL,
  setting_1 FLOAT64,
  setting_2 FLOAT64,
  setting_3 FLOAT64,
"""
            + engine_sensors
            + """,
  ingested_at TIMESTAMP NOT NULL""",
            raw_dataset,
        ),
        (
            "fd001_official_test_truth",
            source_fields
            + """,
  unit_number INT64 NOT NULL,
  true_rul_cycles INT64 NOT NULL,
  ingested_at TIMESTAMP NOT NULL""",
            raw_dataset,
        ),
        (
            "ai4i_model_scores",
            """\n  model_sha256 STRING NOT NULL,
  source_sha256 STRING NOT NULL,
  udi INT64 NOT NULL,
  risk_score FLOAT64 NOT NULL,
  scored_at TIMESTAMP NOT NULL""",
            mart_dataset,
        ),
        (
            "fd001_model_predictions",
            """\n  model_sha256 STRING NOT NULL,
  source_sha256 STRING NOT NULL,
  unit_number INT64 NOT NULL,
  predicted_rul_cycles FLOAT64 NOT NULL,
  scored_at TIMESTAMP NOT NULL""",
            mart_dataset,
        ),
    ]
    statements = [
        {
            "kind": "dataset_ddl",
            "name": dataset,
            "sql": (
                "CREATE SCHEMA IF NOT EXISTS `"
                + project
                + "."
                + dataset
                + "` OPTIONS(location='"
                + location
                + "');"
            ),
        }
        for dataset in (raw_dataset, mart_dataset)
    ]
    statements.extend(
        {
            "kind": "table_ddl",
            "name": name,
            "sql": "CREATE TABLE IF NOT EXISTS "
            + table(dataset, name)
            + " ("
            + cols
            + "\n);",
        }
        for name, cols, dataset in tables
    )
    # SQL views join only within one exact raw source fingerprint. The latest
    # ingested AI4I record is selected before deduplicated scores are attached.
    review_view = f"""CREATE OR REPLACE VIEW {mart("ai4i_review_queue")} AS
WITH raw_latest AS (
  SELECT * FROM {raw("ai4i_records")}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY source_sha256, udi
    ORDER BY ingested_at DESC, ingestion_run_id DESC
  ) = 1
),
scores_latest AS (
  SELECT * FROM {mart("ai4i_model_scores")}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_sha256, source_sha256, udi
    ORDER BY scored_at DESC
  ) = 1
)
SELECT s.model_sha256, s.source_sha256, r.udi, r.product_id,
       r.type, r.machine_failure, s.risk_score, s.scored_at
FROM scores_latest AS s
JOIN raw_latest AS r
  ON s.source_sha256 = r.source_sha256 AND s.udi = r.udi;"""
    rul_view = f"""CREATE OR REPLACE VIEW {mart("fd001_official_evaluation")} AS
WITH last_cycles AS (
  SELECT source_sha256, unit_number, MAX(time_cycles) AS last_observed_cycle
  FROM {raw("fd001_cycles")}
  WHERE split = 'test'
  GROUP BY source_sha256, unit_number
),
truth_latest AS (
  SELECT * FROM {raw("fd001_official_test_truth")}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY source_sha256, unit_number
    ORDER BY ingested_at DESC, ingestion_run_id DESC
  ) = 1
),
predictions_latest AS (
  SELECT * FROM {mart("fd001_model_predictions")}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY model_sha256, source_sha256, unit_number
    ORDER BY scored_at DESC
  ) = 1
)
SELECT p.model_sha256, p.source_sha256, p.unit_number,
       c.last_observed_cycle, t.true_rul_cycles,
       p.predicted_rul_cycles, p.scored_at
FROM predictions_latest AS p
JOIN last_cycles AS c
  ON p.source_sha256 = c.source_sha256 AND p.unit_number = c.unit_number
JOIN truth_latest AS t
  ON p.source_sha256 = t.source_sha256 AND p.unit_number = t.unit_number;"""
    statements.extend(
        (
            {"kind": "view_ddl", "name": "ai4i_review_queue", "sql": review_view},
            {
                "kind": "view_ddl",
                "name": "fd001_official_evaluation",
                "sql": rul_view,
            },
        )
    )
    return {
        "status": "offline_unexecuted",
        "notice": SOURCE_LABEL,
        "project": project,
        "location": location,
        "datasets": [raw_dataset, mart_dataset],
        "maximum_bytes_billed_per_query": MAXIMUM_BYTES_BILLED,
        "remote_bigquery_dry_run_completed": False,
        "cloud_ddl_execution_authorized": False,
        "statements": statements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print offline Smart Factory BigQuery SQL")
    parser.add_argument("--project", required=True)
    parser.add_argument("--prefix", default="smart_factory")
    parser.add_argument("--location", default="EU")
    args = parser.parse_args()
    try:
        plan = build_sql_plan(args.project, args.prefix, args.location)
    except BigQueryPlanError as exc:
        parser.exit(2, "UNSAFE BIGQUERY PLAN: " + str(exc) + "\n")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
