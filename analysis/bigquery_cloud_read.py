"""Optional BigQuery provider-side *read-only* query dry-run and cost gate.

Requires an already authenticated, authorized client and separately installed
BigQuery SDK. Importing this module does NOT connect to GCP or spend money.
All SQL comes from allowlisted historical quality views; the source SHA is a
bound STRING parameter, never interpolated into SQL.
"""

from __future__ import annotations

import os
import re
from typing import Any

from analysis.bigquery_plan import MAXIMUM_BYTES_BILLED, build_sql_plan

SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REPORT_VIEWS = ("ai4i_holdout_quality", "fd001_quality_summary")
EXECUTION_ENV = "NEXUS_ALLOW_BQ_READONLY_EXECUTION"


class CloudReadError(ValueError):
    """Cloud read is unauthorised, costly, misconfigured or unverified."""


def inspect_quality_views(
    client: Any,
    bigquery_sdk: Any,
    *,
    project: str,
    source_sha256: str,
    prefix: str = "smart_factory",
    location: str = "EU",
    max_bytes: int = MAXIMUM_BYTES_BILLED,
    execute: bool = False,
) -> dict:
    """Check actual remote metadata and query dry-run bytes before any read."""
    plan = build_sql_plan(project, prefix, location)
    if not isinstance(source_sha256, str) or not SHA_PATTERN.fullmatch(source_sha256):
        raise CloudReadError("A pinned 64-char source SHA-256 is required")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAXIMUM_BYTES_BILLED:
        raise CloudReadError("Invalid maximum_bytes_billed cap")
    if type(execute) is not bool:
        raise CloudReadError("execute must be a boolean")
    if execute and os.environ.get(EXECUTION_ENV) != "YES":
        raise CloudReadError("Read-only BigQuery execution is not enabled")

    # Verify the *actual* datasets and location; the offline SQL plan proves
    # neither project existence nor IAM permissions.
    for name in plan["datasets"]:
        try:
            dataset = client.get_dataset(project + "." + name)
        except Exception as exc:
            raise CloudReadError("Cannot verify required BigQuery dataset") from exc
        if getattr(dataset, "location", "").upper() != location.upper():
            raise CloudReadError("BigQuery dataset location mismatch")

    checks = []
    for view in REPORT_VIEWS:
        table = project + "." + plan["datasets"][1] + "." + view
        try:
            view_metadata = client.get_table(table)
        except Exception as exc:
            raise CloudReadError("Cannot verify quality report view") from exc
        if getattr(view_metadata, "table_type", None) != "VIEW":
            raise CloudReadError("Expected a read-only BigQuery view")
        query = (
            "SELECT * FROM `"
            + table
            + "` WHERE source_sha256 = @source_sha256 LIMIT 1"
        )
        parameters = [
            bigquery_sdk.ScalarQueryParameter("source_sha256", "STRING", source_sha256)
        ]
        dry_config = bigquery_sdk.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            maximum_bytes_billed=max_bytes,
            query_parameters=parameters,
            labels={"app": "smartfactory", "intent": "qa_dryrun"},
        )
        try:
            dry_job = client.query(query, job_config=dry_config, location=location)
        except Exception as exc:
            raise CloudReadError("BigQuery provider dry run failed") from exc
        estimate = getattr(dry_job, "total_bytes_processed", None)
        if type(estimate) is not int or estimate < 0 or estimate > max_bytes:
            raise CloudReadError("Provider dry run exceeds byte cap or has no estimate")

        details = {
            "view": view,
            "target": table,
            "estimated_bytes_processed": estimate,
            "maximum_bytes_billed": max_bytes,
            "provider_dry_run_completed": True,
            "report_query_executed": False,
            "rows": [],
        }
        if execute:
            real_config = bigquery_sdk.QueryJobConfig(
                dry_run=False,
                use_query_cache=False,
                maximum_bytes_billed=max_bytes,
                query_parameters=parameters,
                labels={"app": "smartfactory", "intent": "qa_read"},
            )
            try:
                job = client.query(query, job_config=real_config, location=location)
                details["rows"] = [dict(row) for row in job.result(max_results=1)]
            except Exception as exc:
                raise CloudReadError("Authorized bounded BigQuery read failed") from exc
            details["report_query_executed"] = True
        checks.append(details)
    return {
        "status": "provider_dry_run_passed",
        "project": project,
        "dataset_location_verified": True,
        "read_only": True,
        "source_sha256": source_sha256,
        "reports": checks,
        "cloud_write_performed": False,
        "historical_offline_benchmarks_only": True,
    }


def manual_cloud_preflight(
    *,
    project: str,
    source_sha256: str,
    prefix: str = "smart_factory",
    location: str = "EU",
    execute: bool = False,
) -> dict:
    """Explicit opt-in use; no credentials are loaded during module import."""
    if execute and os.environ.get(EXECUTION_ENV) != "YES":
        raise CloudReadError("Read-only BigQuery execution is not enabled")
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise CloudReadError(
            "Install the optional pinned BigQuery client before provider preflight"
        ) from exc
    client = bigquery.Client(project=project, location=location)
    return inspect_quality_views(
        client,
        bigquery,
        project=project,
        source_sha256=source_sha256,
        prefix=prefix,
        location=location,
        execute=execute,
    )
