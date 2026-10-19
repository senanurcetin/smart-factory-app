"""No-cloud, fail-closed preflight of Smart Factory BigQuery NDJSON batches.

This does not submit BigQuery load jobs, authenticate, run provider dry-runs,
query information schema or enforce BigQuery-side byte billing ceilings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from analysis.ai4i_batch import RUN_PATTERN
from analysis.bigquery_plan import build_sql_plan

MAX_NDJSON_BYTES = 64 * 1024 * 1024
MAX_NDJSON_ROWS = 100000
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_PATTERN = re.compile(
    r"^  ([a-z][a-z0-9_]*) (STRING|INT64|FLOAT64|TIMESTAMP)(?: NOT NULL)?[,;]?$"
)
ALLOWED_STAGING_TABLES = frozenset(("ai4i_records", "fd001_cycles", "fd001_official_test_truth"))


class LoadPreflightError(ValueError):
    """Batch is not safe, complete or schema-compatible for upload."""


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LoadPreflightError("Duplicate JSON key")
        result[key] = value
    return result


def _checked_line(line: str, fields: dict[str, tuple[str, bool]]) -> dict:
    try:
        item = json.loads(line, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError) as exc:
        raise LoadPreflightError("Invalid or duplicate-key NDJSON record") from exc
    if not isinstance(item, dict) or set(item) != set(fields):
        raise LoadPreflightError("NDJSON fields differ from BigQuery table schema")
    for name, (field_type, required) in fields.items():
        value = item[name]
        if value is None:
            if required:
                raise LoadPreflightError("Required column is null: " + name)
            continue
        if field_type == "STRING":
            if not isinstance(value, str):
                raise LoadPreflightError("Non-string value: " + name)
        elif field_type == "INT64":
            if type(value) is not int or abs(value) > 2**63 - 1:
                raise LoadPreflightError("Invalid INT64 value: " + name)
        elif field_type == "FLOAT64":
            if type(value) not in (float, int) or not math.isfinite(float(value)):
                raise LoadPreflightError("Nonfinite/invalid FLOAT64: " + name)
        elif field_type == "TIMESTAMP":
            if not isinstance(value, str):
                raise LoadPreflightError("Invalid timestamp: " + name)
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise LoadPreflightError("Invalid timestamp: " + name) from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise LoadPreflightError("Missing timestamp timezone: " + name)
    return item


def _schema(project: str, prefix: str, location: str, table: str) -> dict:
    if table not in ALLOWED_STAGING_TABLES:
        raise LoadPreflightError("Table outside approved Smart Factory staging list")
    plan = build_sql_plan(project, prefix, location)
    ddl = next(step["sql"] for step in plan["statements"] if step["name"] == table)
    matches = []
    for line in ddl.splitlines():
        match = SCHEMA_PATTERN.fullmatch(line.rstrip())
        if match:
            matches.append((match.group(1), (match.group(2), "NOT NULL" in line)))
    fields = dict(matches)
    if not fields or len(fields) != len(matches):
        raise LoadPreflightError("Cannot parse trusted BigQuery schema")
    return {"fields": fields, "plan": plan}


def validate_batch_upload(
    ndjson_path: Path,
    *,
    expected_file_sha256: str,
    expected_source_sha256: str,
    expected_rows: int,
    project: str,
    table: str,
    prefix: str = "smart_factory",
    location: str = "EU",
) -> dict:
    """Validate bytes, schema, source identity and key uniqueness locally."""
    for key, value in (
        ("file", expected_file_sha256),
        ("source", expected_source_sha256),
    ):
        if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
            raise LoadPreflightError("Pinned " + key + " SHA-256 is required")
    if type(expected_rows) is not int or not 1 <= expected_rows <= MAX_NDJSON_ROWS:
        raise LoadPreflightError("Expected row count outside allowed bounds")
    path = Path(ndjson_path)
    if not path.is_file() or path.is_symlink():
        raise LoadPreflightError("NDJSON must be a regular non-symlink file")
    if not 0 < path.stat().st_size <= MAX_NDJSON_BYTES:
        raise LoadPreflightError("NDJSON exceeds size policy")
    schema = _schema(project, prefix, location, table)
    digest = hashlib.sha256()
    seen = set()
    rows = 0
    source_set = set()
    run_set = set()
    try:
        with path.open("rb") as source:
            for raw in source:
                digest.update(raw)
                rows += 1
                if rows > expected_rows:
                    raise LoadPreflightError("More rows than expected")
                if not raw.endswith(b"\n"):
                    raise LoadPreflightError("NDJSON record must end with newline")
                line = raw.decode("utf-8")
                item = _checked_line(line, schema["fields"])
                source_set.add(item["source_sha256"])
                run_set.add(item["ingestion_run_id"])
                if table == "ai4i_records":
                    unique = item["udi"]
                elif table == "fd001_cycles":
                    unique = (item["split"], item["unit_number"], item["time_cycles"])
                else:
                    unique = item["unit_number"]
                if unique in seen:
                    raise LoadPreflightError("Duplicate natural key in NDJSON batch")
                seen.add(unique)
                if table == "fd001_cycles":
                    if (
                        item["split"] not in ("train", "test")
                        or item["unit_number"] <= 0
                        or item["time_cycles"] <= 0
                    ):
                        raise LoadPreflightError("Invalid FD001 cycle or split")
                if table == "fd001_official_test_truth":
                    if item["unit_number"] <= 0 or item["true_rul_cycles"] < 0:
                        raise LoadPreflightError("Invalid official RUL truth")
                if table == "ai4i_records":
                    flags = ("twf", "hdf", "pwf", "osf", "rnf")
                    if (
                        item["udi"] <= 0
                        or item["machine_failure"] not in (0, 1)
                        or item["type"] not in ("L", "M", "H")
                        or not item["product_id"]
                        or any(item[field] not in (0, 1) for field in flags)
                    ):
                        raise LoadPreflightError("Invalid AI4I identifiers or labels")
    except (UnicodeError, OSError) as exc:
        raise LoadPreflightError("NDJSON cannot be read safely") from exc
    if digest.hexdigest() != expected_file_sha256:
        raise LoadPreflightError("NDJSON file checksum mismatch")
    if rows != expected_rows:
        raise LoadPreflightError("NDJSON row count mismatch")
    if source_set != {expected_source_sha256}:
        raise LoadPreflightError("NDJSON contains unexpected source identity")
    if len(run_set) != 1:
        raise LoadPreflightError("Mixed ingestion run IDs")
    run_id = next(iter(run_set))
    if not isinstance(run_id, str) or not run_id:
        raise LoadPreflightError("missing ingestion run ID")
    if not RUN_PATTERN.fullmatch(run_id):
        raise LoadPreflightError("Invalid ingestion run ID")
    return {
        "status": "offline_preflight_pass",
        "table": table,
        "planned_target": (
            schema["plan"]["project"] + "." + schema["plan"]["datasets"][0] + "." + table
        ),
        "verified_file_sha256": expected_file_sha256,
        "verified_source_sha256": expected_source_sha256,
        "rows": rows,
        "bytes": path.stat().st_size,
        "ingestion_run_id": run_id,
        "provider_dry_run_completed": False,
        "remote_load_job_submitted": False,
        "cloud_upload_performed": False,
        "cloud_cost_authorized": False,
    }
