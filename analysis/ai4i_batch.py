"""Deterministic, offline AI4I CSV -> BigQuery JSONL adapter.

A local CSV checksum is a distinct identity from its enclosing UCI/Kaggle ZIP
and from the pandas DataFrame hash in the published AI4I model card.
There is no network access, BigQuery SDK, live plant data or cloud upload.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CSV_FIELDS = (
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)
FLOAT_FIELDS = {
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}
LABEL_FIELDS = ("Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF")
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_ROWS = 15000
SHA_PATTERN = re.compile(r"^[a-f0-9]{64}$")
RUN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Ai4iBatchError(ValueError):
    """Unsafe or inconsistent source batch for offline review."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: str, name: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise Ai4iBatchError(name + " must contain decimal digits")
    result = int(value)
    if result < 1:
        raise Ai4iBatchError(name + " must be positive")
    return result


def _bit(value: str, name: str) -> int:
    if value not in ("0", "1"):
        raise Ai4iBatchError(name + " must be 0 or 1")
    return int(value)


def prepare_ai4i_rows(
    csv_path: Path,
    *,
    expected_csv_sha256: str,
    run_id: str,
    ingested_at: datetime,
    expected_rows: int = 10000,
) -> list[dict]:
    """Read one checksum-pinned AI4I CSV with strict row and schema rules."""
    if not isinstance(expected_csv_sha256, str) or not SHA_PATTERN.fullmatch(
        expected_csv_sha256
    ):
        raise Ai4iBatchError("A verified raw CSV SHA-256 is required")
    if not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise Ai4iBatchError("Invalid ingestion run ID")
    if (
        not isinstance(ingested_at, datetime)
        or ingested_at.tzinfo is None
        or ingested_at.utcoffset() is None
    ):
        raise Ai4iBatchError("ingested_at must be a timezone-aware datetime")
    if type(expected_rows) is not int or not 1 <= expected_rows <= MAX_ROWS:
        raise Ai4iBatchError("expected_rows outside allowed bounds")
    path = Path(csv_path)
    if not path.is_file() or path.is_symlink():
        raise Ai4iBatchError("Input CSV must be a regular non-symlink file")
    size = path.stat().st_size
    if not 0 < size <= MAX_SOURCE_BYTES:
        raise Ai4iBatchError("CSV exceeds size policy or is empty")
    if _hash_file(path) != expected_csv_sha256:
        raise Ai4iBatchError("Input CSV SHA-256 mismatch")
    timestamp = ingested_at.astimezone(timezone.utc).isoformat()
    rows = []
    seen_udi = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or set(reader.fieldnames) != set(CSV_FIELDS):
                raise Ai4iBatchError("AI4I CSV fields do not match canonical schema")
            if len(reader.fieldnames) != len(CSV_FIELDS):
                raise Ai4iBatchError("Duplicate AI4I CSV column")
            for raw in reader:
                if len(rows) >= expected_rows:
                    raise Ai4iBatchError("More source records than expected")
                if None in raw or any(v is None for v in raw.values()):
                    raise Ai4iBatchError("Malformed or truncated CSV record")
                udi = _positive_int(raw["UDI"], "UDI")
                if udi in seen_udi:
                    raise Ai4iBatchError("Duplicate UDI in source CSV")
                seen_udi.add(udi)
                if raw["Type"] not in ("L", "M", "H"):
                    raise Ai4iBatchError("Unexpected AI4I product type")
                if not raw["Product ID"].strip():
                    raise Ai4iBatchError("Missing product identity")
                item = {
                    "source_sha256": expected_csv_sha256,
                    "ingestion_run_id": run_id,
                    "udi": udi,
                    "product_id": raw["Product ID"],
                    "type": raw["Type"],
                }
                for column, normalized in FLOAT_FIELDS.items():
                    try:
                        number = float(raw[column])
                    except ValueError as exc:
                        raise Ai4iBatchError("Invalid numeric feature: " + column) from exc
                    if not math.isfinite(number):
                        raise Ai4iBatchError("Nonfinite numeric feature: " + column)
                    item[normalized] = number
                for column in LABEL_FIELDS:
                    item[column.lower().replace(" ", "_")] = _bit(raw[column], column)
                item["ingested_at"] = timestamp
                rows.append(item)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Ai4iBatchError("CSV cannot be read safely") from exc
    if len(rows) != expected_rows:
        raise Ai4iBatchError("AI4I CSV row count does not match expected count")
    if _hash_file(path) != expected_csv_sha256:
        raise Ai4iBatchError("CSV changed while it was being validated")
    return rows


def write_new_jsonl(rows: list[dict], path: Path) -> dict:
    """Publish JSONL create-only; never overwrite an existing batch export."""
    if not isinstance(rows, list) or not 0 < len(rows) <= MAX_ROWS:
        raise Ai4iBatchError("Empty or oversized batch")
    output = Path(path)
    if output.is_symlink() or output.exists():
        raise Ai4iBatchError("Export destination already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise Ai4iBatchError("Export parent must be an existing regular directory")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=".ai4i-", delete=False
    ) as temp:
        temp_path = Path(temp.name)
        try:
            for item in rows:
                temp.write(json.dumps(item, allow_nan=False, sort_keys=True) + "\n")
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.link(temp_path, output)
    except FileExistsError as exc:
        raise Ai4iBatchError("Export destination already exists") from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return {
        "status": "local_jsonl_only",
        "records": len(rows),
        "export_sha256": _hash_file(output),
        "cloud_upload_performed": False,
    }
