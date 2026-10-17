"""Offline, checksum-pinned NASA C-MAPSS FD001 BigQuery staging.

Reads separately verified official train/test/truth text files. Never
downloads, trains, calls cloud APIs or invents RUL truth for interim cycles.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from analysis.ai4i_batch import RUN_PATTERN

RAW_FIELDS = (
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{i}" for i in range(1, 22)),
)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_CYCLES = 100000
MAX_UNITS = 1000
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INTEGER_PATTERN = re.compile(r"^[0-9]+$")


class Fd001BatchError(ValueError):
    """Source integrity, cycle ordering or official truth mismatch."""


def _pinned_file(path: Path, expected_sha: str) -> tuple[str, str]:
    if not isinstance(expected_sha, str) or not SHA_PATTERN.fullmatch(expected_sha):
        raise Fd001BatchError("Each FD001 file requires a pinned SHA-256")
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Fd001BatchError("FD001 input must be a regular non-symlink file")
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise Fd001BatchError("FD001 input size exceeds policy")
    blob = path.read_bytes()
    observed = hashlib.sha256(blob).hexdigest()
    if observed != expected_sha:
        raise Fd001BatchError("FD001 file SHA-256 mismatch")
    try:
        source = blob.decode("utf-8")
    except UnicodeError as exc:
        raise Fd001BatchError("FD001 source is not UTF-8") from exc
    return source, observed


def _positive_int(token: str, field: str) -> int:
    if not INTEGER_PATTERN.fullmatch(token):
        raise Fd001BatchError(field + " must be a positive decimal integer")
    value = int(token)
    if value < 1:
        raise Fd001BatchError(field + " must be positive")
    return value


def parse_cycles(source: str, split: str) -> list[dict]:
    """Read exactly 26 FD001 columns with engine-local contiguous cycles."""
    if split not in ("train", "test"):
        raise Fd001BatchError("FD001 split must be train or test")
    rows = []
    last_cycle: dict[int, int] = {}
    previous_unit = 0
    for line in source.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(RAW_FIELDS):
            raise Fd001BatchError("FD001 record must contain exactly 26 fields")
        unit = _positive_int(fields[0], "unit_number")
        cycle = _positive_int(fields[1], "time_cycles")
        if unit != previous_unit and unit != previous_unit + 1:
            raise Fd001BatchError("FD001 engine IDs must be sequential")
        if unit == previous_unit + 1 and cycle != 1:
            raise Fd001BatchError("FD001 new engine must start at cycle 1")
        if unit in last_cycle and cycle != last_cycle[unit] + 1:
            raise Fd001BatchError("FD001 cycles must be consecutive")
        if unit > MAX_UNITS or len(rows) >= MAX_CYCLES:
            raise Fd001BatchError("FD001 row/unit count exceeds policy")
        previous_unit = unit
        last_cycle[unit] = cycle
        result = {"unit_number": unit, "time_cycles": cycle, "split": split}
        for field, token in zip(RAW_FIELDS[2:], fields[2:]):
            try:
                number = float(token)
            except ValueError as exc:
                raise Fd001BatchError("Non-numeric FD001 field: " + field) from exc
            if not math.isfinite(number):
                raise Fd001BatchError("Nonfinite FD001 field: " + field)
            result[field] = number
        rows.append(result)
    if not rows or len(last_cycle) > MAX_UNITS:
        raise Fd001BatchError("Empty or invalid FD001 cycles")
    return rows


def parse_official_truth(source: str, test_unit_count: int) -> list[int]:
    """NASA official truth gives ONE RUL value per held-out test engine."""
    truth = []
    for line in source.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 1 or not INTEGER_PATTERN.fullmatch(fields[0]):
            raise Fd001BatchError("FD001 truth must contain one nonnegative integer per line")
        truth.append(int(fields[0]))
    if len(truth) != test_unit_count:
        raise Fd001BatchError("FD001 truth count differs from held-out test engines")
    return truth


def prepare_fd001_batches(
    train_path: Path,
    test_path: Path,
    truth_path: Path,
    *,
    expected_sha256: dict[str, str],
    run_id: str,
    ingested_at: datetime,
    expected_train_rows: int = 20631,
    expected_train_units: int = 100,
    expected_test_units: int = 100,
) -> dict:
    """Normalize benchmark files without leaking train labels into test truth."""
    if not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise Fd001BatchError("Invalid ingestion run ID")
    if (
        not isinstance(ingested_at, datetime)
        or ingested_at.tzinfo is None
        or ingested_at.utcoffset() is None
    ):
        raise Fd001BatchError("ingested_at must have a timezone")
    if not isinstance(expected_sha256, dict) or set(expected_sha256) != {
        "train", "test", "truth"
    }:
        raise Fd001BatchError("Must pin all three official source file digests")
    if (
        type(expected_train_rows) is not int
        or not 1 <= expected_train_rows <= MAX_CYCLES
        or type(expected_train_units) is not int
        or not 1 <= expected_train_units <= MAX_UNITS
        or type(expected_test_units) is not int
        or not 1 <= expected_test_units <= MAX_UNITS
    ):
        raise Fd001BatchError("Invalid expected FD001 dataset dimensions")
    file_sources = {}
    digests = {}
    for key, path in (
        ("train", train_path),
        ("test", test_path),
        ("truth", truth_path),
    ):
        file_sources[key], digests[key] = _pinned_file(path, expected_sha256[key])
    train = parse_cycles(file_sources["train"], "train")
    test = parse_cycles(file_sources["test"], "test")
    if len(train) != expected_train_rows:
        raise Fd001BatchError("FD001 train row count differs from benchmark")
    train_units = len({row["unit_number"] for row in train})
    test_units = len({row["unit_number"] for row in test})
    if train_units != expected_train_units or test_units != expected_test_units:
        raise Fd001BatchError("FD001 train/test engine count differs from benchmark")
    truth = parse_official_truth(file_sources["truth"], test_units)
    # A bundle identity provides the same join key for cycles and the separate
    # RUL truth file without pretending their raw-file SHA-256 hashes match.
    bundle_sha = hashlib.sha256(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    timestamp = ingested_at.astimezone(timezone.utc).isoformat()
    for row in train + test:
        row["source_sha256"] = bundle_sha
        row["ingestion_run_id"] = run_id
        row["ingested_at"] = timestamp
    truth_rows = [
        {
            "source_sha256": bundle_sha,
            "ingestion_run_id": run_id,
            "unit_number": index,
            "true_rul_cycles": target,
            "ingested_at": timestamp,
        }
        for index, target in enumerate(truth, start=1)
    ]
    return {
        "cycles": train + test,
        "official_test_truth": truth_rows,
        "evidence": {
            "status": "local_verified_only",
            "bundle_source_sha256": bundle_sha,
            "raw_file_sha256": digests,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_units": train_units,
            "test_units": test_units,
            "truth_labels": len(truth_rows),
            "official_truth_applies_to": "last_observed_test_cycle_only",
            "cloud_upload_performed": False,
        },
    }


def write_fd001_jsonl(batches: dict, destination: Path) -> dict:
    """Atomically publish two bounded local NDJSON files, never overwriting."""
    if not isinstance(batches, dict) or set(batches) != {
        "cycles", "official_test_truth", "evidence"
    }:
        raise Fd001BatchError("Invalid prepared batch shape")
    root = Path(destination)
    if root.is_symlink() or not root.is_dir():
        raise Fd001BatchError("Destination must be an existing regular directory")
    exports = {
        "fd001_cycles.jsonl": batches["cycles"],
        "fd001_official_test_truth.jsonl": batches["official_test_truth"],
    }
    if any((root / name).exists() or (root / name).is_symlink() for name in exports):
        raise Fd001BatchError("FD001 export already exists")
    written = []
    try:
        for name, data in exports.items():
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=root, prefix=".fd001-", delete=False
            ) as stream:
                temp_path = Path(stream.name)
                try:
                    for record in data:
                        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                except BaseException:
                    temp_path.unlink(missing_ok=True)
                    raise
            try:
                os.link(temp_path, root / name)
                written.append(root / name)
            except FileExistsError as exc:
                raise Fd001BatchError("FD001 export already exists") from exc
            finally:
                temp_path.unlink(missing_ok=True)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {
        "status": "local_jsonl_only",
        "export_paths": [path.name for path in written],
        "cloud_upload_performed": False,
    }
