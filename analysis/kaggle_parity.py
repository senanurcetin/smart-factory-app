"""Offline UCI AI4I vs candidate Kaggle mirror row-level parity.

No Kaggle API calls, downloaded datasets, cloud writes or license grants.
A byte-level mismatch can be acceptable if normalized rows are identical,
but source identity and benchmark integrity must be pinned separately.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from analysis.ai4i_batch import Ai4iBatchError, prepare_ai4i_rows

KAGGLE_HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
ROW_METADATA = frozenset(("source_sha256", "ingestion_run_id", "ingested_at"))


class KaggleParityError(ValueError):
    """An unverified Kaggle mirror cannot be accepted as benchmark-equivalent."""


def _canonical(rows: list[dict]) -> list[dict]:
    return [
        {name: value for name, value in row.items() if name not in ROW_METADATA}
        for row in sorted(rows, key=lambda entry: entry["udi"])
    ]


def verify_ai4i_mirror(
    official_csv: Path,
    mirror_csv: Path,
    *,
    official_sha256: str,
    mirror_sha256: str,
    kaggle_handle: str,
    kaggle_version: int,
    license_reviewed: bool = False,
    expected_rows: int = 10000,
) -> dict:
    """Compare *all* normalized fields and failure labels on strict UDI keys."""
    if not isinstance(kaggle_handle, str) or not KAGGLE_HANDLE_PATTERN.fullmatch(kaggle_handle):
        raise KaggleParityError("Kaggle owner/dataset handle is required")
    if type(kaggle_version) is not int or kaggle_version < 1:
        raise KaggleParityError("Explicit positive Kaggle version is required")
    if type(license_reviewed) is not bool:
        raise KaggleParityError("License review flag must be boolean")
    if Path(official_csv).resolve() == Path(mirror_csv).resolve():
        raise KaggleParityError("Official and mirror input paths must be independent")
    try:
        official_rows = prepare_ai4i_rows(
            official_csv,
            expected_csv_sha256=official_sha256,
            run_id="official_source",
            ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            expected_rows=expected_rows,
        )
        mirror_rows = prepare_ai4i_rows(
            mirror_csv,
            expected_csv_sha256=mirror_sha256,
            run_id="candidate_mirror",
            ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            expected_rows=expected_rows,
        )
    except Ai4iBatchError as exc:
        raise KaggleParityError("Official or Kaggle mirror failed source checks") from exc
    if _canonical(official_rows) != _canonical(mirror_rows):
        raise KaggleParityError("Kaggle AI4I rows differ from official source")
    contents = (
        "uci_ai4i_2020\n"
        + official_sha256
        + "\n"
        + mirror_sha256
        + "\n"
        + kaggle_handle
        + "\n"
        + str(kaggle_version)
    ).encode("utf-8")
    return {
        "dataset_id": "uci_ai4i_2020",
        "status": "offline_normalized_rows_identical",
        "official_csv_sha256": official_sha256,
        "mirror_csv_sha256": mirror_sha256,
        "parity_evidence_sha256": hashlib.sha256(contents).hexdigest(),
        "kaggle_handle": kaggle_handle,
        "kaggle_version": kaggle_version,
        "compared_rows": len(official_rows),
        "label_parity_checked": True,
        "license_reviewed": license_reviewed,
        "provider_dataset_version_verified": False,
        "cloud_ingestion_authorized": False,
        "cloud_upload_performed": False,
    }
