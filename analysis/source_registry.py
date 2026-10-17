"""Fail-closed source identity checks for offline AI4I/FD001 batch pipelines.

No downloads, Kaggle API requests, Google Cloud calls or database writes.
Model-card DataFrame fingerprints are NOT treated as raw archive hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 65536
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DATASET_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
KAGGLE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
MANIFEST_FIELDS = frozenset(("schema_version", "description", "datasets"))
DATASET_FIELDS = frozenset(
    (
        "id",
        "official_source",
        "official_url",
        "citation",
        "data_role",
        "expected_training_rows",
        "benchmark_dataframe_sha256",
        "archive_sha256",
        "kaggle_dataset",
        "kaggle_version",
        "license_reviewed",
    )
)
EXPECTED_DATASETS = {
    "uci_ai4i_2020": "failure_classification",
    "nasa_cmapss_fd001": "rul_regression",
}
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "data"
    / "smart-factory-source-manifest.json"
)


class SourceRegistryError(ValueError):
    """Untrusted, incomplete or incompatible dataset provenance."""


def _no_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceRegistryError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _sha256(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise SourceRegistryError(name + " must be a lowercase SHA-256 digest")


def validate_manifest(manifest: object) -> dict:
    """Validate structure and source claims, not the authenticity of a provider."""
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise SourceRegistryError("Invalid source manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise SourceRegistryError("Unsupported source manifest schema")
    if not isinstance(manifest["description"], str) or not manifest["description"].strip():
        raise SourceRegistryError("Manifest description is required")
    datasets = manifest["datasets"]
    if not isinstance(datasets, list) or len(datasets) != len(EXPECTED_DATASETS):
        raise SourceRegistryError("Both canonical benchmark datasets are required")
    seen = set()
    for record in datasets:
        if not isinstance(record, dict) or set(record) != DATASET_FIELDS:
            raise SourceRegistryError("Invalid dataset entry fields")
        dataset_id = record["id"]
        if not isinstance(dataset_id, str) or not DATASET_PATTERN.fullmatch(dataset_id):
            raise SourceRegistryError("Invalid dataset ID")
        if dataset_id in seen or dataset_id not in EXPECTED_DATASETS:
            raise SourceRegistryError("Duplicate or unknown dataset ID")
        seen.add(dataset_id)
        if record["data_role"] != EXPECTED_DATASETS[dataset_id]:
            raise SourceRegistryError("Model/data-role mismatch")
        for key in ("official_source", "citation"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise SourceRegistryError("Missing source metadata: " + key)
        url = record["official_url"]
        if not isinstance(url, str):
            raise SourceRegistryError("Missing official source URL")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise SourceRegistryError("Official source URL must be plain HTTPS")
        count = record["expected_training_rows"]
        if type(count) is not int or count < 1:
            raise SourceRegistryError("Training row count must be positive")
        _sha256(record["benchmark_dataframe_sha256"], "benchmark_dataframe_sha256")
        _sha256(record["archive_sha256"], "archive_sha256", nullable=True)
        kaggle = record["kaggle_dataset"]
        version = record["kaggle_version"]
        if kaggle is None and version is not None:
            raise SourceRegistryError("Kaggle version requires a dataset handle")
        if kaggle is not None and (
            not isinstance(kaggle, str) or not KAGGLE_PATTERN.fullmatch(kaggle)
        ):
            raise SourceRegistryError("Invalid Kaggle owner/dataset handle")
        if version is not None and (type(version) is not int or version < 1):
            raise SourceRegistryError("Kaggle version must be a positive integer")
        if type(record["license_reviewed"]) is not bool:
            raise SourceRegistryError("license_reviewed must be an explicit boolean")
    if seen != set(EXPECTED_DATASETS):
        raise SourceRegistryError("Canonical benchmark dataset missing")
    return manifest


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    """Load bounded JSON without allowing duplicate-key provenance overrides."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SourceRegistryError("Manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if not 0 < size <= MAX_MANIFEST_BYTES:
        raise SourceRegistryError("Manifest size outside allowed bounds")
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_json_keys
        )
    except (UnicodeError, OSError, json.JSONDecodeError) as exc:
        raise SourceRegistryError("Manifest is not valid UTF-8 JSON") from exc
    return validate_manifest(manifest)


def source_status(manifest: dict) -> list[dict]:
    """Be explicit about missing checksums, mirrors and license review."""
    validate_manifest(manifest)
    return [
        {
            "id": item["id"],
            "archive_checksum_pinned": item["archive_sha256"] is not None,
            "kaggle_version_pinned": (
                item["kaggle_dataset"] is not None and item["kaggle_version"] is not None
            ),
            "license_reviewed": item["license_reviewed"],
            "cloud_ingestion_authorized": False,
        }
        for item in manifest["datasets"]
    ]


def verify_local_archive(manifest: dict, dataset_id: str, path: Path) -> dict:
    """Hash already downloaded bytes; refuse unpinned archives, never extract."""
    validate_manifest(manifest)
    item = next((d for d in manifest["datasets"] if d["id"] == dataset_id), None)
    if item is None:
        raise SourceRegistryError("Unknown dataset")
    expected = item["archive_sha256"]
    if expected is None:
        raise SourceRegistryError("Archive SHA-256 must be pinned before verification")
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise SourceRegistryError("Archive must be a regular non-symlink file")
    size = path.stat().st_size
    if not 0 < size <= MAX_ARCHIVE_BYTES:
        raise SourceRegistryError("Archive size outside allowed bounds")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise SourceRegistryError("Archive SHA-256 mismatch")
    return {
        "dataset_id": dataset_id,
        "verified_archive_sha256": observed,
        "archive_bytes": size,
        "benchmark_dataframe_sha256": item["benchmark_dataframe_sha256"],
        "matches_benchmark_dataframe_hash": None,
        "source_provenance": "local_bytes_verified_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline dataset source provenance")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify", nargs=2, metavar=("DATASET_ID", "LOCAL_ARCHIVE"))
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        result = (
            verify_local_archive(manifest, args.verify[0], Path(args.verify[1]))
            if args.verify
            else source_status(manifest)
        )
    except SourceRegistryError as exc:
        parser.exit(2, "SOURCE VERIFICATION BLOCKED: " + str(exc) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
