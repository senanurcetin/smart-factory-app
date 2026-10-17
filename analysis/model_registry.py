"""Offline model artifact registry and manual promotion decision gates.

No model deserialization, retraining, network calls, production substitutions,
or automatic promotion. Only explicit stage_candidate calls write files.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

MODEL_NAMES = {
    "ai4i_failure": "model.pkl",
    "cmapss_fd001_rul": "rul_model.pkl",
}
PRIMARY_METRICS = {
    "ai4i_failure": ("pr_auc", "higher"),
    "cmapss_fd001_rul": ("rmse", "lower"),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MANIFEST_FIELDS = frozenset(
    (
        "schema_version",
        "model_id",
        "relative_path",
        "model_sha256",
        "dataset_sha256",
        "evaluated_at",
        "metrics",
        "status",
    )
)


class RegistryError(ValueError):
    """Invalid or untrusted model registry input."""


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RegistryError("Artifact must be a regular non-symlink file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise RegistryError("Artifact file size outside allowed bounds")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metrics(model_id: str, metrics: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise RegistryError("metrics must be a nonempty mapping")
    validated = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", key):
            raise RegistryError("Invalid metric name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RegistryError("Metrics must be numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RegistryError("Metrics must be finite")
        validated[key] = numeric
    primary, _direction = PRIMARY_METRICS[model_id]
    if primary not in validated:
        raise RegistryError("Required primary benchmark metric missing")
    return dict(sorted(validated.items()))


def _evaluated_at(value: str) -> str:
    if not isinstance(value, str):
        raise RegistryError("evaluated_at must be a timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RegistryError("Invalid evaluation timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RegistryError("Evaluation timestamp must contain timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _registry_dir(root: Path, model_id: str) -> Path:
    if model_id not in MODEL_NAMES:
        raise RegistryError("Unsupported model_id")
    return root / "analysis" / "artifacts" / "registry" / model_id


def stage_candidate(
    repo_root: Path,
    model_id: str,
    *,
    dataset_sha256: str,
    metrics: Mapping[str, float],
    evaluated_at: str,
) -> dict:
    """Copy an explicitly selected local model to a content-addressed candidate.

    This does NOT mark a release as approved or modify the active model file.
    The source is limited to the repository's known AI4I/C-MAPSS file names.
    """
    root = Path(repo_root).resolve()
    folder = _registry_dir(root, model_id)
    if not isinstance(dataset_sha256, str) or not SHA256_RE.fullmatch(dataset_sha256):
        raise RegistryError("dataset_sha256 must be a SHA-256 hex digest")
    checked_metrics = _metrics(model_id, metrics)
    checked_time = _evaluated_at(evaluated_at)
    source = root / "analysis" / "artifacts" / MODEL_NAMES[model_id]
    model_sha256 = _sha256(source)
    name = model_sha256 + ".pkl"
    relative_path = "analysis/artifacts/registry/" + model_id + "/" + name
    destination = folder / name
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise RegistryError("Registry folder must not be a symlink")
    if destination.exists():
        if _sha256(destination) != model_sha256:
            raise RegistryError("Existing immutable candidate is corrupted")
    else:
        # Write complete bytes in the same directory, then atomically publish.
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".candidate-", delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                with source.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        temp.write(block)
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "relative_path": relative_path,
        "model_sha256": model_sha256,
        "dataset_sha256": dataset_sha256,
        "evaluated_at": checked_time,
        "metrics": checked_metrics,
        "status": "candidate",
    }
    verify_candidate(root, manifest)
    return manifest


def verify_candidate(repo_root: Path, manifest: Mapping[str, object]) -> dict:
    """Verify manifest and immutable bytes without unpickling untrusted models."""
    if not isinstance(manifest, Mapping) or set(manifest) != MANIFEST_FIELDS:
        raise RegistryError("Invalid candidate manifest fields")
    model_id = manifest["model_id"]
    if not isinstance(model_id, str) or model_id not in MODEL_NAMES:
        raise RegistryError("Unknown model_id")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise RegistryError("Unsupported manifest schema")
    if manifest["status"] != "candidate":
        raise RegistryError("Only candidate artifacts can be verified here")
    sha = manifest["model_sha256"]
    dataset_sha = manifest["dataset_sha256"]
    if (
        not isinstance(sha, str)
        or not SHA256_RE.fullmatch(sha)
        or not isinstance(dataset_sha, str)
        or not SHA256_RE.fullmatch(dataset_sha)
    ):
        raise RegistryError("Invalid SHA-256 digest")
    expected = "analysis/artifacts/registry/" + model_id + "/" + sha + ".pkl"
    if manifest["relative_path"] != expected:
        raise RegistryError("Unexpected model path")
    if _evaluated_at(manifest["evaluated_at"]) != manifest["evaluated_at"]:
        raise RegistryError("Timestamp not normalized to UTC")
    if _metrics(model_id, manifest["metrics"]) != manifest["metrics"]:
        raise RegistryError("Invalid metrics")
    root = Path(repo_root).resolve()
    folder = _registry_dir(root, model_id)
    if folder.is_symlink():
        raise RegistryError("Registry directory cannot be a symlink")
    if _sha256(root / expected) != sha:
        raise RegistryError("Model bytes do not match manifest SHA-256")
    return dict(manifest)


def assess_candidate(
    repo_root: Path, baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict:
    """Read-only comparison; even an improvement requires manual approval."""
    before = verify_candidate(repo_root, baseline)
    after = verify_candidate(repo_root, candidate)
    if before["model_id"] != after["model_id"]:
        raise RegistryError("Cannot compare different model families")
    metric, direction = PRIMARY_METRICS[before["model_id"]]
    old = before["metrics"][metric]
    new = after["metrics"][metric]
    improved = new > old if direction == "higher" else new < old
    return {
        "model_id": before["model_id"],
        "primary_metric": metric,
        "baseline_value": old,
        "candidate_value": new,
        "status": "manual_review_required" if improved else "rejected_no_improvement",
        "automatic_promotion": False,
    }
