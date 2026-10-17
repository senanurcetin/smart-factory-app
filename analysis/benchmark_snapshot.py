"""Read-only NEXUS benchmark snapshot from committed, separately scoped studies.

Never claims cloud freshness, live telemetry, model retraining or official
source archive verification. No SDK, network calls or production file writes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from analysis.source_registry import load_manifest


class SnapshotError(ValueError):
    """Committed evidence is missing, inconsistent or nonnumeric."""


def _json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise SnapshotError("Required local benchmark evidence is inaccessible")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SnapshotError("Committed benchmark evidence is invalid JSON") from exc
    if not isinstance(data, dict):
        raise SnapshotError("Expected a benchmark evidence object")
    return data


def _metric(value: object, name: str, *, maximum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise SnapshotError("Invalid benchmark metric: " + name)
    result = float(value)
    if result < 0 or maximum is not None and result > maximum:
        raise SnapshotError("Out-of-range benchmark metric: " + name)
    return result


def build_benchmark_snapshot(repo_root: Path) -> dict:
    """Build an explicitly historic, explainable NEXUS analytical snapshot."""
    root = Path(repo_root).resolve()
    manifest = load_manifest(root / "docs/data/smart-factory-source-manifest.json")
    sources = {row["id"]: row for row in manifest["datasets"]}
    ai4i_dir = root / "docs/data/ai4i-case-study"
    rul_dir = root / "docs/data/cmapss-rul-case-study"
    ai4i_summary = _json(ai4i_dir / "summary.json")
    ai4i_card = _json(ai4i_dir / "model-selection.json")["model_card"]
    rul_summary = _json(rul_dir / "summary.json")
    rul_card = _json(rul_dir / "model-selection.json")["model_card"]
    cards = {
        "uci_ai4i_2020": ai4i_card,
        "nasa_cmapss_fd001": rul_card,
    }
    for key, card in cards.items():
        source = sources[key]
        evidence = card.get("dataset")
        if not isinstance(evidence, dict):
            raise SnapshotError("No model-card dataset evidence for " + key)
        if (
            evidence.get("sha256") != source["benchmark_dataframe_sha256"]
            or evidence.get("rows") != source["expected_training_rows"]
        ):
            raise SnapshotError("Committed dataset lineage mismatch: " + key)
    ai4i_model = ai4i_summary["final_model"]
    rul_model = rul_summary["final_model"]
    ai4i_pr_auc = _metric(ai4i_model["pr_auc"], "AI4I PR-AUC", maximum=1)
    ai4i_roc = _metric(ai4i_model["roc_auc"], "AI4I ROC-AUC", maximum=1)
    rul_rmse = _metric(rul_model["rmse"], "FD001 RMSE")
    rul_phm = _metric(rul_model["phm08_score"], "FD001 PHM08")
    ai4i_holdout = ai4i_summary["dataset"]["evaluation_split"]["holdout_rows"]
    fd001_units = rul_summary["dataset"]["test_units"]
    if (
        type(ai4i_holdout) is not int
        or type(fd001_units) is not int
        or ai4i_holdout < 1
        or fd001_units < 1
    ):
        raise SnapshotError("Invalid holdout evaluation dimensions")
    return {
        "schema_version": 1,
        "brand": "NEXUS Smart Factory",
        "data_origin": "committed_offline_case_studies",
        "bigquery_dataset_connected": False,
        "kaggle_source_version_verified": False,
        "real_plant_telemetry": False,
        "model_training_performed": False,
        "sources": [
            {
                "id": source["id"],
                "model_card_dataframe_sha256": source["benchmark_dataframe_sha256"],
                "source_archive_sha256_verified": source["archive_sha256"] is not None,
                "kaggle_mirror_version_pinned": (
                    source["kaggle_dataset"] is not None
                    and source["kaggle_version"] is not None
                ),
            }
            for source in manifest["datasets"]
        ],
        "benchmarks": {
            "ai4i_failure": {
                "task": "static_snapshot_failure_classification",
                "holdout_rows": ai4i_holdout,
                "pr_auc": ai4i_pr_auc,
                "roc_auc": ai4i_roc,
                "risk_is_calibrated_probability": False,
            },
            "cmapss_fd001_rul": {
                "task": "official_last_cycle_regression",
                "heldout_test_engines": fd001_units,
                "rmse_cycles": rul_rmse,
                "phm08_score": rul_phm,
                "estimated_live_plant_rul": False,
            },
        },
        "display_disclosure": (
            "Historical offline benchmark metrics from simulated UCI AI4I and "
            "NASA C-MAPSS datasets; not live factory readings or BigQuery results."
        ),
    }
