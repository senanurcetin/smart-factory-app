"""Read-only, reproducible weighted ECE audit of the committed AI4I model.

Example (download the official UCI AI4I CSV yourself):
    python -m analysis.audit_ai4i_calibration --csv ai4i2020.csv

Only trusted, SHA-256 pinned local inputs are loaded. Never load arbitrary
joblib/pickle files: the format can execute code during deserialization.
The output contains aggregate metrics and bin counts, not row-level scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from analysis.calibration_metrics import quantile_calibration
from analysis.run_ai4i_case_study import (
    MODEL_FEATURES_ENHANCED,
    RANDOM_SEED,
    add_derived_features,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "analysis" / "artifacts" / "model.pkl"
EXPECTED_MODEL_SHA256 = "41be2060a21f4c457c8e2e8e9e505f489d5e482d4ac3bf2939ef026ac452faaa"
EXPECTED_CSV_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(csv_path: Path) -> dict:
    if sha256(csv_path) != EXPECTED_CSV_SHA256:
        raise ValueError("Official AI4I CSV SHA-256 does not match the audited input")
    if sha256(MODEL_PATH) != EXPECTED_MODEL_SHA256:
        raise ValueError("Committed AI4I model SHA-256 changed; a new audit is required")

    data = add_derived_features(pd.read_csv(csv_path))
    x = data[MODEL_FEATURES_ENHANCED]
    y = data["Machine failure"]
    _, test_x, _, test_y = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    if len(data) != 10000 or len(test_y) != 2000 or int(test_y.sum()) != 68:
        raise ValueError("Expected source size / holdout cohort changed")
    with warnings.catch_warnings(record=True) as records:
        model = joblib.load(MODEL_PATH)
    scores = np.asarray(model.predict_proba(test_x)[:, 1], dtype=float)
    labels = np.asarray(test_y, dtype=int)
    bins = quantile_calibration(labels, scores, n_bins=10)
    binary = (scores >= 0.5).astype(int)
    result = {
        "source_model_sha256": EXPECTED_MODEL_SHA256,
        "source_csv_sha256": EXPECTED_CSV_SHA256,
        "scikit_learn_runtime": sklearn.__version__,
        "cross_version_unpickle_warnings": len(records),
        "holdout_rows": len(labels),
        "holdout_failures": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, binary).tolist(),
        "calibration": bins,
    }
    if not (
        np.isclose(result["roc_auc"], 0.9874292108147608, atol=0.000001)
        and np.isclose(result["average_precision"], 0.9018994321689301, atol=0.000001)
        and result["confusion_matrix"] == [[1928, 4], [17, 51]]
        and np.isclose(bins["sample_weighted_ece"], 0.006079623981595835, atol=0.000001)
        and sum(bins["bin_counts"]) == 2000
    ):
        raise ValueError("Recomputed predictions do not match historical evidence")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Official AI4I2020 CSV")
    args = parser.parse_args()
    print(json.dumps(run(args.csv), indent=2))
