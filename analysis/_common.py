"""Small helpers shared by the AI4I and C-MAPSS case-study pipelines."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import shap
import sklearn


def to_float(value: float) -> float:
    return round(float(value), 4)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model_card(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_reference: str,
    random_seed: int,
) -> dict:
    """Minimal model card: enough to know when/how/on-what-data this model was
    trained without standing up a full model registry for a portfolio project.
    """
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "shap_version": shap.__version__,
        "dataset": {
            "name": dataset_name,
            "reference": dataset_reference,
            "rows": int(len(df)),
            "sha256": dataset_hash,
        },
        "random_seed": random_seed,
    }
