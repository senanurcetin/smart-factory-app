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


CHART_PALETTE = {
    "bg": "#0f1929",
    "grid": "#223856",
    "text": "#edf4ff",
    "muted": "#9fb4d1",
    "primary": "#60a5fa",
    "accent": "#5eead4",
    "warn": "#fbbf24",
    "danger": "#f87171",
    "success": "#34d399",
    "highlight": "#c084fc",
    "neutral": "#64748b",
}


def apply_dark_chart_style() -> None:
    """Style matplotlib to match the site's dark/glass theme, so embedded
    charts blend into the page instead of sitting in a stark white box."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": CHART_PALETTE["bg"],
            "axes.facecolor": CHART_PALETTE["bg"],
            "axes.edgecolor": CHART_PALETTE["grid"],
            "axes.labelcolor": CHART_PALETTE["text"],
            "text.color": CHART_PALETTE["text"],
            "xtick.color": CHART_PALETTE["muted"],
            "ytick.color": CHART_PALETTE["muted"],
            "grid.color": CHART_PALETTE["grid"],
            "legend.facecolor": CHART_PALETTE["bg"],
            "legend.edgecolor": CHART_PALETTE["grid"],
            "legend.labelcolor": CHART_PALETTE["text"],
            "savefig.facecolor": CHART_PALETTE["bg"],
        }
    )


def save_chart_figure(fig, out_dir: Path, name: str) -> Path:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


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
