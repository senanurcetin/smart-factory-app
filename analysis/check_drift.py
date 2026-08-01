"""Data-drift check for the AI4I feature space (Population Stability Index).

This demonstrates, rather than just documents, the "drift monitoring" idea
from the README's Engineering Decisions / Production Considerations
sections: it computes PSI between the AI4I training distribution (the
"reference") and a synthetic "live" batch with one feature deliberately
shifted — tool wear accumulating faster than during the training window, a
realistic fleet-aging scenario — so the report shows a genuine catch, not
just "no drift" on data resampled from itself.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from analysis._common import to_float, write_json
    from analysis.run_ai4i_case_study import (
        CATEGORICAL_FEATURES,
        DERIVED_FEATURES,
        NUMERIC_FEATURES,
        OUTPUT_DIR,
        RANDOM_SEED,
        add_derived_features,
        load_dataset,
    )
except ImportError:
    from _common import to_float, write_json
    from run_ai4i_case_study import (
        CATEGORICAL_FEATURES,
        DERIVED_FEATURES,
        NUMERIC_FEATURES,
        OUTPUT_DIR,
        RANDOM_SEED,
        add_derived_features,
        load_dataset,
    )

logger = logging.getLogger(__name__)

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.2
NUM_BUCKETS = 10
SHIFTED_FEATURE = "Tool wear [min]"
SHIFT_MINUTES = 80.0
LIVE_BATCH_SIZE = 1000


def _psi_from_counts(
    ref_counts: np.ndarray,
    cur_counts: np.ndarray,
    ref_total: int,
    cur_total: int,
    eps: float = 1e-4,
) -> float:
    ref_pct = np.clip(ref_counts / max(ref_total, 1), eps, None)
    cur_pct = np.clip(cur_counts / max(cur_total, 1), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_numeric_psi(
    reference: pd.Series, current: pd.Series, buckets: int = NUM_BUCKETS
) -> float:
    """PSI over quantile-binned buckets of the reference distribution."""
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0
    breakpoints = np.unique(reference.quantile(np.linspace(0, 1, buckets + 1)).values).astype(float)
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    return _psi_from_counts(ref_counts, cur_counts, len(reference), len(current))


def compute_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category frequencies instead of quantile buckets."""
    categories = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref_counts = np.array([(reference == c).sum() for c in categories])
    cur_counts = np.array([(current == c).sum() for c in categories])
    return _psi_from_counts(ref_counts, cur_counts, len(reference), len(current))


def verdict(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant_drift"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate_drift"
    return "no_significant_drift"


def build_synthetic_live_batch(
    reference_df: pd.DataFrame, n: int = LIVE_BATCH_SIZE
) -> pd.DataFrame:
    """Simulate a 'live' telemetry batch: resampled from the same distribution
    as training (so most features should show ~zero drift), with one feature
    deliberately shifted to prove the check actually catches something.
    """
    batch = reference_df.sample(n=n, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)
    batch[SHIFTED_FEATURE] = batch[SHIFTED_FEATURE] + SHIFT_MINUTES
    return add_derived_features(batch)


def build_drift_report(reference_df: pd.DataFrame, live_batch: pd.DataFrame) -> dict:
    feature_reports = []
    for col in NUMERIC_FEATURES + DERIVED_FEATURES:
        psi = compute_numeric_psi(reference_df[col], live_batch[col])
        feature_reports.append({"feature": col, "psi": to_float(psi), "verdict": verdict(psi)})
    for col in CATEGORICAL_FEATURES:
        psi = compute_categorical_psi(reference_df[col], live_batch[col])
        feature_reports.append({"feature": col, "psi": to_float(psi), "verdict": verdict(psi)})

    feature_reports.sort(key=lambda row: row["psi"], reverse=True)
    drifted = [row for row in feature_reports if row["verdict"] != "no_significant_drift"]

    return {
        "method": (
            "Population Stability Index (PSI): quantile-binned (10 buckets) for numeric "
            "features, category-frequency for categorical features"
        ),
        "thresholds": {
            "no_significant_drift": f"PSI < {PSI_MODERATE_THRESHOLD}",
            "moderate_drift": f"{PSI_MODERATE_THRESHOLD} <= PSI < {PSI_SIGNIFICANT_THRESHOLD}",
            "significant_drift": f"PSI >= {PSI_SIGNIFICANT_THRESHOLD}",
        },
        "scenario": (
            f"Synthetic 'live' batch resampled from the training distribution, with "
            f"'{SHIFTED_FEATURE}' deliberately shifted by +{SHIFT_MINUTES} minutes to "
            "simulate a fleet whose tools are wearing faster than during the training "
            "window. A realistic single-feature drift scenario, not an artificial "
            "all-features shift, so most features below should show ~zero PSI."
        ),
        "reference_rows": int(len(reference_df)),
        "live_batch_rows": int(len(live_batch)),
        "features": feature_reports,
        "drift_detected": len(drifted) > 0,
        "drifted_features": [row["feature"] for row in drifted],
    }


def main() -> None:
    df = load_dataset()
    reference_df = add_derived_features(df)
    live_batch = build_synthetic_live_batch(df)

    report = build_drift_report(reference_df, live_batch)
    write_json(OUTPUT_DIR / "drift-report.json", report)

    logger.info(f"Wrote drift report to {OUTPUT_DIR / 'drift-report.json'}")
    if report["drift_detected"]:
        logger.info(f"  Drift detected in: {', '.join(report['drifted_features'])}")
    else:
        logger.info("  No drift detected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
