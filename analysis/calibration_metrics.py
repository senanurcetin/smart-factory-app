"""Population-weighted quantile calibration diagnostics.

Unlike an unweighted average of occupied reliability-bin gaps, ECE weights
each bin by its sample count. Duplicate percentile cutpoints are preserved,
matching sklearn.calibration.calibration_curve(strategy="quantile").
This is a descriptive retrospective statistic, not proof of prospective
probability calibration.
"""

from __future__ import annotations

import numpy as np


def quantile_calibration(y_true, probabilities, n_bins: int = 10) -> dict:
    """Compute sample-weighted ECE and per-bin diagnostics without refitting."""
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    labels = np.asarray(y_true, dtype=float).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Nonempty labels and probabilities must have equal length")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError("Labels and probabilities must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1)")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be within [0,1]")

    # Do not np.unique() the percentile edges: duplicate cutoffs are part of
    # sklearn's exact quantile bin assignment when predictions have ties.
    edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    bin_ids = np.searchsorted(edges[1:-1], scores)
    counts = np.bincount(bin_ids, minlength=n_bins)
    positives = np.bincount(bin_ids, weights=labels, minlength=n_bins)
    predicted = np.bincount(bin_ids, weights=scores, minlength=n_bins)
    occupied = counts > 0
    bin_counts = counts[occupied]
    observed_rates = positives[occupied] / bin_counts
    predicted_means = predicted[occupied] / bin_counts
    gaps = np.abs(observed_rates - predicted_means)
    weighted_ece = float(np.dot(bin_counts, gaps) / len(scores))
    return {
        "requested_bins": n_bins,
        "occupied_bins": int(np.count_nonzero(occupied)),
        "sample_count": len(scores),
        "sample_weighted_ece": weighted_ece,
        "legacy_unweighted_bin_gap": float(gaps.mean()),
        "bin_counts": [int(value) for value in bin_counts],
        "reliability_bins": [
            {
                "count": int(count),
                "mean_prediction": float(prediction),
                "observed_rate": float(observation),
                "absolute_gap": float(gap),
            }
            for count, prediction, observation, gap in zip(
                bin_counts, predicted_means, observed_rates, gaps
            )
        ],
    }
