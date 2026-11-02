"""Unit regression tests for AI4I sample-weighted reliability diagnostics."""

import unittest

import numpy as np
from sklearn.calibration import calibration_curve

from analysis.calibration_metrics import quantile_calibration


class QuantileCalibrationTest(unittest.TestCase):
    def test_uneven_bin_weights_differ_from_unweighted_gap(self):
        result = quantile_calibration([0, 0, 1, 1], [0.1, 0.1, 0.1, 0.9], n_bins=2)
        self.assertEqual(result["bin_counts"], [3, 1])
        self.assertAlmostEqual(result["sample_weighted_ece"], 0.2)
        self.assertAlmostEqual(result["legacy_unweighted_bin_gap"], 1 / 6)
        self.assertNotAlmostEqual(
            result["sample_weighted_ece"], result["legacy_unweighted_bin_gap"]
        )

    def test_reliability_matches_sklearn_quantile_implementation(self):
        labels = [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
        scores = [0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.22, 0.39, 0.41, 0.7, 0.75, 0.91]
        report = quantile_calibration(labels, scores, n_bins=5)
        fraction_pos, mean_pred = calibration_curve(
            labels, scores, n_bins=5, strategy="quantile"
        )
        self.assertEqual(report["occupied_bins"], len(fraction_pos))
        self.assertEqual(sum(report["bin_counts"]), len(labels))
        np.testing.assert_allclose(
            [b["observed_rate"] for b in report["reliability_bins"]], fraction_pos
        )
        np.testing.assert_allclose(
            [b["mean_prediction"] for b in report["reliability_bins"]], mean_pred
        )
        weighted = sum(
            b["count"] * abs(b["observed_rate"] - b["mean_prediction"])
            for b in report["reliability_bins"]
        ) / len(scores)
        self.assertAlmostEqual(report["sample_weighted_ece"], weighted)

    def test_rejects_invalid_input(self):
        cases = [
            ([], [], 10), ([0, 1], [0.2], 10),
            ([0, 2], [0.1, 0.3], 10),
            ([0, 1], [0.1, 1.1], 10),
            ([0, 1], [0.1, float("nan")], 10),
            ([0, 1], [0.1, 0.2], 1),
        ]
        for labels, scores, bins in cases:
            with self.subTest(labels=labels, scores=scores, bins=bins):
                with self.assertRaises(ValueError):
                    quantile_calibration(labels, scores, n_bins=bins)


if __name__ == "__main__":
    unittest.main()
