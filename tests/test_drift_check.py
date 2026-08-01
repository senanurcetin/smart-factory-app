"""Tests for the PSI-based drift check (analysis/check_drift.py)."""

import unittest

import numpy as np
import pandas as pd

from analysis.check_drift import (
    build_drift_report,
    build_synthetic_live_batch,
    compute_categorical_psi,
    compute_numeric_psi,
    verdict,
)
from analysis.run_ai4i_case_study import add_derived_features, load_dataset


class VerdictThresholdTests(unittest.TestCase):
    def test_below_moderate_threshold(self):
        self.assertEqual(verdict(0.05), "no_significant_drift")

    def test_moderate_band(self):
        self.assertEqual(verdict(0.15), "moderate_drift")

    def test_significant_band(self):
        self.assertEqual(verdict(0.25), "significant_drift")


class NumericPsiTests(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(42)
        reference = pd.Series(rng.normal(loc=50, scale=5, size=2000))
        current = pd.Series(rng.normal(loc=50, scale=5, size=2000))
        psi = compute_numeric_psi(reference, current)
        self.assertLess(psi, 0.05)

    def test_shifted_distribution_has_high_psi(self):
        rng = np.random.default_rng(42)
        reference = pd.Series(rng.normal(loc=50, scale=5, size=2000))
        current = pd.Series(rng.normal(loc=90, scale=5, size=2000))
        psi = compute_numeric_psi(reference, current)
        self.assertGreater(psi, 0.2)


class CategoricalPsiTests(unittest.TestCase):
    def test_identical_category_proportions_have_near_zero_psi(self):
        reference = pd.Series(["L"] * 600 + ["M"] * 300 + ["H"] * 100)
        current = pd.Series(["L"] * 600 + ["M"] * 300 + ["H"] * 100)
        psi = compute_categorical_psi(reference, current)
        self.assertLess(psi, 0.01)

    def test_shifted_category_proportions_have_high_psi(self):
        reference = pd.Series(["L"] * 600 + ["M"] * 300 + ["H"] * 100)
        current = pd.Series(["L"] * 100 + ["M"] * 300 + ["H"] * 600)
        psi = compute_categorical_psi(reference, current)
        self.assertGreater(psi, 0.2)


class DriftReportIntegrationTests(unittest.TestCase):
    """End-to-end: the real AI4I data + the real injected-shift scenario."""

    @classmethod
    def setUpClass(cls):
        df = load_dataset()
        cls.reference_df = add_derived_features(df)
        cls.live_batch = build_synthetic_live_batch(df)
        cls.report = build_drift_report(cls.reference_df, cls.live_batch)

    def test_shifted_feature_is_flagged(self):
        self.assertIn("Tool wear [min]", self.report["drifted_features"])

    def test_unrelated_feature_is_not_flagged(self):
        self.assertNotIn("Air temperature [K]", self.report["drifted_features"])
        self.assertNotIn("Rotational speed [rpm]", self.report["drifted_features"])

    def test_overall_drift_detected_flag(self):
        self.assertTrue(self.report["drift_detected"])

    def test_report_has_required_fields(self):
        for key in ("method", "thresholds", "scenario", "features", "drift_detected"):
            with self.subTest(key=key):
                self.assertIn(key, self.report)


if __name__ == "__main__":
    unittest.main()
