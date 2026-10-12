"""Tests for the PSI-based drift check (analysis/check_drift.py)."""

import os
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
    """Offline end-to-end check with a seeded AI4I-schema fixture."""

    @classmethod
    def setUpClass(cls):
        # PR checks must never download a dataset. Cover the full PSI/report
        # integration deterministically; real-dataset coverage is opt-in below.
        rng = np.random.default_rng(42)
        rows = 10_000
        df = pd.DataFrame({
            "Type": rng.choice(["L", "M", "H"], size=rows, p=[0.3, 0.5, 0.2]),
            "Air temperature [K]": rng.normal(300, 2, rows),
            "Process temperature [K]": rng.normal(310, 2, rows),
            "Rotational speed [rpm]": rng.normal(1500, 230, rows),
            "Torque [Nm]": rng.uniform(5, 75, rows),
            "Tool wear [min]": rng.uniform(0, 250, rows),
        })
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


@unittest.skipUnless(
    os.environ.get("RUN_REAL_DATA_TESTS") == "1",
    "real AI4I data is only downloaded during manual benchmark validation",
)
class RealDatasetDriftIntegrationTests(DriftReportIntegrationTests):
    """Re-run the same assertions on UCI data during opt-in retraining."""

    @classmethod
    def setUpClass(cls):
        df = load_dataset()
        cls.reference_df = add_derived_features(df)
        cls.live_batch = build_synthetic_live_batch(df)
        cls.report = build_drift_report(cls.reference_df, cls.live_batch)


class PsiBoundaryRegressionTests(unittest.TestCase):
    """Pure statistical checks that do not depend on remote dataset downloads."""

    def test_verdict_thresholds_are_inclusive(self):
        self.assertEqual(verdict(0.1), "moderate_drift")
        self.assertEqual(verdict(0.2), "significant_drift")
        self.assertEqual(verdict(0.0), "no_significant_drift")

    def test_identical_numeric_samples_have_exactly_zero_psi(self):
        values = pd.Series(np.linspace(1.0, 100.0, 1000))
        self.assertAlmostEqual(compute_numeric_psi(values, values.copy()), 0.0)

    def test_numeric_psi_drops_missing_values(self):
        reference = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
        current = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(compute_numeric_psi(reference, current, buckets=4), 0.0)

    def test_empty_numeric_stream_does_not_raise_false_alert(self):
        samples = pd.Series([], dtype=float)
        self.assertEqual(compute_numeric_psi(samples, pd.Series([1.0, 2.0])), 0.0)
        self.assertEqual(compute_numeric_psi(pd.Series([1.0, 2.0]), samples), 0.0)

    def test_constant_numeric_sensor_does_not_crash_quantile_binning(self):
        constant = pd.Series([25.0] * 100)
        self.assertEqual(compute_numeric_psi(constant, constant.copy()), 0.0)

    def test_strong_numeric_shift_triggers_significant_drift(self):
        baseline = pd.Series(np.linspace(1, 100, 1000))
        shifted = baseline + 1000
        self.assertEqual(verdict(compute_numeric_psi(baseline, shifted)), "significant_drift")

    def test_same_categorical_distribution_in_different_order_has_no_drift(self):
        baseline = pd.Series(["A"] * 80 + ["B"] * 20)
        reordered = pd.Series(["B"] * 20 + ["A"] * 80)
        self.assertAlmostEqual(compute_categorical_psi(baseline, reordered), 0.0)

    def test_emerging_sensor_state_is_detected(self):
        baseline = pd.Series(["healthy"] * 95 + ["warning"] * 5)
        current = pd.Series(["healthy"] * 10 + ["warning"] * 5 + ["critical"] * 85)
        self.assertEqual(
            verdict(compute_categorical_psi(baseline, current)),
            "significant_drift",
        )

    def test_empty_categorical_inputs_are_safe(self):
        missing = pd.Series([], dtype=object)
        self.assertEqual(compute_categorical_psi(missing, missing), 0.0)

    def test_identical_numeric_sensor_ignores_missing_in_both_batches(self):
        baseline = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
        current = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
        self.assertAlmostEqual(compute_numeric_psi(baseline, current, buckets=4), 0.0)


if __name__ == "__main__":
    unittest.main()
