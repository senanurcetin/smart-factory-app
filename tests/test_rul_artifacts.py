"""Data quality and metrics contract tests for the C-MAPSS RUL case-study artifacts."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data" / "cmapss-rul-case-study"


def _load(name):
    return json.loads((DATA_DIR / name).read_text())


class ArtifactPresenceTests(unittest.TestCase):
    """All expected RUL case-study artifact files must exist."""

    REQUIRED = [
        "summary.json",
        "model-selection.json",
        "feature-importance.json",
        "sample-trajectories.json",
    ]

    def test_all_artifacts_exist(self):
        for fname in self.REQUIRED:
            with self.subTest(file=fname):
                self.assertTrue((DATA_DIR / fname).exists(), f"Missing artifact: {fname}")


class MetricsRangeTests(unittest.TestCase):
    """Final model metrics must beat the naive baseline by a wide, credible margin."""

    @classmethod
    def setUpClass(cls):
        cls.summary = _load("summary.json")

    def test_rmse_below_threshold(self):
        rmse = self.summary["final_model"]["rmse"]
        self.assertLess(rmse, 30.0, f"RMSE {rmse} above the expected credible range")

    def test_rmse_beats_baseline(self):
        final_rmse = self.summary["final_model"]["rmse"]
        baseline_rmse = self.summary["baseline"]["rmse"]
        self.assertLess(final_rmse, baseline_rmse)

    def test_phm08_score_beats_baseline(self):
        final_score = self.summary["final_model"]["phm08_score"]
        baseline_score = self.summary["baseline"]["phm08_score"]
        self.assertLess(final_score, baseline_score)

    def test_lift_vs_baseline_recorded(self):
        lift = self.summary["lift_vs_baseline"]
        self.assertGreater(lift["rmse_reduction_pct"], 20.0)
        self.assertGreater(lift["phm08_score_reduction_pct"], 20.0)

    def test_dataset_units_recorded(self):
        dataset = self.summary["dataset"]
        self.assertEqual(dataset["train_units"], 100)
        self.assertEqual(dataset["test_units"], 100)


class HyperparameterTuningTests(unittest.TestCase):
    """The RUL model's hyperparameters must be searched with a grouped CV strategy."""

    @classmethod
    def setUpClass(cls):
        cls.model_selection = _load("model-selection.json")

    def test_tuning_summary_present(self):
        self.assertIn("hyperparameter_tuning", self.model_selection)

    def test_tuning_uses_group_kfold(self):
        tuning = self.model_selection["hyperparameter_tuning"]
        self.assertIn("GroupKFold", tuning["cv_strategy"])

    def test_tuning_summary_has_best_params(self):
        tuning = self.model_selection["hyperparameter_tuning"]
        self.assertIn("best_params", tuning)
        self.assertGreater(len(tuning["best_params"]), 0)


class FeatureEngineeringImpactTests(unittest.TestCase):
    """Rolling-window feature impact must be isolated from hyperparameter tuning."""

    @classmethod
    def setUpClass(cls):
        cls.impact = _load("model-selection.json")["feature_engineering_impact"]

    def test_raw_and_enhanced_present(self):
        self.assertIn("raw_features", self.impact)
        self.assertIn("enhanced_features", self.impact)

    def test_delta_recorded(self):
        self.assertIn("delta", self.impact)
        self.assertIn("rmse", self.impact["delta"])


class ModelCardTests(unittest.TestCase):
    """Minimal model-card metadata must be recorded for the RUL model too."""

    @classmethod
    def setUpClass(cls):
        cls.model_card = _load("model-selection.json")["model_card"]

    def test_required_fields_present(self):
        required = {
            "trained_at_utc",
            "python_version",
            "scikit_learn_version",
            "shap_version",
            "dataset",
            "random_seed",
        }
        missing = required - set(self.model_card.keys())
        self.assertEqual(missing, set(), f"Missing fields: {missing}")

    def test_dataset_hash_recorded(self):
        self.assertIn("sha256", self.model_card["dataset"])
        self.assertEqual(len(self.model_card["dataset"]["sha256"]), 64)


class FeatureImportanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feature_importance = _load("feature-importance.json")

    def test_features_list_non_empty(self):
        self.assertGreater(len(self.feature_importance["features"]), 0)

    def test_all_features_have_importance_score(self):
        for row in self.feature_importance["features"]:
            with self.subTest(feature=row.get("feature")):
                self.assertIn("mean_abs_shap", row)


class SampleTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectories = _load("sample-trajectories.json")["sample_units"]

    def test_at_least_one_trajectory(self):
        self.assertGreater(len(self.trajectories), 0)

    def test_trajectory_arrays_aligned(self):
        for unit in self.trajectories:
            with self.subTest(unit=unit["unit_number"]):
                n_cycles = len(unit["cycles"])
                self.assertEqual(len(unit["predicted_rul"]), n_cycles)
                self.assertEqual(len(unit["true_rul_extrapolated"]), n_cycles)


if __name__ == "__main__":
    unittest.main()
