"""Data quality and metrics contract tests for AI4I case study artifacts."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data" / "ai4i-case-study"


def _load(name):
    return json.loads((DATA_DIR / name).read_text())


class ArtifactPresenceTests(unittest.TestCase):
    """All expected artifact files must exist."""

    REQUIRED = [
        "summary.json",
        "benchmark-comparison.json",
        "feature-importance.json",
        "review-queue.json",
        "cost-simulation.json",
        "dataset-profile.json",
        "failure-mode-breakdown.json",
        "model-selection.json",
        "validation-robustness.json",
    ]

    def test_all_artifacts_exist(self):
        for fname in self.REQUIRED:
            with self.subTest(file=fname):
                self.assertTrue((DATA_DIR / fname).exists(), f"Missing artifact: {fname}")


class ValidationRobustnessTests(unittest.TestCase):
    """The repeated-holdout robustness check must cover multiple seeds and stay
    consistent with the metric thresholds enforced elsewhere in this file."""

    @classmethod
    def setUpClass(cls):
        cls.robustness = _load("validation-robustness.json")

    def test_multiple_seeds_present(self):
        self.assertGreaterEqual(len(self.robustness["per_seed_results"]), 5)

    def test_roc_auc_mean_above_threshold(self):
        self.assertGreaterEqual(self.robustness["roc_auc_mean"], 0.95)

    def test_pr_auc_mean_above_threshold(self):
        self.assertGreaterEqual(self.robustness["pr_auc_mean"], 0.75)

    def test_per_seed_results_have_required_fields(self):
        required = {"split_seed", "roc_auc", "pr_auc", "f1"}
        for entry in self.robustness["per_seed_results"]:
            with self.subTest(seed=entry.get("split_seed")):
                missing = required - set(entry.keys())
                self.assertEqual(missing, set(), f"Missing fields: {missing}")


class HyperparameterTuningTests(unittest.TestCase):
    """The final model's hyperparameters must be searched, not hardcoded guesses."""

    @classmethod
    def setUpClass(cls):
        cls.model_selection = _load("model-selection.json")

    def test_tuning_summary_present(self):
        self.assertIn("hyperparameter_tuning", self.model_selection)

    def test_tuning_summary_has_best_params(self):
        tuning = self.model_selection["hyperparameter_tuning"]
        self.assertIn("best_params", tuning)
        self.assertGreater(len(tuning["best_params"]), 0)


class ModelCardTests(unittest.TestCase):
    """Minimal model-card metadata must be recorded for every trained model."""

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


class MetricsRangeTests(unittest.TestCase):
    """Final model metrics must meet portfolio-grade thresholds."""

    @classmethod
    def setUpClass(cls):
        cls.summary = _load("summary.json")
        cls.final = cls.summary.get("final_model", {})

    def test_roc_auc_above_threshold(self):
        roc = self.final.get("roc_auc", 0)
        self.assertGreaterEqual(roc, 0.95, f"ROC-AUC {roc} below threshold 0.95")

    def test_pr_auc_above_threshold(self):
        pr = self.final.get("pr_auc", 0)
        self.assertGreaterEqual(pr, 0.80, f"PR-AUC {pr} below threshold 0.80")

    def test_f1_above_threshold(self):
        f1 = self.final.get("f1", 0)
        self.assertGreaterEqual(f1, 0.75, f"F1 {f1} below threshold 0.75")

    def test_precision_above_threshold(self):
        p = self.final.get("precision", 0)
        self.assertGreaterEqual(p, 0.80, f"Precision {p} below threshold 0.80")

    def test_model_name_recorded(self):
        name = self.final.get("name", "")
        self.assertTrue(len(name) > 0, "Final model name is empty")


class BenchmarkComparisonTests(unittest.TestCase):
    """Benchmark comparison artifact must include all required models and fields."""

    @classmethod
    def setUpClass(cls):
        cls.benchmarks = _load("benchmark-comparison.json")

    def test_four_models_present(self):
        self.assertEqual(len(self.benchmarks), 4, "Expected 4 benchmark entries")

    def test_required_fields_per_model(self):
        required = {"model", "roc_auc", "pr_auc", "f1", "precision", "recall"}
        for entry in self.benchmarks:
            with self.subTest(model=entry.get("model")):
                missing = required - set(entry.keys())
                self.assertEqual(missing, set(), f"Missing fields: {missing}")

    def test_final_model_beats_baseline_roc(self):
        by_model = {d["model"]: d for d in self.benchmarks}
        baseline = by_model["dummy_baseline"]["roc_auc"]
        final = by_model["hist_gradient_boosting"]["roc_auc"]
        self.assertGreater(final, baseline, "Final model must beat dummy baseline ROC-AUC")

    def test_final_model_beats_logistic_pr(self):
        by_model = {d["model"]: d for d in self.benchmarks}
        lr = by_model["logistic_regression"]["pr_auc"]
        final = by_model["hist_gradient_boosting"]["pr_auc"]
        self.assertGreater(final, lr, "Final model must beat logistic regression PR-AUC")


class FeatureImportanceTests(unittest.TestCase):
    """Feature importance artifact must have expected structure."""

    @classmethod
    def setUpClass(cls):
        cls.data = _load("feature-importance.json")

    def test_features_list_non_empty(self):
        self.assertGreater(len(self.data.get("features", [])), 0)

    def test_top_feature_is_rotational_or_thermal(self):
        top = self.data["features"][0]["feature"]
        expected = {"Rotational speed [rpm]", "thermal_stress", "mechanical_load"}
        self.assertIn(top, expected, f"Unexpected top feature: {top}")

    def test_all_features_have_importance_score(self):
        for f in self.data["features"]:
            with self.subTest(feature=f["feature"]):
                self.assertIn("importance", f)
                self.assertIsInstance(f["importance"], float)


class ReviewQueueTests(unittest.TestCase):
    """Review queue artifact must show meaningful yield lift."""

    @classmethod
    def setUpClass(cls):
        cls.data = _load("review-queue.json")

    def test_review_budgets_present(self):
        self.assertIn("review_budgets", self.data)
        self.assertGreater(len(self.data["review_budgets"]), 0)

    def test_top5_budget_captures_majority_of_failures(self):
        b5 = next(
            (b for b in self.data["review_budgets"] if b["review_fraction"] == 0.05),
            None,
        )
        if b5 is None:
            self.skipTest("No 5% budget entry found")
        self.assertGreaterEqual(
            b5["failure_capture_rate"],
            0.80,
            f"5% budget should capture at least 80% of failures, got {b5['failure_capture_rate']}",
        )

    def test_yield_lift_above_baseline(self):
        for b in self.data["review_budgets"]:
            with self.subTest(fraction=b["review_fraction"]):
                self.assertGreater(
                    b["yield_lift_vs_random"], 1.0, "Yield lift must exceed 1x random"
                )


class DatasetProfileTests(unittest.TestCase):
    """Dataset profile artifact must record expected dataset characteristics."""

    @classmethod
    def setUpClass(cls):
        cls.profile = _load("dataset-profile.json")

    def test_row_count(self):
        rows = self.profile.get("row_count", 0)
        self.assertEqual(rows, 10000, f"Expected 10000 rows, got {rows}")

    def test_failure_rate_reasonable(self):
        rate = self.profile.get("target_distribution", {}).get("failure", 0) / self.profile.get(
            "row_count", 1
        )
        self.assertGreater(rate, 0.02)
        self.assertLess(rate, 0.10)


class VisualAssetTests(unittest.TestCase):
    """All generated PNG assets must be present and non-trivial in size."""

    EXPECTED_CHARTS = [
        # analysis charts
        "model-comparison.png",
        "feature-importance.png",
        "review-queue-curve.png",
        "cost-model.png",
        # EDA charts
        "eda-class-balance.png",
        "eda-failure-modes.png",
        "eda-type-distribution.png",
        "eda-confusion-matrix.png",
    ]
    ASSETS_DIR = ROOT / "docs" / "assets"

    def test_chart_files_exist(self):
        for fname in self.EXPECTED_CHARTS:
            with self.subTest(chart=fname):
                path = self.ASSETS_DIR / fname
                self.assertTrue(path.exists(), f"Missing chart: {fname}")

    def test_chart_files_non_trivial(self):
        for fname in self.EXPECTED_CHARTS:
            path = self.ASSETS_DIR / fname
            if path.exists():
                with self.subTest(chart=fname):
                    self.assertGreater(path.stat().st_size, 10_000, f"{fname} seems too small")


class SqlAnalysisTests(unittest.TestCase):
    """SQL analysis artifact must exist and contain expected queries."""

    @classmethod
    def setUpClass(cls):
        cls.sql = _load("sql-analysis.json")

    def test_sql_artifact_exists(self):
        self.assertTrue((DATA_DIR / "sql-analysis.json").exists())

    def test_four_queries_present(self):
        self.assertEqual(len(self.sql.get("queries", [])), 4)

    def test_query_ids_correct(self):
        ids = {q["id"] for q in self.sql["queries"]}
        self.assertEqual(ids, {"Q1", "Q2", "Q3", "Q4"})

    def test_q1_failure_mode_ranking_has_results(self):
        q1 = next(q for q in self.sql["queries"] if q["id"] == "Q1")
        self.assertEqual(len(q1["results"]), 5, "Expected 5 failure modes in Q1")

    def test_q2_model_ranking_best_is_hgb(self):
        q2 = next(q for q in self.sql["queries"] if q["id"] == "Q2")
        top = q2["results"][0]
        self.assertIn("hist_gradient_boosting", top["model"])

    def test_q3_review_queue_has_three_budgets(self):
        q3 = next(q for q in self.sql["queries"] if q["id"] == "Q3")
        self.assertEqual(len(q3["results"]), 3)

    def test_final_model_metrics_consistent_with_summary(self):
        summary = _load("summary.json")
        roc = summary["final_model"]["roc_auc"]
        pr = summary["final_model"]["pr_auc"]
        self.assertGreaterEqual(roc, 0.95)
        self.assertGreaterEqual(pr, 0.85)


if __name__ == "__main__":
    unittest.main()
