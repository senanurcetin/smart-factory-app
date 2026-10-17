"""Offline regression tests for versioned model candidates; no real pickle loads."""

import json
import tempfile
import unittest
from pathlib import Path

from analysis.model_registry import (
    RegistryError,
    assess_candidate,
    stage_candidate,
    verify_candidate,
)

EVALUATED_AT = "2026-10-17T18:00:00+03:00"
DATASET_SHA = "a" * 64


class OfflineModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifacts = self.root / "analysis" / "artifacts"
        self.artifacts.mkdir(parents=True)
        self.active = self.artifacts / "model.pkl"
        self.active.write_bytes(b"offline-test-not-a-real-model-v1")

    def stage(self, pr_auc=0.90):
        return stage_candidate(
            self.root,
            "ai4i_failure",
            dataset_sha256=DATASET_SHA,
            metrics={"pr_auc": pr_auc, "roc_auc": 0.98},
            evaluated_at=EVALUATED_AT,
        )

    def test_staging_is_content_addressed_and_keeps_active_model_unchanged(self):
        before = self.active.read_bytes()
        manifest = self.stage()
        self.assertEqual(manifest["model_id"], "ai4i_failure")
        self.assertEqual(manifest["status"], "candidate")
        self.assertEqual(manifest["evaluated_at"], "2026-10-17T15:00:00+00:00")
        self.assertEqual(len(manifest["model_sha256"]), 64)
        self.assertEqual(self.active.read_bytes(), before)
        self.assertEqual(verify_candidate(self.root, manifest), manifest)

    def test_restage_identical_model_keeps_same_candidate_location(self):
        first = self.stage()
        second = self.stage()
        self.assertEqual(first, second)

    def test_two_versions_keep_independent_copies_and_require_manual_review(self):
        baseline = self.stage(0.90)
        self.active.write_bytes(b"offline-test-not-a-real-model-v2")
        candidate = self.stage(0.93)
        self.assertNotEqual(baseline["relative_path"], candidate["relative_path"])
        self.assertTrue((self.root / baseline["relative_path"]).is_file())
        review = assess_candidate(self.root, baseline, candidate)
        self.assertEqual(review["status"], "manual_review_required")
        self.assertFalse(review["automatic_promotion"])

    def test_worse_candidate_is_rejected(self):
        baseline = self.stage(0.90)
        self.active.write_bytes(b"offline-test-not-a-real-model-v2")
        candidate = self.stage(0.80)
        self.assertEqual(
            assess_candidate(self.root, baseline, candidate)["status"],
            "rejected_no_improvement",
        )

    def test_rul_comparison_minimizes_rmse(self):
        (self.artifacts / "rul_model.pkl").write_bytes(b"rul-v1")
        old = stage_candidate(
            self.root,
            "cmapss_fd001_rul",
            dataset_sha256=DATASET_SHA,
            metrics={"rmse": 18.05},
            evaluated_at=EVALUATED_AT,
        )
        (self.artifacts / "rul_model.pkl").write_bytes(b"rul-v2")
        new = stage_candidate(
            self.root,
            "cmapss_fd001_rul",
            dataset_sha256=DATASET_SHA,
            metrics={"rmse": 17.5},
            evaluated_at=EVALUATED_AT,
        )
        self.assertEqual(assess_candidate(self.root, old, new)["status"], "manual_review_required")

    def test_refuses_corrupted_candidate_bytes(self):
        manifest = self.stage()
        (self.root / manifest["relative_path"]).write_bytes(b"tampered")
        with self.assertRaises(RegistryError):
            verify_candidate(self.root, manifest)

    def test_refuses_manifest_path_traversal(self):
        manifest = self.stage()
        manifest["relative_path"] = "../../unsafe.pkl"
        with self.assertRaises(RegistryError):
            verify_candidate(self.root, manifest)

    def test_refuses_forged_approval_flag_and_schema_version(self):
        manifest = self.stage()
        forged = dict(manifest, status="approved")
        with self.assertRaises(RegistryError):
            verify_candidate(self.root, forged)
        forged = dict(manifest, schema_version=2)
        with self.assertRaises(RegistryError):
            verify_candidate(self.root, forged)
        forged = dict(manifest, automatic_promotion=True)
        with self.assertRaises(RegistryError):
            verify_candidate(self.root, forged)

    def test_candidate_manifest_is_persisted_and_matches(self):
        manifest = self.stage()
        manifest_path = self.root / (
            "analysis/artifacts/registry/ai4i_failure/"
            + manifest["model_sha256"]
            + ".json"
        )
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)

    def test_forged_benchmark_metrics_cannot_be_verified(self):
        manifest = self.stage()
        forged = dict(manifest, metrics={"pr_auc": 0.999, "roc_auc": 0.999})
        with self.assertRaisesRegex(RegistryError, "does not match stored metadata"):
            verify_candidate(self.root, forged)

    def test_restage_same_model_with_changed_metadata_fails(self):
        self.stage(0.90)
        with self.assertRaisesRegex(RegistryError, "metadata differs"):
            self.stage(0.93)

    def test_corrupted_manifest_fails_closed(self):
        manifest = self.stage()
        path = self.root / (
            "analysis/artifacts/registry/ai4i_failure/"
            + manifest["model_sha256"]
            + ".json"
        )
        path.write_text("{corrupt json", encoding="utf-8")
        with self.assertRaises(RegistryError):
            verify_candidate(self.root, manifest)

    def test_refuses_nonnumeric_nan_or_missing_primary_metric(self):
        for metrics in ({"pr_auc": float("nan")}, {"pr_auc": True}, {"roc_auc": 0.9}):
            with self.subTest(metrics=repr(metrics)):
                with self.assertRaises(RegistryError):
                    stage_candidate(
                        self.root,
                        "ai4i_failure",
                        dataset_sha256=DATASET_SHA,
                        metrics=metrics,
                        evaluated_at=EVALUATED_AT,
                    )

    def test_rejects_impossible_classification_metrics(self):
        for metric_values in (
            {"pr_auc": 1.1},
            {"pr_auc": -0.01},
            {"pr_auc": 0.9, "roc_auc": 1.2},
            {"pr_auc": 0.9, "f1": -0.4},
        ):
            with self.subTest(metrics=metric_values):
                with self.assertRaises(RegistryError):
                    stage_candidate(
                        self.root,
                        "ai4i_failure",
                        dataset_sha256=DATASET_SHA,
                        metrics=metric_values,
                        evaluated_at=EVALUATED_AT,
                    )

    def test_rejects_negative_rul_error(self):
        (self.artifacts / "rul_model.pkl").write_bytes(b"fake-regression-model")
        with self.assertRaises(RegistryError):
            stage_candidate(
                self.root,
                "cmapss_fd001_rul",
                dataset_sha256=DATASET_SHA,
                metrics={"rmse": -1},
                evaluated_at=EVALUATED_AT,
            )

    def test_refuses_missing_or_mismatched_dataset_digest(self):
        for invalid in ("broken", "0" * 63, "X" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RegistryError):
                    stage_candidate(
                        self.root,
                        "ai4i_failure",
                        dataset_sha256=invalid,
                        metrics={"pr_auc": 0.9},
                        evaluated_at=EVALUATED_AT,
                    )

    def test_refuses_model_symlink(self):
        target = self.artifacts / "hidden.pkl"
        target.write_bytes(b"local-test")
        self.active.unlink()
        try:
            self.active.symlink_to(target)
        except OSError:
            self.skipTest("Host cannot create symlinks")
        with self.assertRaises(RegistryError):
            self.stage()

    def test_requires_timezone_in_manifest(self):
        with self.assertRaises(RegistryError):
            stage_candidate(
                self.root,
                "ai4i_failure",
                dataset_sha256=DATASET_SHA,
                metrics={"pr_auc": 0.9},
                evaluated_at="2026-10-17T12:00:00",
            )

    def test_manifest_can_be_serialized_without_unpickling(self):
        manifest = self.stage()
        restored = json.loads(json.dumps(manifest))
        self.assertEqual(verify_candidate(self.root, restored), manifest)


if __name__ == "__main__":
    unittest.main()
