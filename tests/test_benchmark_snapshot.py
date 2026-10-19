"""No-network NEXUS historical benchmark snapshot QA."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from analysis.benchmark_snapshot import SnapshotError, build_benchmark_snapshot

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for stem in ("ai4i-case-study", "cmapss-rul-case-study"):
            dest = self.root / "docs/data" / stem
            dest.mkdir(parents=True)
            for name in ("summary.json", "model-selection.json"):
                shutil.copyfile(ROOT / "docs/data" / stem / name, dest / name)
        source = ROOT / "docs/data/smart-factory-source-manifest.json"
        shutil.copyfile(source, self.root / "docs/data/smart-factory-source-manifest.json")

    def snapshot(self):
        return build_benchmark_snapshot(self.root)

    def test_snapshot_stays_historical_and_no_live_claim(self):
        result = self.snapshot()
        self.assertEqual(result["data_origin"], "committed_offline_case_studies")
        self.assertFalse(result["bigquery_dataset_connected"])
        self.assertFalse(result["real_plant_telemetry"])
        self.assertFalse(result["model_training_performed"])
        self.assertIn("not live factory readings", result["display_disclosure"])

    def test_existing_ai4i_and_fd001_evidence_has_distinct_tasks(self):
        result = self.snapshot()["benchmarks"]
        self.assertEqual(result["ai4i_failure"]["holdout_rows"], 2000)
        self.assertAlmostEqual(result["ai4i_failure"]["pr_auc"], 0.9019)
        self.assertEqual(result["cmapss_fd001_rul"]["heldout_test_engines"], 100)
        self.assertAlmostEqual(result["cmapss_fd001_rul"]["rmse_cycles"], 18.0535)
        self.assertFalse(result["cmapss_fd001_rul"]["estimated_live_plant_rul"])

    def test_source_archive_checks_are_explicitly_pending(self):
        data = self.snapshot()
        self.assertEqual(len(data["sources"]), 2)
        for source in data["sources"]:
            self.assertFalse(source["source_archive_sha256_verified"])
            self.assertFalse(source["kaggle_mirror_version_pinned"])

    def test_forged_ai4i_evaluation_data_rejected(self):
        file = self.root / "docs/data/ai4i-case-study/summary.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["final_model"]["pr_auc"] = 1.2
        file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotError, "Out-of-range"):
            self.snapshot()

    def test_forged_negative_rul_rmse_rejected(self):
        file = self.root / "docs/data/cmapss-rul-case-study/summary.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["final_model"]["rmse"] = -20
        file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotError, "Out-of-range"):
            self.snapshot()

    def test_dataset_fingerprint_forgery_rejected(self):
        file = self.root / "docs/data/ai4i-case-study/model-selection.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["model_card"]["dataset"]["sha256"] = "f" * 64
        file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotError, "lineage mismatch"):
            self.snapshot()

    def test_missing_evidence_rejected(self):
        (self.root / "docs/data/cmapss-rul-case-study/summary.json").unlink()
        with self.assertRaisesRegex(SnapshotError, "inaccessible"):
            self.snapshot()

    def test_wrong_holdout_dimension_rejected(self):
        file = self.root / "docs/data/ai4i-case-study/summary.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        payload["dataset"]["evaluation_split"]["holdout_rows"] = -1
        file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotError, "dimensions"):
            self.snapshot()


if __name__ == "__main__":
    unittest.main()
