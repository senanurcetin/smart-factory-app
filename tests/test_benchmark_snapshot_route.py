"""Flask evidence endpoint contract; never invokes BigQuery or Kaggle."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from analysis.benchmark_snapshot import SnapshotError
from main import app


class BenchmarkSnapshotApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_snapshot_http_200_and_committed_benchmark_metrics(self):
        response = self.client.get("/api/benchmark-snapshot")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        payload = response.get_json()
        self.assertEqual(payload["data_origin"], "committed_offline_case_studies")
        self.assertAlmostEqual(payload["benchmarks"]["ai4i_failure"]["pr_auc"], 0.9019)
        self.assertAlmostEqual(
            payload["benchmarks"]["cmapss_fd001_rul"]["rmse_cycles"], 18.0535
        )

    def test_never_claims_live_bigquery_or_factory_telemetry(self):
        payload = self.client.get("/api/benchmark-snapshot").get_json()
        self.assertFalse(payload["bigquery_dataset_connected"])
        self.assertFalse(payload["kaggle_source_version_verified"])
        self.assertFalse(payload["model_training_performed"])
        self.assertFalse(payload["real_plant_telemetry"])
        self.assertIn("not live factory readings", payload["display_disclosure"])

    def test_invalid_committed_evidence_returns_http_503_without_leaking_paths(self):
        with patch(
            "main.build_benchmark_snapshot", side_effect=SnapshotError("secret local path")
        ):
            response = self.client.get("/api/benchmark-snapshot")
        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["reason"], "committed_benchmark_evidence_invalid")
        self.assertFalse(payload["bigquery_dataset_connected"])
        self.assertNotIn("secret local path", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
