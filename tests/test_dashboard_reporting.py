"""Verify exported BI data originates from the application's scored session.

A simulated telemetry series is an honest *working* dashboard, but not
evidence of plant production or calibrated failure probability.
"""

import csv
import io
import unittest
from datetime import datetime, timedelta, timezone

from main import app, historical_data


class SessionReportTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_csv_export_contains_model_scored_api_reading(self):
        response = self.client.get("/api/data")
        self.assertEqual(response.status_code, 200)
        scored = response.get_json()["current_reading"]

        export = self.client.get("/api/report.csv")
        self.assertEqual(export.status_code, 200)
        self.assertEqual(export.mimetype, "text/csv")
        self.assertEqual(
            export.headers["X-Data-Provenance"],
            "simulated-inputs;trained-model-scores",
        )
        self.assertEqual(
            export.headers["X-Data-Scope"],
            "shared-worker-buffer;last-240-readings",
        )
        self.assertIn("attachment;", export.headers["Content-Disposition"])
        rows = list(csv.DictReader(io.StringIO(export.get_data(as_text=True))))
        self.assertTrue(rows)
        self.assertEqual(float(rows[-1]["Model risk score (0-1)"]), scored["RiskScore"])
        self.assertEqual(
            rows[-1]["Timestamp (server local time of day)"],
            scored["Timestamp"],
        )
        self.assertEqual(
            rows[-1]["Recorded at (UTC, ISO 8601)"],
            scored["RecordedAtUTC"],
        )
        sample_time = datetime.fromisoformat(scored["RecordedAtUTC"])
        self.assertEqual(sample_time.utcoffset(), timedelta(0))
        self.assertLessEqual(abs(datetime.now(timezone.utc) - sample_time), timedelta(minutes=2))
        self.assertEqual(
            float(rows[-1]["Process temperature (C)"]),
            scored["Temperature"],
        )
        self.assertEqual(int(rows[-1]["Heuristic RUL proxy (hours)"]), scored["RUL"])
        self.assertGreaterEqual(float(rows[-1]["Model risk score (0-1)"]), 0)
        self.assertLessEqual(float(rows[-1]["Model risk score (0-1)"]), 1)

    def test_export_does_not_advance_or_rescore_simulation(self):
        self.client.get("/api/data")
        before = list(historical_data)
        self.client.get("/api/report.csv")
        self.assertEqual(list(historical_data), before)

    def test_dashboard_charts_fit_flex_viewport_and_kpis_start_unavailable(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("flex: 1 1 auto;", html)
        self.assertIn("min-height: 0;", html)
        self.assertIn(".chart-card { height:360px; }", html)
        for element in ("kpi-risk", "kpi-rul", "kpi-oee", "kpi-prod"):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"', html)
                self.assertNotIn(f'id="{element}">0', html)

    def test_telemetry_polling_retries_and_rejects_incomplete_payloads(self):
        html = self.client.get("/").get_data(as_text=True)
        for contract in (
            "if (!response.ok)",
            "Incomplete telemetry payload",
            "if (dashboardRefreshing) return;",
            "setInterval(refreshDashboard, refreshMs);",
            "Telemetry temporarily unavailable",
            'role="status" aria-live="polite"',
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, html)

    def test_dashboard_renders_live_chart_and_derived_oee(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for evidence in (
            'id="riskChart"',
            "riskChart.data.datasets[0].data.push(current.RiskScore * 100)",
            'id="oee-availability"',
            'id="oee-performance"',
            'id="oee-quality"',
            'href="/api/report.csv"',
            "SIMULATED PLANT DATA + TRAINED ML MODEL",
            "SHARED DEMO STREAM",
            'id="telemetry-clock"',
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, html)


if __name__ == "__main__":
    unittest.main()
