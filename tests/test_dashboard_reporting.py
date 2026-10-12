"""Verify exported BI data originates from the application's scored session.

A simulated telemetry series is an honest *working* dashboard, but not
evidence of plant production or calibrated failure probability.
"""

import csv
import io
import unittest

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
        self.assertIn("attachment;", export.headers["Content-Disposition"])
        rows = list(csv.DictReader(io.StringIO(export.get_data(as_text=True))))
        self.assertTrue(rows)
        self.assertEqual(float(rows[-1]["Model risk score (0-1)"]), scored["RiskScore"])
        self.assertEqual(rows[-1]["Timestamp"], scored["Timestamp"])
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
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, html)


if __name__ == "__main__":
    unittest.main()
