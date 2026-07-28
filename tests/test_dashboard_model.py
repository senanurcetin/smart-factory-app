"""Sanity checks that the live dashboard is wired to the real trained model."""

import unittest

import pandas as pd

from analysis.run_ai4i_case_study import MODEL_FEATURES_ENHANCED, add_derived_features
from main import app, risk_model


class DashboardModelTests(unittest.TestCase):
    def test_risk_model_is_the_persisted_pipeline(self):
        self.assertTrue(hasattr(risk_model, "predict_proba"))

    def test_risk_score_responds_to_stress_level(self):
        """A low-stress reading and a high-stress reading must not score the same."""
        low_stress = pd.DataFrame(
            [
                {
                    "Type": "L",
                    "Air temperature [K]": 298.0,
                    "Process temperature [K]": 308.5,
                    "Rotational speed [rpm]": 1500.0,
                    "Torque [Nm]": 40.0,
                    "Tool wear [min]": 10.0,
                }
            ]
        )
        high_stress = pd.DataFrame(
            [
                {
                    "Type": "L",
                    "Air temperature [K]": 298.0,
                    "Process temperature [K]": 308.5,
                    "Rotational speed [rpm]": 1500.0,
                    "Torque [Nm]": 75.0,
                    "Tool wear [min]": 240.0,
                }
            ]
        )
        low_risk = risk_model.predict_proba(
            add_derived_features(low_stress)[MODEL_FEATURES_ENHANCED]
        )[:, 1][0]
        high_risk = risk_model.predict_proba(
            add_derived_features(high_stress)[MODEL_FEATURES_ENHANCED]
        )[:, 1][0]
        self.assertGreater(
            high_risk,
            low_risk,
            "High torque/tool-wear reading should score a higher failure risk "
            "than a nominal reading.",
        )

    def test_api_data_risk_score_is_not_constant(self):
        """/api/data must not return a hardcoded or effectively-random-but-static score."""
        client = app.test_client()
        scores = []
        for _ in range(20):
            response = client.get("/api/data")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            reading = payload["current_reading"]
            self.assertIn("RiskScore", reading)
            self.assertGreaterEqual(reading["RiskScore"], 0.0)
            self.assertLessEqual(reading["RiskScore"], 1.0)
            scores.append(reading["RiskScore"])
        self.assertGreater(len(set(scores)), 1, "RiskScore should vary across ticks.")

    def test_api_data_rul_is_consistent_with_risk_score(self):
        """RUL should fall as the model's risk score rises (monotonic proxy)."""
        client = app.test_client()
        response = client.get("/api/data")
        payload = response.get_json()
        reading = payload["current_reading"]
        expected_rul = max(1, round(48 * (1 - reading["RiskScore"])))
        self.assertEqual(reading["RUL"], expected_rul)


if __name__ == "__main__":
    unittest.main()
