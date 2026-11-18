"""POST /api/rul/fd001-demo: real FD001 RUL model behind the input contract (issue #52)."""

import math
import unittest

import joblib
import pandas as pd

import main
import rul_case_study
from analysis.rul_input_contract import MODEL_COLUMNS, build_fd001_feature_row

URL = "/api/rul/fd001-demo"


def fd001_history(cycles=7, unit=3):
    rows = []
    for cycle in range(1, cycles + 1):
        row = {
            "unit_number": unit,
            "time_cycles": cycle,
            "setting_1": -0.0007,
            "setting_2": -0.0004,
            "setting_3": 100.0,
        }
        # Values near the FD001 operating point, drifting slowly with each cycle.
        base = {
            1: 518.67,
            2: 642.3,
            3: 1589.7,
            4: 1400.6,
            5: 14.62,
            6: 21.61,
            7: 554.36,
            8: 2388.06,
            9: 9046.19,
            10: 1.3,
            11: 47.47,
            12: 521.66,
            13: 2388.02,
            14: 8138.62,
            15: 8.42,
            16: 0.03,
            17: 392.0,
            18: 2388.0,
            19: 100.0,
            20: 39.06,
            21: 23.42,
        }
        row.update({f"sensor_{i}": base[i] + 0.01 * cycle * (i % 3) for i in range(1, 22)})
        rows.append(row)
    return rows


def ai4i_reading():
    return {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }


class Fd001DemoEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = main.app.test_client()

    def post(self, body):
        return self.client.post(URL, json=body)

    def test_valid_history_returns_the_model_estimate(self):
        history = fd001_history()
        response = self.post({"history": history})
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()

        pipeline = joblib.load(rul_case_study.RUL_MODEL_PATH)
        features = build_fd001_feature_row(history)["model_features"]
        expected = float(pipeline.predict(pd.DataFrame([features], columns=list(MODEL_COLUMNS)))[0])
        expected = round(min(max(expected, 0.0), 125.0), 1)

        self.assertEqual(body["predicted_rul_cycles"], expected)
        self.assertTrue(math.isfinite(body["predicted_rul_cycles"]))
        self.assertEqual((body["unit_number"], body["last_cycle"], body["window"]), (3, 7, 5))
        self.assertEqual(body["rul_cap_cycles"], 125)

    def test_response_is_labelled_as_benchmark_not_plant_data(self):
        body = self.post({"history": fd001_history()}).get_json()
        self.assertEqual(body["source"], "NASA C-MAPSS FD001 simulated benchmark")
        self.assertIs(body["is_live_plant_data"], False)
        self.assertIs(body["automated_maintenance_action"], False)
        self.assertIn("not plant telemetry", body["note"])

    def test_too_short_history_is_rejected(self):
        response = self.post({"history": fd001_history(cycles=4)})
        self.assertEqual(response.status_code, 422)
        self.assertIn("five contiguous cycles", response.get_json()["error"])

    def test_ai4i_shaped_input_is_rejected(self):
        for body in ({"history": [ai4i_reading()] * 6}, ai4i_reading(), [ai4i_reading()]):
            with self.subTest(body=type(body).__name__):
                response = self.post(body)
                self.assertIn(response.status_code, (400, 422))
                self.assertNotIn("predicted_rul_cycles", response.get_json())

    def test_wrong_schema_and_bad_cycles_are_rejected(self):
        missing = fd001_history()
        del missing[2]["sensor_7"]
        gap = fd001_history()
        gap[3]["time_cycles"] = 99
        for history in (missing, gap, "not a list", [1, 2, 3, 4, 5]):
            with self.subTest(history=str(history)[:40]):
                self.assertEqual(self.post({"history": history}).status_code, 422)

    def test_malformed_requests_are_rejected(self):
        self.assertEqual(
            self.client.post(URL, data="nope", content_type="text/plain").status_code, 400
        )
        self.assertEqual(self.post({"history": fd001_history(), "extra": 1}).status_code, 400)
        self.assertEqual(self.client.get(URL).status_code, 405)

    def test_oversized_history_is_rejected_before_scoring(self):
        response = self.post({"history": fd001_history(cycles=rul_case_study.MAX_DEMO_CYCLES + 1)})
        self.assertEqual(response.status_code, 422)

    def test_dashboard_rul_tile_stays_the_ai4i_heuristic(self):
        reading = self.client.get("/api/data").get_json()["current_reading"]
        self.assertIn("RUL", reading)
        self.assertLessEqual(reading["RUL"], main.MAX_RUL_HOURS)
        self.assertNotIn("predicted_rul_cycles", reading)


if __name__ == "__main__":
    unittest.main()
