"""Offline FD001 engine-history contract tests without model weights or network."""

import json
import math
import unittest
from pathlib import Path

from analysis.rul_input_contract import (
    DROPPED_FD001_SENSORS,
    INFORMATIVE_SENSORS,
    MODEL_COLUMNS,
    RAW_COLUMNS,
    RulInputError,
    build_fd001_feature_row,
)


def history(cycles=7):
    values = []
    for cycle in range(1, cycles + 1):
        row = {
            "unit_number": 1,
            "time_cycles": cycle,
            "setting_1": 0.1,
            "setting_2": 0.2,
            "setting_3": 0.3,
        }
        row.update({f"sensor_{i}": float(i + cycle) for i in range(1, 22)})
        values.append(row)
    return values


class RulFeatureContractTests(unittest.TestCase):
    def test_fd001_schema_produces_ordered_48_features(self):
        output = build_fd001_feature_row(history())
        self.assertEqual(len(RAW_COLUMNS), 26)
        self.assertEqual(len(INFORMATIVE_SENSORS), 15)
        self.assertEqual(len(MODEL_COLUMNS), 48)
        self.assertEqual(tuple(output["model_features"]), MODEL_COLUMNS)
        self.assertFalse(output["is_live_plant_data"])
        self.assertFalse(output["automated_maintenance_action"])

    def test_selected_sensors_match_committed_fd001_evidence(self):
        root = Path(__file__).resolve().parents[1]
        summary_path = root / "docs/data/cmapss-rul-case-study/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(summary["sensors"]["dropped_near_constant"]),
            DROPPED_FD001_SENSORS,
        )
        self.assertEqual(summary["sensors"]["kept_informative"], len(INFORMATIVE_SENSORS))

    def test_rolling_features_use_only_last_five_cycles(self):
        result = build_fd001_feature_row(history(7))
        self.assertEqual(result["last_cycle"], 7)
        features = result["model_features"]
        self.assertEqual(features["sensor_2"], 9.0)
        self.assertAlmostEqual(features["sensor_2_roll_mean"], 7.0)
        self.assertAlmostEqual(features["sensor_2_roll_std"], math.sqrt(2.5))

    def test_short_history_rejected(self):
        with self.assertRaisesRegex(RulInputError, "five contiguous"):
            build_fd001_feature_row(history(4))

    def test_engine_identity_mismatch_rejected(self):
        rows = history()
        rows[-1]["unit_number"] = 2
        with self.assertRaisesRegex(RulInputError, "mixed engine"):
            build_fd001_feature_row(rows)

    def test_duplicate_or_skipped_cycle_rejected(self):
        for cycle in (4, 7):
            rows = history()
            rows[4]["time_cycles"] = cycle
            with self.subTest(cycle=cycle):
                with self.assertRaisesRegex(RulInputError, "strictly increasing"):
                    build_fd001_feature_row(rows)

    def test_nonfinite_values_rejected(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            rows = history()
            rows[2]["sensor_5"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RulInputError, "finite"):
                    build_fd001_feature_row(rows)

    def test_missing_and_unexpected_fields_rejected(self):
        rows = history()
        rows[1].pop("sensor_20")
        with self.assertRaisesRegex(RulInputError, "exactly 26"):
            build_fd001_feature_row(rows)
        rows = history()
        rows[1]["predicted_rul"] = 125
        with self.assertRaisesRegex(RulInputError, "exactly 26"):
            build_fd001_feature_row(rows)

    def test_rejects_boolean_ids_and_sensor_values(self):
        rows = history()
        rows[0]["unit_number"] = True
        with self.assertRaisesRegex(RulInputError, "positive integer"):
            build_fd001_feature_row(rows)
        rows = history()
        rows[0]["sensor_1"] = False
        with self.assertRaisesRegex(RulInputError, "numeric"):
            build_fd001_feature_row(rows)

    def test_no_timestamp_or_failure_labels_are_fabricated(self):
        result = build_fd001_feature_row(history())
        self.assertNotIn("RUL", result)
        self.assertNotIn("predicted_rul", result)
        self.assertNotIn("recorded_at", result)
        self.assertEqual(result["source"], "NASA C-MAPSS FD001 simulated benchmark")


if __name__ == "__main__":
    unittest.main()
