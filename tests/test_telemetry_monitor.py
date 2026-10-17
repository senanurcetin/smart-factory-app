"""Offline advisory-monitor checks; never connect to plant endpoints."""

import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from analysis.telemetry_ingestion import parse_record
from analysis.telemetry_monitor import evaluate_telemetry_window

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def synthetic_reference(rows=100):
    return pd.DataFrame(
        {
            "Type": [("L", "M", "H")[i % 3] for i in range(rows)],
            "Air temperature [K]": [295 + (i % 10) * 0.5 for i in range(rows)],
            "Process temperature [K]": [306 + (i % 10) * 0.65 for i in range(rows)],
            "Rotational speed [rpm]": [1400 + (i % 10) * 20 for i in range(rows)],
            "Torque [Nm]": [40 + (i % 10) for i in range(rows)],
            "Tool wear [min]": [20 + (i % 10) * 4 for i in range(rows)],
        }
    )


def synthetic_window(frame, *, wear_shift=0, source="simulated"):
    samples = []
    for i, row in frame.iterrows():
        payload = {
            "asset_id": "CNC-001",
            "recorded_at": (NOW - timedelta(seconds=i)).isoformat(),
            **row.to_dict(),
        }
        payload["Tool wear [min]"] += wear_shift
        samples.append(parse_record(payload, source=source))
    return samples


class TelemetryMonitorTests(unittest.TestCase):
    def setUp(self):
        self.reference = synthetic_reference()
        self.samples = synthetic_window(self.reference)

    def test_identical_offline_distributions_are_ok(self):
        result = evaluate_telemetry_window(self.reference, self.samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["alert_features"], [])
        self.assertEqual(result["source"], "simulated")
        self.assertEqual(result["action"], "advisory_only_no_plant_control")
        self.assertIn("simulated telemetry window", result["report"]["scenario"])
        self.assertNotIn("+80", result["report"]["scenario"])

    def test_external_source_is_not_mislabelled_as_synthetic_benchmark(self):
        samples = synthetic_window(self.reference, source="mqtt")
        result = evaluate_telemetry_window(self.reference, samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "ok")
        self.assertIn("mqtt telemetry window", result["report"]["scenario"])
        self.assertNotIn("+80", result["report"]["scenario"])

    def test_sensor_shift_generates_advisory_alert(self):
        samples = synthetic_window(self.reference, wear_shift=200)
        result = evaluate_telemetry_window(self.reference, samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "drift_alert")
        self.assertIn("Tool wear [min]", {item["feature"] for item in result["alert_features"]})

    def test_too_few_observations_are_not_healthy(self):
        result = evaluate_telemetry_window(self.reference, self.samples[:2], evaluated_at=NOW)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["errors"], ["window_insufficient_records"])
        self.assertNotIn("report", result)

    def test_stale_batch_is_data_quality_error(self):
        result = evaluate_telemetry_window(
            self.reference,
            self.samples,
            evaluated_at=NOW + timedelta(hours=2),
        )
        self.assertEqual(result["status"], "data_quality_error")
        self.assertEqual(result["errors"], ["stale_timestamp"])

    def test_future_observation_is_data_quality_error(self):
        result = evaluate_telemetry_window(
            self.reference,
            self.samples,
            evaluated_at=NOW - timedelta(hours=1),
        )
        self.assertEqual(result["status"], "data_quality_error")
        self.assertEqual(result["errors"], ["future_timestamp"])

    def test_mixed_provenance_rejected(self):
        samples = self.samples.copy()
        samples[0] = synthetic_window(self.reference.iloc[:1], source="mqtt")[0]
        result = evaluate_telemetry_window(self.reference, samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "data_quality_error")
        self.assertEqual(result["errors"], ["mixed_simulated_and_external_sources"])

    def test_duplicate_asset_timestamp_rejected(self):
        result = evaluate_telemetry_window(
            self.reference,
            self.samples + [self.samples[0]],
            evaluated_at=NOW,
        )
        self.assertEqual(result["status"], "data_quality_error")
        self.assertEqual(result["errors"], ["duplicate_asset_timestamp"])

    def test_invalid_reference_numeric_rejected(self):
        bad = self.reference.copy()
        bad.loc[0, "Torque [Nm]"] = float("nan")
        result = evaluate_telemetry_window(bad, self.samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "data_quality_error")
        self.assertTrue(result["errors"][0].startswith("reference_invalid_numeric:"))

    def test_constant_shifted_reference_never_silent_ok(self):
        baseline = self.reference.copy()
        baseline["Tool wear [min]"] = 10.0
        result = evaluate_telemetry_window(baseline, self.samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "data_quality_error")
        self.assertIn("unusable_constant_reference:Tool wear [min]", result["errors"])

    def test_missing_reference_column_rejected(self):
        baseline = self.reference.drop(columns=["Torque [Nm]"])
        result = evaluate_telemetry_window(baseline, self.samples, evaluated_at=NOW)
        self.assertEqual(result["status"], "data_quality_error")
        self.assertTrue(result["errors"][0].startswith("reference_missing_fields:"))

    def test_configuration_requires_positive_window_and_aware_time(self):
        with self.assertRaises(ValueError):
            evaluate_telemetry_window(
                self.reference,
                self.samples,
                evaluated_at=NOW,
                min_records=1,
            )
        with self.assertRaises(ValueError):
            evaluate_telemetry_window(
                self.reference,
                self.samples,
                evaluated_at=datetime(2026, 10, 17, 12, 0),
            )


if __name__ == "__main__":
    unittest.main()
