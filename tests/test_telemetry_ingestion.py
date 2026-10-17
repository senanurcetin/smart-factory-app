"""Offline contract tests: no broker, PLC, network or model-weight dependency."""

import json
import unittest
from datetime import datetime, timezone

from analysis.telemetry_ingestion import (
    MODEL_FIELDS,
    TelemetryReplayWindow,
    TelemetryValidationError,
    parse_mqtt_message,
    parse_opcua_snapshot,
    parse_record,
)


def sample_payload():
    return {
        "asset_id": "CNC-001",
        "recorded_at": "2026-10-17T12:00:00+03:00",
        "Type": "L",
        "Air temperature [K]": 299.4,
        "Process temperature [K]": 309.2,
        "Rotational speed [rpm]": 1550,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 100,
    }


class TelemetryInputTests(unittest.TestCase):
    def test_simulated_contract_preserves_model_inputs(self):
        payload = sample_payload()
        got = parse_record(payload)
        self.assertEqual(got.source, "simulated")
        self.assertEqual(got.recorded_at.isoformat(), "2026-10-17T09:00:00+00:00")
        self.assertEqual(tuple(got.model_features()), MODEL_FIELDS)
        self.assertEqual(got.model_features()["Torque [Nm]"], 42.0)
        self.assertEqual(payload["Torque [Nm]"], 42.0)

    def test_mqtt_topic_and_asset_must_agree(self):
        encoded = json.dumps(sample_payload()).encode("utf-8")
        got = parse_mqtt_message("smartfactory/telemetry/CNC-001", encoded)
        self.assertEqual(got.source, "mqtt")
        self.assertEqual(got.asset_id, "CNC-001")
        for topic in (
            "smartfactory/telemetry/CNC-002",
            "smartfactory/telemetry/+/CNC-001",
            "untrusted/CNC-001",
        ):
            with self.subTest(topic=topic):
                with self.assertRaises(TelemetryValidationError):
                    parse_mqtt_message(topic, encoded)

    def test_opcua_normalizes_readings_without_plant_access(self):
        payload = sample_payload()
        readings = {field: payload[field] for field in MODEL_FIELDS}
        sample = parse_opcua_snapshot("CNC-001", payload["recorded_at"], readings)
        self.assertEqual(sample.source, "opcua")
        self.assertEqual(sample.model_features()["Rotational speed [rpm]"], 1550.0)

    def test_invalid_and_nonfinite_numeric_values_fail_closed(self):
        for bad in (True, None, "1500", float("nan"), float("inf"), -1, 50001):
            payload = sample_payload()
            payload["Rotational speed [rpm]"] = bad
            with self.subTest(value=str(bad)):
                with self.assertRaises(TelemetryValidationError):
                    parse_record(payload)

    def test_source_is_not_trusted_from_external_payload(self):
        payload = sample_payload()
        payload["source"] = "opcua"
        with self.assertRaises(TelemetryValidationError):
            parse_record(payload)

    def test_rejects_missing_extra_fields_and_unsafe_asset_ids(self):
        for mutation in ("missing", "extra", "bad_id"):
            payload = sample_payload()
            if mutation == "missing":
                payload.pop("Torque [Nm]")
            elif mutation == "extra":
                payload["Target"] = 1
            else:
                payload["asset_id"] = "CNC-001/../../bad"
            with self.subTest(mutation=mutation):
                with self.assertRaises(TelemetryValidationError):
                    parse_record(payload)

    def test_naive_or_invalid_timestamp_rejected(self):
        for stamp in ("2026-10-17T12:00:00", "not-a-date", None):
            payload = sample_payload()
            payload["recorded_at"] = stamp
            with self.subTest(stamp=stamp):
                with self.assertRaises(TelemetryValidationError):
                    parse_record(payload)

    def test_json_duplicate_fields_and_oversize_payload_rejected(self):
        raw = json.dumps(sample_payload())
        raw = raw.replace('"Type": "L"', '"Type": "L", "Type": "H"')
        with self.assertRaises(TelemetryValidationError):
            parse_record(raw)
        with self.assertRaises(TelemetryValidationError):
            parse_record(" " * 8193)

    def test_bad_type_never_causes_internal_type_error(self):
        for value in ([], {}, None, 1):
            payload = sample_payload()
            payload["Type"] = value
            with self.subTest(value=str(value)):
                with self.assertRaises(TelemetryValidationError):
                    parse_record(payload)

    def test_opcua_metadata_cannot_override_caller_identity(self):
        readings = {field: sample_payload()[field] for field in MODEL_FIELDS}
        readings["asset_id"] = "INTRUDER"
        with self.assertRaises(TelemetryValidationError):
            parse_opcua_snapshot("CNC-001", sample_payload()["recorded_at"], readings)

    def test_replay_window_deduplicates_and_rejects_older_timestamps(self):
        sample = parse_record(sample_payload())
        window = TelemetryReplayWindow(max_assets=1)
        self.assertTrue(window.accept(sample))
        self.assertFalse(window.accept(sample))
        earlier = sample_payload()
        earlier["recorded_at"] = "2026-10-17T08:00:00Z"
        self.assertFalse(window.accept(parse_record(earlier)))
        another = sample_payload()
        another["asset_id"] = "CNC-002"
        with self.assertRaises(TelemetryValidationError):
            window.accept(parse_record(another))

    def test_newer_sample_accepted_for_same_asset(self):
        window = TelemetryReplayWindow()
        self.assertTrue(window.accept(parse_record(sample_payload())))
        newer = sample_payload()
        newer["recorded_at"] = "2026-10-17T12:00:01+03:00"
        self.assertTrue(window.accept(parse_record(newer)))
        self.assertEqual(
            parse_record(newer).recorded_at,
            datetime(2026, 10, 17, 9, 0, 1, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
