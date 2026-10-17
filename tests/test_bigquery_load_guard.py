"""No-network BigQuery staging preflight checks using synthetic AI4I data."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from analysis.ai4i_batch import CSV_FIELDS, prepare_ai4i_rows, write_new_jsonl
from analysis.bigquery_load_guard import LoadPreflightError, validate_batch_upload


class BigQueryLoadGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        csv_file = self.folder / "ai4i.csv"
        self.row = {
            "UDI": "1",
            "Product ID": "L4711",
            "Type": "L",
            "Air temperature [K]": "300",
            "Process temperature [K]": "310",
            "Rotational speed [rpm]": "1500",
            "Torque [Nm]": "40",
            "Tool wear [min]": "120",
            "Machine failure": "0",
            "TWF": "0",
            "HDF": "0",
            "PWF": "0",
            "OSF": "0",
            "RNF": "0",
        }
        with csv_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(self.row)
        csv_sha = hashlib.sha256(csv_file.read_bytes()).hexdigest()
        rows = prepare_ai4i_rows(
            csv_file,
            expected_csv_sha256=csv_sha,
            run_id="integration_test",
            ingested_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
            expected_rows=1,
        )
        self.export = self.folder / "ready.jsonl"
        write_new_jsonl(rows, self.export)
        self.source_sha = csv_sha

    def validate(self, **overrides):
        args = {
            "expected_file_sha256": hashlib.sha256(self.export.read_bytes()).hexdigest(),
            "expected_source_sha256": self.source_sha,
            "expected_rows": 1,
            "project": "workintech-working",
            "table": "ai4i_records",
        }
        args.update(overrides)
        return validate_batch_upload(self.export, **args)

    def rewrite(self, changes):
        record = json.loads(self.export.read_text(encoding="utf-8"))
        record.update(changes)
        self.export.write_text(
            json.dumps(record, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_checked_jsonl_has_no_cloud_side_effect(self):
        result = self.validate()
        self.assertEqual(result["status"], "offline_preflight_pass")
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["planned_target"], "workintech-working.smart_factory_raw.ai4i_records")
        self.assertFalse(result["cloud_upload_performed"])
        self.assertFalse(result["provider_dry_run_completed"])
        self.assertFalse(result["cloud_cost_authorized"])

    def test_file_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(LoadPreflightError, "checksum mismatch"):
            self.validate(expected_file_sha256="f" * 64)

    def test_incorrect_source_hash_rejected(self):
        with self.assertRaisesRegex(LoadPreflightError, "source identity"):
            self.validate(expected_source_sha256="f" * 64)

    def test_unapproved_table_rejected(self):
        with self.assertRaisesRegex(LoadPreflightError, "outside approved"):
            self.validate(table="ai4i_model_scores")

    def test_unpinned_checksum_rejected(self):
        with self.assertRaisesRegex(LoadPreflightError, "Pinned file"):
            self.validate(expected_file_sha256="")

    def test_lost_label_or_added_field_rejected(self):
        for changes in ({"machine_failure": None}, {"new_model_prediction": 0.95}):
            with self.subTest(changes=changes):
                self.rewrite(changes)
                with self.assertRaises(LoadPreflightError):
                    self.validate()

    def test_boolean_numeric_field_rejected(self):
        self.rewrite({"machine_failure": True})
        with self.assertRaisesRegex(LoadPreflightError, "INT64"):
            self.validate()

    def test_duplicate_key_and_extra_newline_rejected(self):
        self.export.write_text(
            '{"source_sha256":"x","source_sha256":"y"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(LoadPreflightError, "duplicate-key"):
            self.validate()

    def test_invalid_timestamp_timezone_rejected(self):
        self.rewrite({"ingested_at": "2026-10-17T14:00:00"})
        with self.assertRaisesRegex(LoadPreflightError, "timezone"):
            self.validate()

    def test_wrong_number_of_rows_rejected(self):
        with self.assertRaisesRegex(LoadPreflightError, "row count"):
            self.validate(expected_rows=2)
        with self.assertRaisesRegex(LoadPreflightError, "More rows"):
            self.validate(expected_rows=0 + 1, expected_file_sha256="e" * 64)

    def test_duplicate_natural_key_rejected(self):
        single = self.export.read_bytes()
        self.export.write_bytes(single + single)
        with self.assertRaisesRegex(LoadPreflightError, "Duplicate natural key"):
            self.validate(expected_rows=2)

    def test_bad_project_identifier_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(project="bad.project;DROP")

    def test_malformed_ndjson_line_rejected(self):
        self.export.write_bytes(b'{"broken_json":\n')
        with self.assertRaisesRegex(LoadPreflightError, "Invalid or duplicate-key"):
            self.validate()

    def test_non_newline_terminated_file_rejected(self):
        self.export.write_bytes(self.export.read_bytes().rstrip(b"\n"))
        with self.assertRaisesRegex(LoadPreflightError, "end with newline"):
            self.validate()

    def test_symlink_source_file_rejected(self):
        original = self.folder / "real.jsonl"
        self.export.rename(original)
        try:
            self.export.symlink_to(original)
        except OSError:
            self.skipTest("Host does not allow symlinks")
        with self.assertRaisesRegex(LoadPreflightError, "non-symlink"):
            self.validate()

    def test_one_ingestion_run_only(self):
        self.rewrite({"ingestion_run_id": ""})
        with self.assertRaisesRegex(LoadPreflightError, "missing ingestion"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
