"""No-network AI4I normalization tests with synthetic CSV fixtures only."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from analysis.ai4i_batch import (
    CSV_FIELDS,
    Ai4iBatchError,
    prepare_ai4i_rows,
    write_new_jsonl,
)

STAMP = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)


class Ai4iBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.path = self.folder / "ai4i.csv"
        self.records = [
            {
                "UDI": str(i + 1),
                "Product ID": "L471" + str(i),
                "Type": "L",
                "Air temperature [K]": "300.0",
                "Process temperature [K]": "310.0",
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
            for i in range(2)
        ]
        self.save()

    def save(self, headers=CSV_FIELDS):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def prepare(self, **kwargs):
        settings = {
            "expected_csv_sha256": self.digest,
            "run_id": "offline_test_20261017",
            "ingested_at": STAMP,
            "expected_rows": 2,
        }
        settings.update(kwargs)
        return prepare_ai4i_rows(self.path, **settings)

    def test_valid_csv_matches_staging_schema(self):
        rows = self.prepare()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["udi"], 1)
        self.assertEqual(rows[0]["source_sha256"], self.digest)
        self.assertEqual(rows[0]["air_temperature_k"], 300.0)
        self.assertEqual(rows[0]["tool_wear_min"], 120.0)
        self.assertEqual(rows[0]["machine_failure"], 0)
        self.assertEqual(rows[0]["ingestion_run_id"], "offline_test_20261017")

    def test_explicit_utc_timestamp(self):
        timezone_plus_three = timezone(offset=__import__("datetime").timedelta(hours=3))
        stamped = datetime(2026, 10, 17, 17, 0, tzinfo=timezone_plus_three)
        rows = self.prepare(ingested_at=stamped)
        self.assertEqual(rows[0]["ingested_at"], "2026-10-17T14:00:00+00:00")

    def test_checksum_mismatch_fails_before_parsing(self):
        with self.assertRaisesRegex(Ai4iBatchError, "mismatch"):
            self.prepare(expected_csv_sha256="f" * 64)

    def test_missing_checksum_and_bad_run_id_rejected(self):
        with self.assertRaises(Ai4iBatchError):
            self.prepare(expected_csv_sha256="")
        with self.assertRaises(Ai4iBatchError):
            self.prepare(run_id="unsafe; DROP TABLE")

    def test_mismatched_row_count_rejected(self):
        with self.assertRaisesRegex(Ai4iBatchError, "row count"):
            self.prepare(expected_rows=3)
        with self.assertRaisesRegex(Ai4iBatchError, "More source records"):
            self.prepare(expected_rows=1)

    def test_duplicate_udi_rejected(self):
        self.records[1]["UDI"] = self.records[0]["UDI"]
        self.save()
        with self.assertRaisesRegex(Ai4iBatchError, "Duplicate UDI"):
            self.prepare()

    def test_missing_or_unexpected_header_rejected(self):
        self.save(headers=CSV_FIELDS[:-1])
        with self.assertRaisesRegex(Ai4iBatchError, "fields"):
            self.prepare()
        self.save(headers=(*CSV_FIELDS, "Untrusted Extra"))
        with self.assertRaisesRegex(Ai4iBatchError, "fields"):
            self.prepare()

    def test_invalid_numeric_and_target_rejected(self):
        self.records[0]["Torque [Nm]"] = "NaN"
        self.save()
        with self.assertRaisesRegex(Ai4iBatchError, "Nonfinite"):
            self.prepare()
        self.records[0]["Torque [Nm]"] = "40"
        self.records[1]["Machine failure"] = "2"
        self.save()
        with self.assertRaisesRegex(Ai4iBatchError, "must be 0 or 1"):
            self.prepare()

    def test_invalid_product_type_rejected(self):
        self.records[1]["Type"] = "Z"
        self.save()
        with self.assertRaisesRegex(Ai4iBatchError, "product type"):
            self.prepare()

    def test_naive_ingestion_timestamp_rejected(self):
        with self.assertRaisesRegex(Ai4iBatchError, "timezone-aware"):
            self.prepare(ingested_at=datetime(2026, 10, 17))

    def test_existing_export_must_not_be_overwritten(self):
        output = self.folder / "batch.jsonl"
        output.write_text("protected", encoding="utf-8")
        with self.assertRaisesRegex(Ai4iBatchError, "exists"):
            write_new_jsonl(self.prepare(), output)
        self.assertEqual(output.read_text(encoding="utf-8"), "protected")

    def test_new_jsonl_export_is_local_and_reproducible(self):
        output = self.folder / "batch.jsonl"
        summary = write_new_jsonl(self.prepare(), output)
        self.assertEqual(summary["records"], 2)
        self.assertFalse(summary["cloud_upload_performed"])
        self.assertEqual(summary["export_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1]["udi"], 2)

    def test_source_symlink_refused(self):
        target = self.folder / "real.csv"
        self.path.rename(target)
        try:
            self.path.symlink_to(target)
        except OSError:
            self.skipTest("Host cannot create symlinks")
        with self.assertRaisesRegex(Ai4iBatchError, "non-symlink"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
