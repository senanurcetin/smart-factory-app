"""Synthetic offline source-vs-Kaggle mirror parity contracts."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from analysis.ai4i_batch import CSV_FIELDS
from analysis.kaggle_parity import KaggleParityError, verify_ai4i_mirror


class KaggleParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.official = self.root / "official.csv"
        self.mirror = self.root / "mirror.csv"
        self.rows = []
        for i in (1, 2):
            self.rows.append(
                dict(
                    zip(
                        CSV_FIELDS,
                        [
                            str(i),
                            "L2026" + str(i),
                            "L",
                            "300",
                            "310",
                            "1500",
                            "42",
                            str(90 + i),
                            str(i - 1),
                            str(i - 1),
                            "0",
                            "0",
                            "0",
                            "0",
                        ],
                    )
                )
            )
        self.write(self.official, self.rows)
        self.write(self.mirror, self.rows)

    def write(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self, **changes):
        args = {
            "official_sha256": self.sha(self.official),
            "mirror_sha256": self.sha(self.mirror),
            "kaggle_handle": "candidate-owner/ai4i-2020",
            "kaggle_version": 4,
            "expected_rows": 2,
        }
        args.update(changes)
        return verify_ai4i_mirror(self.official, self.mirror, **args)

    def test_identical_rows_verify_offline_without_claiming_provider(self):
        result = self.check()
        self.assertEqual(result["status"], "offline_normalized_rows_identical")
        self.assertEqual(result["compared_rows"], 2)
        self.assertTrue(result["label_parity_checked"])
        self.assertFalse(result["provider_dataset_version_verified"])
        self.assertFalse(result["cloud_ingestion_authorized"])

    def test_reordered_rows_are_accepted_by_stable_udi(self):
        self.write(self.mirror, list(reversed(self.rows)))
        result = self.check()
        self.assertEqual(result["compared_rows"], 2)
        self.assertNotEqual(result["mirror_csv_sha256"], result["official_csv_sha256"])

    def test_modified_failure_label_rejected(self):
        altered = [dict(row) for row in self.rows]
        altered[1]["Machine failure"] = "0"
        self.write(self.mirror, altered)
        with self.assertRaisesRegex(KaggleParityError, "rows differ"):
            self.check()

    def test_modified_sensor_rejected(self):
        altered = [dict(row) for row in self.rows]
        altered[1]["Torque [Nm]"] = "88"
        self.write(self.mirror, altered)
        with self.assertRaisesRegex(KaggleParityError, "rows differ"):
            self.check()

    def test_missing_row_rejected(self):
        self.write(self.mirror, self.rows[:-1])
        with self.assertRaisesRegex(KaggleParityError, "source checks"):
            self.check()

    def test_missing_or_unversioned_kaggle_reference_rejected(self):
        with self.assertRaisesRegex(KaggleParityError, "owner/dataset"):
            self.check(kaggle_handle="unknown")
        with self.assertRaisesRegex(KaggleParityError, "positive Kaggle version"):
            self.check(kaggle_version=0)

    def test_license_flag_never_auto_authorizes_upload(self):
        result = self.check(license_reviewed=True)
        self.assertTrue(result["license_reviewed"])
        self.assertFalse(result["cloud_ingestion_authorized"])

    def test_same_source_file_cannot_masquerade_as_independent_mirror(self):
        with self.assertRaisesRegex(KaggleParityError, "independent"):
            verify_ai4i_mirror(
                self.official,
                self.official,
                official_sha256=self.sha(self.official),
                mirror_sha256=self.sha(self.official),
                kaggle_handle="candidate-owner/ai4i-2020",
                kaggle_version=1,
                expected_rows=2,
            )

    def test_digest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(KaggleParityError, "source checks"):
            self.check(mirror_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
