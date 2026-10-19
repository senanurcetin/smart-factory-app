"""No-network end-to-end acceptance for one-shot local benchmark staging."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from analysis.ai4i_batch import CSV_FIELDS, Ai4iBatchError
from analysis.fd001_batch import Fd001BatchError
from analysis.stage_benchmarks import stage_ai4i, stage_fd001

STAMP = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)


def fd_cycle(unit, cycle):
    return f"{unit} {cycle} " + " ".join("0.1" for _ in range(24)) + "\n"


class StageBenchmarksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.out.mkdir()
        self.ai4i_csv = self.root / "ai4i.csv"
        row = dict(
            zip(
                CSV_FIELDS,
                [
                    "1",
                    "L1234",
                    "L",
                    "299",
                    "310",
                    "1500",
                    "40",
                    "100",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                ],
            )
        )
        with self.ai4i_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        self.paths = {
            "train": self.root / "train_FD001.txt",
            "test": self.root / "test_FD001.txt",
            "truth": self.root / "RUL_FD001.txt",
        }
        self.paths["train"].write_text(fd_cycle(1, 1) + fd_cycle(1, 2), encoding="utf-8")
        self.paths["test"].write_text(fd_cycle(1, 1) + fd_cycle(1, 2), encoding="utf-8")
        self.paths["truth"].write_text("17\n", encoding="utf-8")

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def ai4i(self, **changes):
        args = {
            "run_id": "test_001",
            "evaluated_at": STAMP,
            "project": "workintech-working",
            "expected_rows": 1,
        }
        args.update(changes)
        return stage_ai4i(self.ai4i_csv, self.sha(self.ai4i_csv), self.out, **args)

    def fd001(self, **changes):
        args = {
            "run_id": "test_001",
            "evaluated_at": STAMP,
            "project": "workintech-working",
            "expected_train_rows": 2,
            "expected_train_units": 1,
            "expected_test_units": 1,
        }
        args.update(changes)
        return stage_fd001(
            self.paths,
            {k: self.sha(path) for k, path in self.paths.items()},
            self.out,
            **args,
        )

    def test_ai4i_local_staging_passes_full_ddl_preflight(self):
        result = self.ai4i()
        self.assertEqual(result["status"], "local_export_preflight_pass")
        self.assertEqual(result["preflight"]["rows"], 1)
        self.assertFalse(result["preflight"]["cloud_upload_performed"])
        data = json.loads((self.out / "ai4i_records.jsonl").read_text())
        self.assertEqual(data["udi"], 1)

    def test_fd001_local_staging_passes_two_table_preflights(self):
        result = self.fd001()
        self.assertEqual(result["status"], "local_export_preflight_pass")
        self.assertEqual(len(result["preflights"]), 2)
        self.assertEqual(
            {entry["table"] for entry in result["preflights"]},
            {"fd001_cycles", "fd001_official_test_truth"},
        )
        self.assertTrue(all(not entry["cloud_upload_performed"] for entry in result["preflights"]))

    def test_no_export_overwrite_on_second_run(self):
        self.ai4i()
        with self.assertRaisesRegex(Ai4iBatchError, "already exists"):
            self.ai4i()

    def test_fd001_wrong_official_truth_fails_before_export(self):
        self.paths["truth"].write_text("17\n18\n", encoding="utf-8")
        with self.assertRaisesRegex(Fd001BatchError, "truth count"):
            self.fd001()
        self.assertFalse((self.out / "fd001_cycles.jsonl").exists())

    def test_stage_requires_valid_project_before_writing(self):
        with self.assertRaises(ValueError):
            self.ai4i(project="evil.project\u0060")
        self.assertFalse((self.out / "ai4i_records.jsonl").exists())

    def test_empty_local_output_directory_must_already_exist(self):
        self.out.rmdir()
        with self.assertRaises(Ai4iBatchError):
            self.ai4i()

    def test_fd001_negative_run_id_fails_before_export(self):
        with self.assertRaisesRegex(Fd001BatchError, "run ID"):
            self.fd001(run_id="bad;DROP")
        self.assertFalse((self.out / "fd001_cycles.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
