"""Offline FD001 cycle/truth fidelity checks. No downloads, cloud or models."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis.bigquery_plan import build_sql_plan
from analysis.fd001_batch import (
    Fd001BatchError,
    parse_cycles,
    parse_official_truth,
    prepare_fd001_batches,
    write_fd001_jsonl,
)

STAMP = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)
RAW_LABELS = ("setting_1", "setting_2", "setting_3") + tuple(
    f"sensor_{i}" for i in range(1, 22)
)


def raw_cycle(unit: int, cycle: int) -> str:
    values = " ".join(str(0.1 * i) for i in range(1, 25))
    return f"{unit} {cycle} {values}\n"


class Fd001BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.paths = {
            "train": self.dir / "train_FD001.txt",
            "test": self.dir / "test_FD001.txt",
            "truth": self.dir / "RUL_FD001.txt",
        }
        self.write(
            "train",
            "".join(raw_cycle(unit, cycle) for unit in (1, 2) for cycle in (1, 2, 3)),
        )
        self.write(
            "test", "".join(raw_cycle(unit, cycle) for unit in (1, 2) for cycle in (1, 2))
        )
        self.write("truth", "11\n23\n")

    def write(self, key, data):
        self.paths[key].write_text(data, encoding="utf-8")

    def prepare(self, **overrides):
        digests = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in self.paths.items()
        }
        args = {
            "expected_sha256": digests,
            "run_id": "fd001_offline_test",
            "ingested_at": STAMP,
            "expected_train_rows": 6,
            "expected_train_units": 2,
            "expected_test_units": 2,
        }
        args.update(overrides)
        return prepare_fd001_batches(
            self.paths["train"],
            self.paths["test"],
            self.paths["truth"],
            **args,
        )

    def test_official_truth_is_only_one_record_per_test_engine(self):
        result = self.prepare()
        rows = result["cycles"]
        labels = result["official_test_truth"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(labels), 2)
        self.assertEqual(
            [(row["unit_number"], row["true_rul_cycles"]) for row in labels],
            [(1, 11), (2, 23)],
        )
        self.assertNotIn("true_rul_cycles", rows[0])
        self.assertNotIn("RUL", rows[-1])
        self.assertEqual(
            result["evidence"]["official_truth_applies_to"],
            "last_observed_test_cycle_only",
        )

    def test_separate_file_shas_and_shared_bundle_join_key(self):
        result = self.prepare()
        bundle = result["evidence"]["bundle_source_sha256"]
        self.assertEqual(len(bundle), 64)
        self.assertEqual(
            {row["source_sha256"] for row in result["cycles"]}, {bundle}
        )
        self.assertEqual(
            {row["source_sha256"] for row in result["official_test_truth"]},
            {bundle},
        )
        self.assertNotEqual(bundle, result["evidence"]["raw_file_sha256"]["train"])

    def test_fd001_cycles_match_bigquery_ddl_column_names(self):
        result = self.prepare()
        plan = build_sql_plan("workintech-working")
        statement = next(
            item["sql"] for item in plan["statements"] if item["name"] == "fd001_cycles"
        )
        names = set()
        for line in statement.splitlines():
            tokens = line.strip().rstrip(",").split()
            if len(tokens) >= 2 and tokens[1] in ("INT64", "FLOAT64", "STRING", "TIMESTAMP"):
                names.add(tokens[0])
        self.assertEqual(set(result["cycles"][0]), names)
        self.assertEqual(len(RAW_LABELS), 24)

    def test_official_truth_matches_bigquery_ddl_columns(self):
        result = self.prepare()
        plan = build_sql_plan("workintech-working")
        ddl = next(
            item["sql"]
            for item in plan["statements"]
            if item["name"] == "fd001_official_test_truth"
        )
        names = {
            line.strip().split()[0]
            for line in ddl.splitlines()
            if len(line.strip().split()) >= 2
            and line.strip().split()[1] in ("INT64", "FLOAT64", "STRING", "TIMESTAMP")
        }
        self.assertEqual(set(result["official_test_truth"][0]), names)

    def test_missing_file_digest_rejected_before_parsing(self):
        with self.assertRaisesRegex(Fd001BatchError, "pin all three"):
            self.prepare(expected_sha256={"train": "a" * 64})

    def test_corrupted_file_sha_rejected(self):
        digests = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in self.paths.items()
        }
        digests["test"] = "0" * 64
        with self.assertRaisesRegex(Fd001BatchError, "mismatch"):
            self.prepare(expected_sha256=digests)

    def test_short_row_and_malformed_number_rejected(self):
        with self.assertRaisesRegex(Fd001BatchError, "exactly 26"):
            parse_cycles("1 1 0.1\n", "train")
        with self.assertRaisesRegex(Fd001BatchError, "Non-numeric"):
            parse_cycles(raw_cycle(1, 1).replace("0.2", "oops", 1), "train")

    def test_nan_or_infinite_sensor_rejected(self):
        for value in ("nan", "inf", "-inf"):
            source = raw_cycle(1, 1).replace("0.2", value, 1)
            with self.subTest(value=value):
                with self.assertRaisesRegex(Fd001BatchError, "Nonfinite"):
                    parse_cycles(source, "test")

    def test_cycle_gap_and_duplicate_rejected(self):
        for cycle in (2, 4):
            source = raw_cycle(1, 1) + raw_cycle(1, cycle)
            with self.subTest(cycle=cycle):
                with self.assertRaisesRegex(Fd001BatchError, "consecutive"):
                    parse_cycles(source, "train")

    def test_mixed_or_skipped_engine_order_rejected(self):
        with self.assertRaisesRegex(Fd001BatchError, "sequential"):
            parse_cycles(raw_cycle(1, 1) + raw_cycle(3, 1), "test")
        with self.assertRaisesRegex(Fd001BatchError, "start at cycle 1"):
            parse_cycles(raw_cycle(1, 1) + raw_cycle(2, 5), "train")

    def test_truth_row_count_must_match_test_engine_count(self):
        with self.assertRaisesRegex(Fd001BatchError, "differs"):
            parse_official_truth("23\n", 2)
        with self.assertRaisesRegex(Fd001BatchError, "nonnegative"):
            parse_official_truth("-1\n", 1)
        with self.assertRaisesRegex(Fd001BatchError, "one nonnegative"):
            parse_official_truth("12 14\n", 1)

    def test_row_and_unit_count_mismatch_rejected(self):
        with self.assertRaisesRegex(Fd001BatchError, "train row count"):
            self.prepare(expected_train_rows=20)
        with self.assertRaisesRegex(Fd001BatchError, "engine count"):
            self.prepare(expected_test_units=3)

    def test_timezone_aware_utc_conversion(self):
        plus_three = timezone(timedelta(hours=3))
        result = self.prepare(ingested_at=datetime(2026, 10, 17, 17, 0, tzinfo=plus_three))
        self.assertEqual(result["cycles"][0]["ingested_at"], "2026-10-17T14:00:00+00:00")
        with self.assertRaisesRegex(Fd001BatchError, "timezone"):
            self.prepare(ingested_at=datetime(2026, 10, 17, 14, 0))

    def test_create_only_jsonl_exports_and_records_preserved(self):
        result = self.prepare()
        outcome = write_fd001_jsonl(result, self.dir)
        self.assertEqual(len(outcome["export_paths"]), 2)
        self.assertFalse(outcome["cloud_upload_performed"])
        export = self.dir / "fd001_official_test_truth.jsonl"
        lines = [json.loads(line) for line in export.read_text().splitlines()]
        self.assertEqual(lines[0]["true_rul_cycles"], 11)
        with self.assertRaisesRegex(Fd001BatchError, "already exists"):
            write_fd001_jsonl(result, self.dir)

    def test_no_live_plant_or_model_prediction_claim(self):
        evidence = self.prepare()["evidence"]
        self.assertFalse(evidence["cloud_upload_performed"])
        self.assertEqual(evidence["status"], "local_verified_only")
        self.assertNotIn("predicted_rul_cycles", evidence)

    def test_symlink_truth_file_rejected(self):
        original = self.paths["truth"]
        target = self.dir / "real-truth.txt"
        original.rename(target)
        try:
            original.symlink_to(target)
        except OSError:
            self.skipTest("Host does not support symlinks")
        with self.assertRaisesRegex(Fd001BatchError, "non-symlink"):
            self.prepare()

    def test_malicious_run_id_rejected(self):
        with self.assertRaisesRegex(Fd001BatchError, "run ID"):
            self.prepare(run_id="bad;DROP TABLE")
        with self.assertRaisesRegex(Fd001BatchError, "run ID"):
            self.prepare(run_id="")

    def test_invalid_split_rejected(self):
        with self.assertRaisesRegex(Fd001BatchError, "split must"):
            parse_cycles(raw_cycle(1, 1), "production")


if __name__ == "__main__":
    unittest.main()
