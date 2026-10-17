"""Mocked BigQuery provider dry-run tests; no cloud SDK or network required."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from analysis.bigquery_cloud_read import (
    EXECUTION_ENV,
    CloudReadError,
    inspect_quality_views,
)
from analysis.bigquery_plan import MAXIMUM_BYTES_BILLED

SHA = "a" * 64


class FakeBigQuerySdk:
    class QueryJobConfig:
        def __init__(self, **options):
            self.options = options

    class ScalarQueryParameter:
        def __init__(self, name, field_type, value):
            self.name = name
            self.field_type = field_type
            self.value = value


class FakeJob:
    total_bytes_processed = 1024

    def result(self, max_results):
        assert max_results == 1
        return [{"model_sha256": "b" * 64, "holdout_rows": 2000}]


class FakeBigQueryClient:
    def __init__(self):
        self.datasets = []
        self.tables = []
        self.calls = []
        self.location = "EU"
        self.view_type = "VIEW"
        self.estimated_bytes = 1024

    def get_dataset(self, name):
        self.datasets.append(name)
        return SimpleNamespace(location=self.location)

    def get_table(self, name):
        self.tables.append(name)
        return SimpleNamespace(table_type=self.view_type)

    def query(self, sql, *, job_config, location):
        self.calls.append((sql, job_config.options, location))
        return SimpleNamespace(
            total_bytes_processed=self.estimated_bytes,
            result=FakeJob().result,
        )


class BigQueryCloudReadTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeBigQueryClient()

    def inspect(self, **changes):
        settings = {
            "project": "workintech-working",
            "source_sha256": SHA,
        }
        settings.update(changes)
        return inspect_quality_views(self.client, FakeBigQuerySdk, **settings)

    def test_only_read_only_provider_dry_runs_are_submitted(self):
        output = self.inspect()
        self.assertEqual(output["status"], "provider_dry_run_passed")
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(len(self.client.datasets), 2)
        self.assertTrue(all(c[1]["dry_run"] for c in self.client.calls))
        self.assertTrue(all(not r["report_query_executed"] for r in output["reports"]))
        self.assertTrue(output["read_only"])
        self.assertFalse(output["cloud_write_performed"])

    def test_source_digest_is_parameterized_not_in_sql_text(self):
        self.inspect()
        for sql, config, _ in self.client.calls:
            self.assertNotIn(SHA, sql)
            self.assertIn("@source_sha256", sql)
            self.assertEqual(config["query_parameters"][0].value, SHA)

    def test_billing_cap_is_passed_to_provider(self):
        self.inspect(max_bytes=12345)
        self.assertEqual(
            [call[1]["maximum_bytes_billed"] for call in self.client.calls],
            [12345, 12345],
        )

    def test_dry_run_estimate_above_cap_rejected(self):
        self.client.estimated_bytes = 1200
        with self.assertRaisesRegex(CloudReadError, "exceeds byte cap"):
            self.inspect(max_bytes=1000)

    def test_provider_without_estimate_rejected(self):
        self.client.estimated_bytes = None
        with self.assertRaisesRegex(CloudReadError, "no estimate"):
            self.inspect()

    def test_wrong_dataset_location_rejected_without_query(self):
        self.client.location = "US"
        with self.assertRaisesRegex(CloudReadError, "location mismatch"):
            self.inspect()
        self.assertEqual(self.client.calls, [])

    def test_missing_or_wrong_report_view_rejected(self):
        self.client.view_type = "TABLE"
        with self.assertRaisesRegex(CloudReadError, "read-only"):
            self.inspect()
        self.assertEqual(self.client.calls, [])

    def test_malformed_source_sha_rejected_before_cloud(self):
        with self.assertRaisesRegex(CloudReadError, "source SHA-256"):
            self.inspect(source_sha256="bad")
        self.assertEqual(self.client.datasets, [])

    def test_overly_permissive_byte_cap_rejected(self):
        with self.assertRaisesRegex(CloudReadError, "Invalid maximum"):
            self.inspect(max_bytes=MAXIMUM_BYTES_BILLED + 1)
        self.assertEqual(self.client.calls, [])

    def test_billable_reads_rejected_without_opt_in(self):
        with patch.dict(os.environ, {EXECUTION_ENV: ""}):
            with self.assertRaisesRegex(CloudReadError, "not enabled"):
                self.inspect(execute=True)
        self.assertEqual(self.client.calls, [])

    def test_explicit_opt_in_executes_only_bounded_two_selects(self):
        with patch.dict(os.environ, {EXECUTION_ENV: "YES"}):
            output = self.inspect(execute=True)
        self.assertEqual(len(self.client.calls), 4)
        self.assertEqual(
            [config["dry_run"] for _, config, _ in self.client.calls],
            [True, False, True, False],
        )
        self.assertEqual(len(output["reports"]), 2)
        self.assertTrue(all(r["report_query_executed"] for r in output["reports"]))
        self.assertFalse(output["cloud_write_performed"])

    def test_non_boolean_execute_rejected(self):
        with self.assertRaisesRegex(CloudReadError, "execute must"):
            self.inspect(execute="yes")
        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()
