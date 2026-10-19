"""Offline BigQuery planning tests: no SDK, credentials, API calls or billing."""

from __future__ import annotations

import unittest

from analysis.bigquery_plan import (
    MAXIMUM_BYTES_BILLED,
    BigQueryPlanError,
    build_sql_plan,
)


class BigQueryPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_sql_plan("workintech-working")

    def test_dry_run_is_local_only_and_cannot_mutate_cloud(self):
        self.assertEqual(self.plan["status"], "offline_unexecuted")
        self.assertFalse(self.plan["remote_bigquery_dry_run_completed"])
        self.assertFalse(self.plan["cloud_ddl_execution_authorized"])
        self.assertIn("OFFLINE PLAN ONLY", self.plan["notice"])

    def test_preconfigured_datasets_are_not_claimed_to_exist(self):
        self.assertEqual(self.plan["datasets"], ["smart_factory_raw", "smart_factory_mart"])
        self.assertEqual(self.plan["project"], "workintech-working")
        self.assertEqual(self.plan["location"], "EU")

    def test_plan_contains_five_tables_and_four_views(self):
        items = self.plan["statements"]
        self.assertEqual(len(items), 11)
        self.assertEqual(sum(s["kind"] == "dataset_ddl" for s in items), 2)
        self.assertEqual(sum(s["kind"] == "table_ddl" for s in items), 5)
        self.assertEqual(sum(s["kind"] == "view_ddl" for s in items), 4)

    def test_ai4i_raw_schema_separates_provenance_and_labels(self):
        schema = next(x["sql"] for x in self.plan["statements"] if x["name"] == "ai4i_records")
        for column in (
            "source_sha256 STRING NOT NULL",
            "ingestion_run_id STRING NOT NULL",
            "udi INT64 NOT NULL",
            "machine_failure INT64",
            "tool_wear_min FLOAT64",
            "ingested_at TIMESTAMP NOT NULL",
        ):
            self.assertIn(column, schema)

    def test_fd001_raw_schema_has_all_21_sensors_and_cycle_ids(self):
        schema = next(x["sql"] for x in self.plan["statements"] if x["name"] == "fd001_cycles")
        self.assertIn("unit_number INT64 NOT NULL", schema)
        self.assertIn("time_cycles INT64 NOT NULL", schema)
        self.assertIn("split STRING NOT NULL", schema)
        for i in range(1, 22):
            self.assertIn("sensor_" + str(i) + " FLOAT64", schema)

    def test_fd001_truth_and_evaluation_join_cannot_invent_labels(self):
        evaluation = next(
            x["sql"] for x in self.plan["statements"] if x["name"] == "fd001_official_evaluation"
        )
        self.assertIn("WHERE split = 'test'", evaluation)
        self.assertIn("true_rul_cycles", evaluation)
        self.assertIn("last_observed_cycle", evaluation)
        self.assertIn("p.source_sha256 = t.source_sha256", evaluation)
        self.assertIn("p.unit_number = t.unit_number", evaluation)

    def test_ai4i_review_queue_uses_model_and_source_fingerprints(self):
        review = next(x["sql"] for x in self.plan["statements"] if x["name"] == "ai4i_review_queue")
        self.assertIn("PARTITION BY model_sha256, source_sha256, udi", review)
        self.assertIn("s.source_sha256 = r.source_sha256", review)
        self.assertIn("ORDER BY scored_at DESC", review)

    def test_ai4i_model_scores_identify_evaluation_split(self):
        schema = next(
            entry["sql"]
            for entry in self.plan["statements"]
            if entry["name"] == "ai4i_model_scores"
        )
        self.assertIn("evaluation_split STRING NOT NULL", schema)

    def test_ai4i_holdout_capture_excludes_train_and_demo(self):
        query = next(
            entry["sql"]
            for entry in self.plan["statements"]
            if entry["name"] == "ai4i_holdout_quality"
        )
        self.assertIn("WHERE evaluation_split = 'holdout'", query)
        self.assertIn("NTILE(10)", query)
        self.assertIn("SAFE_DIVIDE", query)
        self.assertIn("failure_capture_top_decile", query)
        self.assertIn("PARTITION BY model_sha256, source_sha256", query)

    def test_fd001_metrics_use_official_test_truth_only(self):
        query = next(
            entry["sql"]
            for entry in self.plan["statements"]
            if entry["name"] == "fd001_quality_summary"
        )
        self.assertIn("FROM", query)
        self.assertIn("fd001_official_evaluation", query)
        self.assertIn("SQRT(AVG(POW(", query)
        self.assertIn("rmse_cycles", query)
        self.assertIn("mae_cycles", query)

    def test_query_budget_is_explicit_not_claimed_enforced(self):
        self.assertEqual(self.plan["maximum_bytes_billed_per_query"], MAXIMUM_BYTES_BILLED)
        self.assertEqual(MAXIMUM_BYTES_BILLED, 1024**3)
        self.assertFalse(self.plan["cloud_ddl_execution_authorized"])

    def test_custom_dataset_prefix_is_escaped_by_whitelist(self):
        custom = build_sql_plan("data-analiz-490513", prefix="smart_factory_v2")
        self.assertEqual(custom["datasets"], ["smart_factory_v2_raw", "smart_factory_v2_mart"])
        self.assertIn(
            "`data-analiz-490513.smart_factory_v2_raw.ai4i_records`",
            custom["statements"][2]["sql"],
        )

    def test_project_sql_injection_is_rejected(self):
        for malicious in (
            "project.x` DROP TABLE a",
            "a;DELETE",
            "Project_with_caps",
            "",
            "bad project",
            "abc",
        ):
            with self.subTest(value=malicious):
                with self.assertRaises(BigQueryPlanError):
                    build_sql_plan(malicious)

    def test_dataset_prefix_sql_injection_is_rejected(self):
        for malicious in ("a.b", "x`", "bad prefix", "x;DROP", "a" * 239):
            with self.subTest(value=malicious):
                with self.assertRaises(BigQueryPlanError):
                    build_sql_plan("workintech-working", prefix=malicious)

    def test_region_string_injection_is_rejected(self):
        for malicious in ("EU'; DROP TABLE x;--", "europe west", "us/east1", ""):
            with self.subTest(value=malicious):
                with self.assertRaises(BigQueryPlanError):
                    build_sql_plan("workintech-working", location=malicious)

    def test_bigquery_plan_does_not_embed_secrets_or_provider_keys(self):
        combined = "\n".join(s["sql"] for s in self.plan["statements"])
        self.assertNotIn("service_account", combined)
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", combined)
        self.assertNotIn("kaggle.json", combined)
        self.assertNotIn("BEGIN PRIVATE KEY", combined)


if __name__ == "__main__":
    unittest.main()
