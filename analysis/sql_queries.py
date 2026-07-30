"""SQL analysis layer using DuckDB.

Runs analytical queries on the pre-computed JSON artifacts.
DuckDB reads JSON files natively — no dataset download required.
Results are saved to docs/data/ai4i-case-study/sql-analysis.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data" / "ai4i-case-study"
OUT_FILE = DATA_DIR / "sql-analysis.json"


def run():
    conn = duckdb.connect()

    failure_modes_file = str(DATA_DIR / "failure-mode-breakdown.json").replace("\\", "/")
    benchmarks_file = str(DATA_DIR / "benchmark-comparison.json").replace("\\", "/")
    review_queue_file = str(DATA_DIR / "review-queue.json").replace("\\", "/")
    feature_importance_file = str(DATA_DIR / "feature-importance.json").replace("\\", "/")

    # Q1: Failure modes ranked by holdout frequency
    q1 = conn.execute(f"""
        SELECT
            failure_code,
            label,
            holdout_failures,
            captured_in_top_10_percent_queue,
            ROUND(capture_rate * 100, 1) AS capture_rate_pct,
            RANK() OVER (ORDER BY holdout_failures DESC) AS frequency_rank
        FROM read_json_auto('{failure_modes_file}')
        ORDER BY holdout_failures DESC
    """).fetchall()
    q1_cols = [
        "failure_code",
        "label",
        "holdout_failures",
        "captured_top10",
        "capture_rate_pct",
        "frequency_rank",
    ]

    # Q2: Model comparison — focused on imbalance-aware metrics
    q2 = conn.execute(f"""
        SELECT
            model,
            ROUND(roc_auc, 4)  AS roc_auc,
            ROUND(pr_auc, 4)   AS pr_auc,
            ROUND(f1, 4)       AS f1,
            ROUND(precision, 4) AS precision_score,
            ROUND(recall, 4)   AS recall_score,
            RANK() OVER (ORDER BY pr_auc DESC) AS pr_auc_rank
        FROM read_json_auto('{benchmarks_file}')
        ORDER BY pr_auc DESC
    """).fetchall()
    q2_cols = ["model", "roc_auc", "pr_auc", "f1", "precision_score", "recall_score", "pr_auc_rank"]

    # Q3: Review queue ROI — cost per captured failure by budget level
    q3 = conn.execute(f"""
        SELECT
            ROUND(review_fraction * 100, 0) AS review_budget_pct,
            reviewed_assets,
            captured_failures,
            ROUND(failure_capture_rate * 100, 1) AS failure_capture_rate_pct,
            ROUND(yield_lift_vs_random, 2) AS yield_lift_vs_random,
            ROUND(reviewed_assets::DOUBLE / NULLIF(captured_failures, 0), 1) AS assets_per_failure_caught
        FROM read_json_auto('{review_queue_file}', format='auto')
        CROSS JOIN LATERAL (
            SELECT unnest(review_budgets) AS rb
        ) t
        CROSS JOIN LATERAL (
            SELECT
                rb.review_fraction AS review_fraction,
                rb.reviewed_assets AS reviewed_assets,
                rb.captured_failures AS captured_failures,
                rb.failure_capture_rate AS failure_capture_rate,
                rb.yield_lift_vs_random AS yield_lift_vs_random
        ) t2
        ORDER BY review_fraction
    """).fetchall()
    q3_cols = [
        "review_budget_pct",
        "reviewed_assets",
        "captured_failures",
        "failure_capture_rate_pct",
        "yield_lift_vs_random",
        "assets_per_failure_caught",
    ]

    # Q4: Feature importance — top 5 and their contribution share
    q4 = conn.execute(f"""
        WITH fi AS (
            SELECT
                feature,
                importance,
                CASE WHEN importance > 0 THEN importance ELSE 0 END AS positive_importance
            FROM read_json_auto('{feature_importance_file}', format='auto')
            CROSS JOIN LATERAL (SELECT unnest(features) AS feat) t
            CROSS JOIN LATERAL (SELECT feat.feature AS feature, feat.importance AS importance) t2
        ),
        total AS (SELECT SUM(positive_importance) AS total_pos FROM fi)
        SELECT
            feature,
            ROUND(importance, 4) AS importance,
            ROUND(positive_importance / NULLIF(total_pos, 0) * 100, 1) AS share_of_total_pct,
            RANK() OVER (ORDER BY importance DESC) AS importance_rank
        FROM fi, total
        ORDER BY importance DESC
        LIMIT 5
    """).fetchall()
    q4_cols = ["feature", "importance", "share_of_total_pct", "importance_rank"]

    def to_records(rows, cols):
        return [dict(zip(cols, row)) for row in rows]

    result = {
        "description": "SQL analysis on pre-computed JSON artifacts using DuckDB in-memory engine",
        "queries": [
            {
                "id": "Q1",
                "title": "Failure modes ranked by holdout frequency",
                "business_question": "Which failure types occur most often and are hardest to catch?",
                "results": to_records(q1, q1_cols),
            },
            {
                "id": "Q2",
                "title": "Model comparison ranked by PR-AUC (imbalance-aware metric)",
                "business_question": "Which model best identifies rare failures in an imbalanced dataset?",
                "results": to_records(q2, q2_cols),
            },
            {
                "id": "Q3",
                "title": "Review queue ROI by budget level",
                "business_question": "How many assets must be reviewed to catch each failure at different budget levels?",
                "results": to_records(q3, q3_cols),
            },
            {
                "id": "Q4",
                "title": "Top 5 features by permutation importance",
                "business_question": "Which sensor signals drive the failure risk score most?",
                "results": to_records(q4, q4_cols),
            },
        ],
    }

    OUT_FILE.write_text(json.dumps(result, indent=2))
    logger.info(f"  saved -> {OUT_FILE.relative_to(ROOT)}")

    # Print summary to stdout for CI visibility
    logger.info("\n--- Q1: Failure mode ranking ---")
    for r in result["queries"][0]["results"]:
        logger.info(
            f"  {r['failure_code']:4s}  {r['holdout_failures']:3d} holdout failures  "
            f"capture rate: {r['capture_rate_pct']}%"
        )

    logger.info("\n--- Q2: Model ranking by PR-AUC ---")
    for r in result["queries"][1]["results"]:
        logger.info(f"  Rank {r['pr_auc_rank']}: {r['model']:35s}  PR-AUC={r['pr_auc']}")

    logger.info("\n--- Q3: Review queue ROI ---")
    for r in result["queries"][2]["results"]:
        logger.info(
            f"  {r['review_budget_pct']:.0f}% budget  "
            f"capture={r['failure_capture_rate_pct']}%  "
            f"lift={r['yield_lift_vs_random']}x  "
            f"assets/failure={r['assets_per_failure_caught']}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Running SQL analysis with DuckDB...")
    run()
    logger.info("Done.")
