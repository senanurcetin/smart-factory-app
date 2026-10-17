"""One-shot offline Smart Factory AI4I/FD001 → verified BigQuery NDJSON.

Never downloads data, runs cloud queries, writes BigQuery or authenticates.
Official source archive trust and Kaggle license/version parity are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.ai4i_batch import Ai4iBatchError, prepare_ai4i_rows, write_new_jsonl
from analysis.bigquery_load_guard import LoadPreflightError, validate_batch_upload
from analysis.bigquery_plan import BigQueryPlanError, build_sql_plan
from analysis.fd001_batch import (
    Fd001BatchError,
    prepare_fd001_batches,
    write_fd001_jsonl,
)


class StagingError(ValueError):
    """Input data or local staging verification failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_ai4i(
    path: Path,
    sha: str,
    output_dir: Path,
    *,
    run_id: str,
    evaluated_at: datetime,
    project: str,
    prefix: str = "smart_factory",
    location: str = "EU",
    expected_rows: int = 10000,
) -> dict:
    """Stage one already-verified local AI4I file and preflight it."""
    build_sql_plan(project, prefix, location)
    rows = prepare_ai4i_rows(
        path,
        expected_csv_sha256=sha,
        run_id=run_id,
        ingested_at=evaluated_at,
        expected_rows=expected_rows,
    )
    destination = Path(output_dir) / "ai4i_records.jsonl"
    receipt = write_new_jsonl(rows, destination)
    try:
        check = validate_batch_upload(
            destination,
            expected_file_sha256=receipt["export_sha256"],
            expected_source_sha256=sha,
            expected_rows=expected_rows,
            project=project,
            prefix=prefix,
            location=location,
            table="ai4i_records",
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "dataset": "uci_ai4i_2020",
        "status": "local_export_preflight_pass",
        "receipt": receipt,
        "preflight": check,
    }


def stage_fd001(
    paths: dict[str, Path],
    digests: dict[str, str],
    output_dir: Path,
    *,
    run_id: str,
    evaluated_at: datetime,
    project: str,
    prefix: str = "smart_factory",
    location: str = "EU",
    expected_train_rows: int = 20631,
    expected_train_units: int = 100,
    expected_test_units: int = 100,
) -> dict:
    """Stage official FD001 cycle/truth files with two independent QA passes."""
    build_sql_plan(project, prefix, location)
    if not isinstance(paths, dict) or set(paths) != {"train", "test", "truth"}:
        raise StagingError("All three FD001 paths are required")
    prepared = prepare_fd001_batches(
        paths["train"],
        paths["test"],
        paths["truth"],
        expected_sha256=digests,
        run_id=run_id,
        ingested_at=evaluated_at,
        expected_train_rows=expected_train_rows,
        expected_train_units=expected_train_units,
        expected_test_units=expected_test_units,
    )
    directory = Path(output_dir)
    published = write_fd001_jsonl(prepared, directory)
    try:
        checks = []
        for path_name, table, expected_rows in (
            ("fd001_cycles.jsonl", "fd001_cycles", len(prepared["cycles"])),
            (
                "fd001_official_test_truth.jsonl",
                "fd001_official_test_truth",
                len(prepared["official_test_truth"]),
            ),
        ):
            destination = directory / path_name
            checks.append(
                validate_batch_upload(
                    destination,
                    expected_file_sha256=_sha256(destination),
                    expected_source_sha256=prepared["evidence"]["bundle_source_sha256"],
                    expected_rows=expected_rows,
                    project=project,
                    prefix=prefix,
                    location=location,
                    table=table,
                )
            )
    except BaseException:
        for filename in published["export_paths"]:
            (directory / filename).unlink(missing_ok=True)
        raise
    return {
        "dataset": "nasa_cmapss_fd001",
        "status": "local_export_preflight_pass",
        "receipt": published,
        "preflights": checks,
        "source_evidence": prepared["evidence"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage offline UCI/NASA BigQuery JSONL")
    parser.add_argument("--project", required=True)
    parser.add_argument("--prefix", default="smart_factory")
    parser.add_argument("--location", default="EU")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="dataset", required=True)
    ai4i = sub.add_parser("ai4i")
    ai4i.add_argument("--csv", type=Path, required=True)
    ai4i.add_argument("--csv-sha256", required=True)
    fd001 = sub.add_parser("fd001")
    fd001.add_argument("--train", type=Path, required=True)
    fd001.add_argument("--test", type=Path, required=True)
    fd001.add_argument("--truth", type=Path, required=True)
    fd001.add_argument("--train-sha256", required=True)
    fd001.add_argument("--test-sha256", required=True)
    fd001.add_argument("--truth-sha256", required=True)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    try:
        if args.dataset == "ai4i":
            result = stage_ai4i(
                args.csv,
                args.csv_sha256,
                args.output_dir,
                run_id=args.run_id,
                evaluated_at=now,
                project=args.project,
                prefix=args.prefix,
                location=args.location,
            )
        else:
            result = stage_fd001(
                {"train": args.train, "test": args.test, "truth": args.truth},
                {
                    "train": args.train_sha256,
                    "test": args.test_sha256,
                    "truth": args.truth_sha256,
                },
                args.output_dir,
                run_id=args.run_id,
                evaluated_at=now,
                project=args.project,
                prefix=args.prefix,
                location=args.location,
            )
    except (
        Ai4iBatchError,
        BigQueryPlanError,
        Fd001BatchError,
        LoadPreflightError,
        StagingError,
    ) as exc:
        parser.exit(2, "LOCAL STAGING REJECTED: " + str(exc) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
