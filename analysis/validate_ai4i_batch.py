"""Deterministic, read-only AI4I telemetry batch quality checks.

This is a *data contract*, not a predictive model or a drift detector. It rejects
missing/invalid sensor values before a batch is sent for inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_FIELD = "Type"
NUMERIC_FIELDS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
REQUIRED_FIELDS = (CATEGORICAL_FIELD, *NUMERIC_FIELDS)
ALLOWED_TYPES = frozenset({"L", "M", "H"})


def validate_batch(frame: pd.DataFrame, sample_limit: int = 10) -> dict:
    """Return an auditable JSON-ready report without mutating the input.

    `row_number` is a 1-based CSV line number including the header (row 2 is
    the first data row), independent of a caller's DataFrame index.
    """
    if sample_limit < 0:
        raise ValueError("sample_limit must be nonnegative")

    row_count = len(frame)
    missing_columns = [field for field in REQUIRED_FIELDS if field not in frame.columns]
    invalid = np.zeros(row_count, dtype=bool)
    errors_by_field: dict[str, dict[str, int]] = {}
    examples: list[dict[str, str | int]] = []

    def record(field: str, reason: str, mask: np.ndarray) -> None:
        count = int(mask.sum())
        if not count:
            return
        errors_by_field.setdefault(field, {})[reason] = count
        invalid[:] |= mask
        for pos in np.flatnonzero(mask):
            if len(examples) >= sample_limit:
                break
            examples.append({"row_number": int(pos) + 2, "field": field, "reason": reason})

    if not row_count:
        errors_by_field["_batch"] = {"empty_batch": 1}

    for field in REQUIRED_FIELDS:
        if field not in frame.columns:
            continue
        values = frame[field]
        null_mask = values.isna().to_numpy(dtype=bool)
        record(field, "missing_value", null_mask)

        if field == CATEGORICAL_FIELD:
            categories = values.astype("string")
            invalid_type = (~values.isna() & ~categories.isin(ALLOWED_TYPES)).to_numpy(dtype=bool)
            record(field, "unknown_category", invalid_type)
            continue

        numbers = pd.to_numeric(values, errors="coerce")
        nonnumeric = (~values.isna() & numbers.isna()).to_numpy(dtype=bool)
        record(field, "not_numeric", nonnumeric)

        numeric_values = numbers.to_numpy(dtype=float, na_value=np.nan)
        nonfinite = (~np.isfinite(numeric_values)) & ~null_mask & ~nonnumeric
        record(field, "non_finite", nonfinite)

        eligible = np.isfinite(numeric_values)
        if field.endswith("[K]"):
            out_of_range = eligible & (numeric_values <= 0)
        else:
            out_of_range = eligible & (numeric_values < 0)
        record(field, "physically_invalid", out_of_range)

    if missing_columns:
        invalid[:] = True

    invalid_rows = int(invalid.sum())
    passed = not missing_columns and row_count > 0 and invalid_rows == 0
    return {
        "dataset_contract": "AI4I 2020 raw inference features",
        "status": "pass" if passed else "fail",
        "rows": row_count,
        "valid_rows": row_count - invalid_rows,
        "invalid_rows": invalid_rows,
        "required_columns": list(REQUIRED_FIELDS),
        "missing_columns": missing_columns,
        "errors_by_field": errors_by_field,
        "error_examples": examples,
        "error_examples_truncated": sum(sum(v.values()) for k, v in errors_by_field.items() if k != "_batch") > len(examples),
        "notes": [
            "Unknown additional columns are allowed; they are not validated.",
            "Sensor row duplicates are permitted because repeated readings can be legitimate.",
            "This validates schema/basic physical constraints, not model performance, population drift, or process-specific safe operating limits.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check AI4I CSV batches against the inference data contract.")
    parser.add_argument("csv", type=Path, help="CSV with AI4I raw feature columns")
    parser.add_argument("--report", type=Path, help="Optional path to save the JSON report")
    parser.add_argument("--sample-limit", type=int, default=10, help="Maximum error examples (default: 10)")
    args = parser.parse_args(argv)
    if args.sample_limit < 0:
        parser.error("--sample-limit cannot be negative")

    try:
        frame = pd.read_csv(args.csv, low_memory=False)
        report = validate_batch(frame, sample_limit=args.sample_limit)
        serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if report["status"] == "pass" else 2
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        print(f"Cannot inspect batch: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
