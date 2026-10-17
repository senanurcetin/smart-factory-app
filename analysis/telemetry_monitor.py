"""Advisory PSI monitoring for validated offline telemetry windows.

No MQTT subscriptions, PLC access, scheduled jobs, alerts to external
services, model retraining, or automated maintenance actions are performed.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from analysis.check_drift import build_drift_report
from analysis.run_ai4i_case_study import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    NUMERIC_FEATURES,
    add_derived_features,
)
from analysis.telemetry_ingestion import MODEL_FIELDS, TelemetrySample

MINIMUM_RECORDS = 30
MAX_AGE_SECONDS = 3600


def _result(status: str, errors: list[str], **fields: object) -> dict:
    return {"status": status, "errors": errors, **fields}


def _validate_reference(reference: pd.DataFrame, min_records: int) -> list[str]:
    if not isinstance(reference, pd.DataFrame):
        return ["reference_not_dataframe"]
    missing = [field for field in MODEL_FIELDS if field not in reference]
    if missing:
        return ["reference_missing_fields:" + ",".join(missing)]
    if len(reference) < min_records:
        return ["reference_insufficient_records"]
    for field in NUMERIC_FEATURES:
        numeric = pd.to_numeric(reference[field], errors="coerce")
        if numeric.isna().any() or not all(math.isfinite(float(v)) for v in numeric):
            return ["reference_invalid_numeric:" + field]
    for field in CATEGORICAL_FEATURES:
        if not reference[field].isin(("L", "M", "H")).all():
            return ["reference_invalid_category:" + field]
    return []


def evaluate_telemetry_window(
    reference: pd.DataFrame,
    samples: Sequence[TelemetrySample],
    *,
    evaluated_at: datetime,
    min_records: int = MINIMUM_RECORDS,
    max_age_seconds: int = MAX_AGE_SECONDS,
) -> dict:
    """Evaluate a finite, fresh, homogeneous batch against an offline baseline.

    Fail closed on missing, stale, mixed-source or low-volume windows; never
    label inadequate coverage as a healthy `no_significant_drift` state.
    """
    if min_records < 2 or max_age_seconds < 1:
        raise ValueError("min_records >= 2 and max_age_seconds >= 1 required")
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("evaluated_at must be timezone-aware")
    now = evaluated_at.astimezone(timezone.utc)

    errors = _validate_reference(reference, min_records)
    if errors:
        return _result("data_quality_error", errors, observed_records=len(samples))

    if len(samples) < min_records:
        return _result(
            "insufficient_data",
            ["window_insufficient_records"],
            observed_records=len(samples),
        )

    sources = set()
    seen = set()
    for sample in samples:
        if not isinstance(sample, TelemetrySample):
            return _result("data_quality_error", ["unvalidated_sample"])
        sources.add(sample.source)
        key = (sample.asset_id, sample.recorded_at)
        if key in seen:
            return _result("data_quality_error", ["duplicate_asset_timestamp"])
        seen.add(key)
        age_seconds = (now - sample.recorded_at).total_seconds()
        if age_seconds < 0:
            return _result("data_quality_error", ["future_timestamp"])
        if age_seconds > max_age_seconds:
            return _result("data_quality_error", ["stale_timestamp"])

    if len(sources) != 1:
        return _result("data_quality_error", ["mixed_simulated_and_external_sources"])

    # Keep reference and telemetry separate. The model is not used to infer
    # missing sensor measurements or invent ground-truth maintenance outcomes.
    current = pd.DataFrame([sample.model_features() for sample in samples])
    try:
        baseline = add_derived_features(reference.loc[:, list(MODEL_FIELDS)].copy())
        observed = add_derived_features(current)
        # The base PSI helper returns zero on a constant reference column.
        # That can be misread as healthy if the current sensor changes: block it.
        for field in NUMERIC_FEATURES + DERIVED_FEATURES:
            unique_reference = baseline[field].dropna().unique()
            if len(unique_reference) <= 1 and not observed[field].eq(
                unique_reference[0] if len(unique_reference) else float("nan")
            ).all():
                return _result(
                    "data_quality_error",
                    ["unusable_constant_reference:" + field],
                    observed_records=len(samples),
                )
        drift = build_drift_report(baseline, observed)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return _result(
            "data_quality_error",
            ["drift_computation_failed:" + type(exc).__name__],
            observed_records=len(samples),
        )

    flagged = [
        {"feature": row["feature"], "psi": row["psi"], "verdict": row["verdict"]}
        for row in drift["features"]
        if row["verdict"] != "no_significant_drift"
    ]
    return _result(
        "drift_alert" if flagged else "ok",
        [],
        source=next(iter(sources)),
        reference_records=len(reference),
        observed_records=len(samples),
        evaluated_at=now.isoformat(),
        alert_features=flagged,
        report=drift,
        action="advisory_only_no_plant_control",
    )
