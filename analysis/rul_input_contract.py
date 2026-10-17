"""Offline NASA C-MAPSS FD001 feature contract. No model loading or plant access.

The AI4I dashboard's heuristic RUL has a DIFFERENT input domain and must
never be replaced with this separate engine-degradation regression output.
"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Mapping, Sequence

WINDOW = 5
SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
SENSOR_COLUMNS = tuple(f"sensor_{i}" for i in range(1, 22))
DROPPED_FD001_SENSORS = frozenset(
    ("sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19")
)
INFORMATIVE_SENSORS = tuple(
    column for column in SENSOR_COLUMNS if column not in DROPPED_FD001_SENSORS
)
RAW_COLUMNS = ("unit_number", "time_cycles", *SETTING_COLUMNS, *SENSOR_COLUMNS)
MODEL_COLUMNS = (
    *SETTING_COLUMNS,
    *INFORMATIVE_SENSORS,
    *(f"{col}_roll_mean" for col in INFORMATIVE_SENSORS),
    *(f"{col}_roll_std" for col in INFORMATIVE_SENSORS),
)


class RulInputError(ValueError):
    """Missing, incompatible, stale or ambiguous engine history."""


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RulInputError(f"{name} must be a positive integer")
    return value


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RulInputError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RulInputError(f"{name} must be finite")
    return numeric


def build_fd001_feature_row(
    history: Sequence[Mapping[str, object]],
) -> dict:
    """Build one model-compatible feature row from 5+ contiguous engine cycles.

    All 21 raw sensor channels and all three settings are required for a
    provenance-checked FD001 snapshot. Sensor selection is fixed to the
    15 fields reported in the committed FD001 benchmark. No prediction
    is performed and the return value never claims live asset validity.
    """
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise RulInputError("history must be a sequence of engine-cycle records")
    if len(history) < WINDOW:
        raise RulInputError("at least five contiguous cycles are required")

    rows = []
    expected = frozenset(RAW_COLUMNS)
    unit_number = None
    previous_cycle = None
    for raw in history:
        if not isinstance(raw, Mapping) or frozenset(raw) != expected:
            raise RulInputError("record must contain exactly 26 FD001 raw fields")
        unit = _positive_integer("unit_number", raw["unit_number"])
        cycle = _positive_integer("time_cycles", raw["time_cycles"])
        if unit_number is None:
            unit_number = unit
        if unit != unit_number:
            raise RulInputError("mixed engine identities are not allowed")
        if previous_cycle is not None and cycle != previous_cycle + 1:
            raise RulInputError("cycles must be strictly increasing without gaps")
        previous_cycle = cycle
        checked = {
            column: _finite_number(column, raw[column])
            for column in (*SETTING_COLUMNS, *SENSOR_COLUMNS)
        }
        rows.append(checked)

    recent = rows[-WINDOW:]
    latest = recent[-1]
    features = {col: latest[col] for col in (*SETTING_COLUMNS, *INFORMATIVE_SENSORS)}
    for sensor in INFORMATIVE_SENSORS:
        values = [row[sensor] for row in recent]
        features[f"{sensor}_roll_mean"] = mean(values)
        features[f"{sensor}_roll_std"] = stdev(values)
    if tuple(features) != MODEL_COLUMNS:
        raise RulInputError("FD001 feature schema/order mismatch")

    return {
        "unit_number": unit_number,
        "last_cycle": previous_cycle,
        "window": WINDOW,
        "model_features": features,
        "source": "NASA C-MAPSS FD001 simulated benchmark",
        "is_live_plant_data": False,
        "automated_maintenance_action": False,
    }
