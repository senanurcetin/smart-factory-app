"""Offline, read-only telemetry boundary for future MQTT / OPC UA integrations.

This module does not connect to a broker, PLC, or plant network. Transport
adapters must authenticate upstream; this layer validates and normalizes data.
It deliberately never executes equipment commands or labels samples as live.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

MODEL_FIELDS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
REQUIRED_FIELDS = frozenset(("asset_id", "recorded_at", *MODEL_FIELDS))
ASSET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MQTT_TOPIC_PREFIX = "smartfactory/telemetry/"
MAX_PAYLOAD_BYTES = 8192

# Broad input-sanity limits, NOT PLC operating or safety thresholds.
SENSOR_BOUNDS = {
    "Air temperature [K]": (1.0, 1000.0),
    "Process temperature [K]": (1.0, 1000.0),
    "Rotational speed [rpm]": (0.0, 50000.0),
    "Torque [Nm]": (0.0, 100000.0),
    "Tool wear [min]": (0.0, 10000000.0),
}


class TelemetryValidationError(ValueError):
    """A transport message does not satisfy the offline telemetry contract."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise TelemetryValidationError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _decode_payload(payload: str | bytes | Mapping[str, object]) -> dict:
    if isinstance(payload, bytes):
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise TelemetryValidationError("Payload too large")
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TelemetryValidationError("Payload is not UTF-8") from exc
    if isinstance(payload, str):
        if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise TelemetryValidationError("Payload too large")
        try:
            payload = json.loads(payload, object_pairs_hook=_unique_json_object)
        except (ValueError, TypeError) as exc:
            raise TelemetryValidationError("Invalid JSON payload") from exc
    if not isinstance(payload, Mapping):
        raise TelemetryValidationError("Payload must be a JSON object")
    return dict(payload)


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TelemetryValidationError("recorded_at must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TelemetryValidationError("Invalid recorded_at timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TelemetryValidationError("recorded_at requires timezone")
    return parsed.astimezone(timezone.utc)


def _bounded_number(key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryValidationError(f"{key}: expected a numeric value")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TelemetryValidationError(f"{key}: invalid numeric value") from exc
    lower, upper = SENSOR_BOUNDS[key]
    if not math.isfinite(numeric) or not lower <= numeric <= upper:
        raise TelemetryValidationError(f"{key}: outside finite input bounds")
    return numeric


@dataclass(frozen=True)
class TelemetrySample:
    """Validated source-tagged record; never a calibrated plant observation."""

    asset_id: str
    recorded_at: datetime
    source: str
    machine_type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_minutes: float

    def model_features(self) -> dict[str, str | float]:
        """Return exactly the six AI4I raw features, with no metadata leakage."""
        return {
            "Type": self.machine_type,
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_minutes,
        }


def parse_record(
    payload: str | bytes | Mapping[str, object], *, source: str = "simulated"
) -> TelemetrySample:
    """Normalize a single record. Source is caller-supplied, not payload-supplied."""
    if source not in {"simulated", "mqtt", "opcua"}:
        raise TelemetryValidationError("Unknown transport source")
    record = _decode_payload(payload)
    extra = set(record) - REQUIRED_FIELDS
    missing = REQUIRED_FIELDS - set(record)
    if missing or extra:
        raise TelemetryValidationError(
            f"Unexpected telemetry fields; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    asset_id = record["asset_id"]
    if not isinstance(asset_id, str) or not ASSET_ID_PATTERN.fullmatch(asset_id):
        raise TelemetryValidationError("Invalid asset_id")
    machine_type = record["Type"]
    if not isinstance(machine_type, str) or machine_type not in {"L", "M", "H"}:
        raise TelemetryValidationError("Type must be L, M or H")
    values = {field: _bounded_number(field, record[field]) for field in SENSOR_BOUNDS}
    return TelemetrySample(
        asset_id=asset_id,
        recorded_at=_timestamp(record["recorded_at"]),
        source=source,
        machine_type=machine_type,
        air_temperature_k=values["Air temperature [K]"],
        process_temperature_k=values["Process temperature [K]"],
        rotational_speed_rpm=values["Rotational speed [rpm]"],
        torque_nm=values["Torque [Nm]"],
        tool_wear_minutes=values["Tool wear [min]"],
    )


def parse_mqtt_message(topic: str, payload: str | bytes) -> TelemetrySample:
    """Pure MQTT decoder; connection, TLS, ACL and authentication stay upstream."""
    if not isinstance(topic, str) or not topic.startswith(MQTT_TOPIC_PREFIX):
        raise TelemetryValidationError("Invalid MQTT topic")
    topic_asset = topic[len(MQTT_TOPIC_PREFIX) :]
    if not ASSET_ID_PATTERN.fullmatch(topic_asset):
        raise TelemetryValidationError("Invalid MQTT asset segment")
    sample = parse_record(payload, source="mqtt")
    if sample.asset_id != topic_asset:
        raise TelemetryValidationError("MQTT topic and payload asset_id mismatch")
    return sample


def parse_opcua_snapshot(
    asset_id: str, recorded_at: str, readings: Mapping[str, object]
) -> TelemetrySample:
    """Normalize an already-read snapshot; no OPC UA session or node writes."""
    if not isinstance(readings, Mapping):
        raise TelemetryValidationError("OPC UA readings must be a mapping")
    if "asset_id" in readings or "recorded_at" in readings:
        raise TelemetryValidationError("OPC UA readings cannot override source metadata")
    return parse_record(
        {"asset_id": asset_id, "recorded_at": recorded_at, **readings},
        source="opcua",
    )


class TelemetryReplayWindow:
    """Bounded offline per-asset deduplication, with fail-closed timestamp order."""

    def __init__(self, max_assets: int = 1000):
        if not isinstance(max_assets, int) or isinstance(max_assets, bool) or max_assets < 1:
            raise ValueError("max_assets must be positive")
        self._last: dict[str, datetime] = {}
        self.max_assets = max_assets

    def accept(self, sample: TelemetrySample) -> bool:
        if not isinstance(sample, TelemetrySample):
            raise TypeError("Expected TelemetrySample")
        previous = self._last.get(sample.asset_id)
        if previous is not None and sample.recorded_at <= previous:
            return False
        if previous is None and len(self._last) >= self.max_assets:
            raise TelemetryValidationError("Asset window capacity reached")
        self._last[sample.asset_id] = sample.recorded_at
        return True
