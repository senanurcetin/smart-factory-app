# Smart Factory — Telemetry ingestion contract (offline stage)

**Scope:** A read-only input-validation boundary for a future OPC UA / MQTT
pipeline. This change adds *no network connections, broker credentials, PLC
sessions, actuator commands, or user-facing claim of real telemetry*.
The public dashboard continues generating **simulated AI4I-schema data**.

## Canonical record

```json
{
  "asset_id": "CNC-001",
  "recorded_at": "2026-10-17T12:00:00+03:00",
  "Type": "L",
  "Air temperature [K]": 299.4,
  "Process temperature [K]": 309.2,
  "Rotational speed [rpm]": 1550,
  "Torque [Nm]": 42.0,
  "Tool wear [min]": 100
}
```

- `recorded_at` **must** include UTC offset; normalized to UTC.
- `asset_id` must be an ASCII identifier (letters, digits, `_`, `-`, `.`, `:`).
- Exactly six AI4I raw features are accepted. Missing or unexpected fields
  (including untrusted `source` and `Target`) fail closed.
- Numeric fields must be finite numbers within broad *input plausibility*
  ranges; those bounds are **not** machine operating or SIL safety limits.
- Each source is tagged by the authorized adapter, not by message-supplied
  source strings. Invalid/non-UTF8/oversized/duplicate-key JSON fails closed.
- In-memory `TelemetryReplayWindow` rejects duplicate or older samples **per
  asset**. It is local-process only; production deduplication will require a
  persistent state store keyed by authenticated asset identity.

## Read-only decoder entry points

- `analysis.telemetry_ingestion.parse_record(payload, source="simulated")`
  validates a fixture independently from any transport.
- `parse_mqtt_message("smartfactory/telemetry/CNC-001", payload_bytes)`
  enforces agreement between the topic asset segment and the payload's asset
  field. **It does not subscribe** to MQTT.
- `parse_opcua_snapshot(asset_id, recorded_at, readings)` normalizes already
  fetched node readings. **It does not open** an OPC UA session or issue writes.
- `sample.model_features()` yields exactly six features for later
  `add_derived_features` / model scoring after an explicit integration gate.

## Validate without network or plant access

```bash
python -m unittest discover -s tests -p test_telemetry_ingestion.py -v
ruff check analysis/telemetry_ingestion.py tests/test_telemetry_ingestion.py
black --check analysis/telemetry_ingestion.py tests/test_telemetry_ingestion.py
```

The test suite covers timestamp offsets, expected features, finite sensor
bounds, MQTT identity mismatch, OPC UA metadata override, duplicate JSON
keys, oversized payload, source spoofing, replay/out-of-order observations,
and bounded asset-state capacity.

## Later live-adapter gate (not implemented)

1. Configure a read-only OPC UA endpoint with authenticated client,
   certificates/trust list, specific node-ID → canonical-unit mapping, and
   tested reconnect/backoff. Never expose the PLC directly to the public demo.
2. Configure a broker connection with verified TLS identity, least-privilege
   topic ACL, QoS/deduplication policy, and bounded input queues. MQTT
   wildcard topic subscriptions require their own scoped approval.
3. Separate real and simulated input pipelines. Track data provenance,
   timestamp freshness, missingness, machine/unit calibration and connection
   health; fail closed on out-of-distribution hardware signals.
4. Only after offline replay acceptance, add persistence, live risk scoring,
   scheduled PSI evaluation, alerts, audit logs and performance monitoring.
   Explicit approval is required before using live factory access.
5. No automatic maintenance instruction or PLC write is permitted by this
   input boundary; model estimates are advisory only.
