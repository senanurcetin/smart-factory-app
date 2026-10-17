# Smart Factory — offline telemetry drift and data-quality monitor

**Status:** Offline advisory analytics only. No broker subscriptions,
OPC UA connections, scheduled jobs, notification delivery, equipment actions,
or claims about real factory readings.

This stage connects the strict validated input contract in
[`TELEMETRY_INPUT_CONTRACT.md`](TELEMETRY_INPUT_CONTRACT.md) to the
pre-existing AI4I Population Stability Index (PSI) calculation.

## API

```python
from datetime import datetime, timezone
from analysis.telemetry_monitor import evaluate_telemetry_window

# offline_reference is a validated reference dataframe of the six AI4I
# raw feature columns; samples are prevalidated TelemetrySample objects.
result = evaluate_telemetry_window(
    offline_reference,
    samples,
    evaluated_at=datetime.now(timezone.utc),
    min_records=30,
    max_age_seconds=3600,
)
```

Response `status` values:

| Status | Meaning | Downstream action |
| --- | --- | --- |
| `ok` | Valid, sufficiently sized fresh window; no selected PSI alerts | Informational only |
| `drift_alert` | One or more PSI values crossed moderate/significant thresholds | Advisory review |
| `insufficient_data` | Too few observations | **Never** show as healthy |
| `data_quality_error` | Bad reference, mixed provenance, stale/future data, replay or unusable constant reference | **Never** show as healthy |

All successful results include `source`, `alert_features`, complete
offline `report`, and `action="advisory_only_no_plant_control"`.
PSI thresholds are inherited from `analysis/check_drift.py`: moderate
`>=0.1`, significant `>=0.2`. They are statistical diagnostics, **not**
operational safety limits.

The evaluator refuses a changed value against a fully constant reference
feature instead of accepting the PSI helper's degenerate zero result.
It also fails closed on insufficient or nonfinite reference data, duplicate
asset/time pairs, mixed simulated and external sources, timestamp staleness
and future timestamps. No missing values are fabricated.

The original generic PSI helper has fixed descriptive metadata for its
historical +80-minute synthetic benchmark. This adapter explicitly replaces
that `scenario` description with a truthful source-tagged observation-window
description while retaining the calculated PSI values. A decoded MQTT/OPC UA
record is **not evidence that a PLC was accessed or that the factory is live**.

## Offline verification

```bash
python -m unittest discover -s tests -p test_telemetry_monitor.py -v
```

Test fixtures are synthetic and deterministic, no upstream dataset downloads.
Before an actual installation, define plant-approved tags, units, authenticated
asset IDs, freshness thresholds, reference collection procedures, durable
deduplication, alert acknowledgement and suppression, and escalation owners.
Do not enable real plant data collection without specific authorization.
