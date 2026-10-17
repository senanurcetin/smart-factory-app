# NASA FD001 — offline RUL feature contract

**Implementation:** `analysis/rul_input_contract.py`. This is an offline
schema check and feature generator, not a prediction endpoint or live plant
data connector.

## Why RUL must stay separate from AI4I

The dashboard's AI4I failure-risk model receives six raw machine-tool
features and shows a labelled heuristic RUL proxy. NASA C-MAPSS FD001 is
a **different**, simulated turbofan-engine degradation domain. Its trained
regressor uses one engine's sequential history and a separate 48-feature
matrix. Passing AI4I observations into the FD001 estimator, or publishing
FD001 estimates as live plant measurements, would be invalid.

## Raw input schema and acceptance gate

An input history must contain 5+ consecutive cycles of **one engine**.
Each record must contain exactly:

- `unit_number` — positive integer, unchanged throughout the history.
- `time_cycles` — positive integer, strictly increasing by 1.
- `setting_1`, `setting_2`, `setting_3`.
- `sensor_1` through `sensor_21`, all finite numbers.

The contract rejects partial sensor inputs, mixed engine histories,
duplicated/missing cycles, nonfinite observations and extra fields.
It does not infer timestamps, machine identity mapping or failure labels.
The full model-training data is not required by the offline contract.

## Feature engineering, aligned with the training script

The known FD001 feature-selection result excludes sensors 1, 5, 10,
16, 18 and 19 as near-constant for this benchmark, leaving 15 sensors.
The feature row is ordered as:

1. Three operating settings.
2. Fifteen selected current-cycle sensor values.
3. Fifteen five-cycle rolling means.
4. Fifteen five-cycle sample standard deviations (ddof=1).

Total: **48 model features** in the same ordering used by
`analysis/run_cmapss_rul_case_study.py`. The generator returns a labelled
feature record with `is_live_plant_data=False` and
`automated_maintenance_action=False`. It never loads a model or produces
a claimed RUL prediction.

## Offline test

```bash
python -m unittest discover -s tests -p test_rul_input_contract.py -v
```

The nine no-network tests check correct feature order/window statistics,
valid field counts, engine grouping, duplicate or skipped cycles, nonfinite
readings, false identities and absence of invented predictions.

**Before actual inference integration:** independently verify the feature
order against the serialized FD001 pipeline's `feature_names_in_`, test
prediction shape and units on the official FD001 held-out set, and add a
versioned model card. Do not connect to an operational facility without a
separate approved mapping and safety review.
