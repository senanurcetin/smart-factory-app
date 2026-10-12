# AI4I telemetry batch quality gate

This repository contains a **read-only, offline data-contract validator** for batches intended for the AI4I predictive-maintenance model. It checks the six raw inputs used by the trained model before inference or drift analysis.

## Quick start

From the repository root, install the existing project dependencies and run:

```bash
python -m analysis.validate_ai4i_batch path/to/readings.csv
python -m analysis.validate_ai4i_batch path/to/readings.csv --report reports/batch-quality.json
python -m unittest tests.test_batch_quality -v
```

The command prints structured JSON to standard output. `--report` saves the same report to a file; it never changes the source CSV, the training artifacts or the deployed model.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Input batch passed the basic contract |
| `1` | File could not be opened or parsed |
| `2` | Data contract failed (schema, values or empty batch) |

## Required CSV fields

| Column | Contract |
| --- | --- |
| `Type` | One of `L`, `M`, `H` |
| `Air temperature [K]` | Finite number greater than 0 |
| `Process temperature [K]` | Finite number greater than 0 |
| `Rotational speed [rpm]` | Finite, nonnegative number |
| `Torque [Nm]` | Finite, nonnegative number |
| `Tool wear [min]` | Finite, nonnegative number |

Column names must match exactly. Additional CSV columns (including identifiers and targets) are allowed. Duplicate telemetry rows are **not** a failure, since identical readings may be genuine independent events. The contract intentionally does not impose narrow training-set ranges; that requires a separate, justified out-of-distribution monitor.

Example header:

```csv
Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
L,298.1,308.6,1500,40.5,20
H,300.4,310.2,1700,55.0,125
```

## Reading the report

- `status` is `pass` or `fail`.
- `rows`, `valid_rows` and `invalid_rows` count *unique data rows*, not failed cells.
- `missing_columns` lists required fields that were absent; this invalidates the entire batch.
- `errors_by_field` counts failed rules within each feature.
- `error_examples` identifies problematic input rows with **1-based CSV line numbers including the header**, plus field and issue type. Raw offending values are omitted intentionally.
- `error_examples_truncated` indicates additional errors beyond the sample limit.

To limit report size and exposure, the validator never includes raw row values in its error examples. `--sample-limit 0` suppresses examples entirely.

## Scope and operating guidance

This is an engineering demonstration of **schema/data-quality checks**, not evidence that incoming signals are safe for a plant controller. It does not verify sensor calibration, data provenance, timestamp sequencing, operating envelopes, PSI drift, cybersecurity, model accuracy or false-alarm cost. Do not feed production equipment automatically from this demonstration.

For population drift in the simulated AI4I scenario, see [`analysis/check_drift.py`](../analysis/check_drift.py). For model selection and evaluation context, see [the case study](case-study.md).

Questions and security concerns: **senanur.cetin.work@gmail.com**.
