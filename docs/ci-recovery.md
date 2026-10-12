# CI reliability: offline verification and manual benchmark regeneration

## Why the pipeline was unstable

Previously the normal `CI` workflow downloaded datasets from UCI and NASA on
**every** push and pull request, retrained the two ML pipelines, and regenerated
the drift report. A TLS certificate/remote service outage could therefore
fail an otherwise valid application change. It also consumed substantial CI
runtime, even for documentation or UI-only changes.

## Normal pull-request CI (no external benchmark download)

`.github/workflows/ci.yml` now uses checked-in binary models and metrics
artifacts, validates their existence and JSON readability, renders the
visuals, and runs the existing application/model/contract tests. CI does
**not** retrain either benchmark or fetch UCI/NASA datasets.

This makes source-code PR verification repeatable and does **not** require
disabling TLS certificate validation. Dependency installation and browser
setup still require their normal package registries.

**Note:** these checks validate the shipped artifacts; they do not prove that
retraining is currently reproducible from the original external data.

## Regenerating models intentionally

1. Open **Actions > Retrain ML artifacts (manual) > Run workflow**.
2. Choose `ai4i`, `cmapss`, or `all` and run only when public source data
   or ML pipeline code changes warrant revalidation.
3. Review logs, unit tests, and the `smart-factory-retrained-*` artifact.
4. Review artifact changes and model metrics before deciding whether to
   commit replacements, using a normal PR.

Manual retraining can fail if UCI/NASA sources or their TLS certificate
chains are broken. The correct remedy is to restore verified connectivity
or use an authorized dataset copy in `analysis/.cache/` with provenance
preserved; do **not** use `verify=False` or disable certificate checks.

## Drift tests

The default `tests/test_drift_check.py` suite uses a deterministic
AI4I-schema synthetic fixture and performs no network calls. The same
integration assertions are also run against actual UCI data only when
`RUN_REAL_DATA_TESTS=1` is set. The manual AI4I workflow enables this flag.

For an offline quality check after dependencies have been installed:

```bash
python -m unittest discover -s tests -p "test_drift_check.py" -v
python -m unittest discover -s tests -p "test_artifacts.py" -v
python -m unittest discover -s tests -p "test_rul_artifacts.py" -v
```

## Actions usage

Routine PR checks no longer pay the compute and download cost for two
retraining pipelines. The `retrain-manual.yml` workflow has **no**
`push`, `pull_request`, or `schedule` trigger. This repository still
requires standard CI compute for unit tests, rendering, browser smoke tests
and Docker smoke tests. A GitHub-wide quota/billing issue cannot be fixed
by workflow changes in a single repo.
