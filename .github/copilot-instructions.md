# GitHub Copilot instructions — Smart Factory App

## Project context
This Python/Flask industrial-analytics case study uses UCI AI4I 2020 predictive maintenance and NASA C-MAPSS remaining-useful-life (RUL) evaluations. Inspect `README.md`, `case_study.py`, `rul_case_study.py`, `analysis/check_drift.py`, affected `tests/` and `pyproject.toml` before editing.

## Scientific integrity
- Distinguish model failure-risk scores and RUL estimates from plant guarantees. Never call an offline benchmark a verified live predictive-maintenance deployment.
- Validate PR-AUC/recall, cost-sensitive review queues, probability calibration and class-imbalance effects; do not optimize for accuracy on an imbalanced dataset.
- Separate train/validation/test data and preserve committed model artifact provenance. No retraining, UCI/NASA downloads, live telemetrics or metric updates unless explicitly requested and reproducible.
- Handle malformed/nonfinite sensor values, unknown categories and missing columns without producing misleading green/secure risk predictions.
- Keep customer, plant and secret data out of tests; mock any external dependencies.

## Offline workflow
- Tests are Python `unittest`: e.g. `python -m unittest tests.test_drift_check -v`, `python -m unittest tests.test_dashboard_model -v` or `python -m unittest discover -s tests -v` if practical.
- Static checks: `ruff check .` and `black --check .` when installed. Avoid `analysis/run_*case_study.py` retraining and `.github/workflows/retrain-manual.yml` without permission.
- CI includes browser, Docker and artifact verification. Do not trigger manual workflow runs simply to test agent instructions.
- Report only verified tests, exact commands, and limitations. Keep changes small, preserve dashboard files and commit artifacts except when part of an approved task.

## Copilot specialist
`.github/agents/maintenance-model-qa.agent.md` is opt-in for offline data/model/regression tasks. No automatic merges.
