---
description: Read-only source-backed acceptance for Maintenance Model QA agent
---

# Maintenance Model QA — custom agent acceptance

Read-only source acceptance. Invocation must be explicit: GitHub Copilot CLI `--agent=maintenance-model-qa` or a supported IDE agent selector. The external test harness, not the model response, must record the requested agent ID and CLI exit code. Do not mark NOT RUN just because no IDE selector was used. If the actual loaded profile is not independently shown by runtime evidence, mark profile identity UNVERIFIED separately from source-review results.

Read these checked-in files (if missing, record that fact):
- `README.md`
- `analysis/check_drift.py`
- `tests/test_drift_check.py`
- `tests/test_dashboard_model.py`
- `tests/test_rul_case_study.py`
- `docs/data/ai4i-case-study/summary.json`

Using ONLY files you actually inspect, answer:
1. Trace the PSI numeric/categorical drift calculation and finite-input handling; distinguish synthetic batch drift checks from observed plant drift.
2. Identify how missing/nonfinite sensor values, feature schema errors, and probability/risk ranges are handled. Cite concrete tests; propose one missing synthetic regression without claiming it ran.
3. Explain why imbalanced AI4I maintenance classification needs PR-AUC/recall/false-negative costs rather than accuracy alone, and why published benchmark artifacts cannot be retroactively inferred to be live performance.
4. Describe RUL chronological/group leakage risks and what is not established by checked-in artifacts or unrun unit tests.

For each numbered check, give PASS, FAIL, or UNVERIFIED and cite concrete path names plus the relevant function, condition or test. A passing source review is NOT proof that code was executed. Label all unrun tests, live-device claims, runtime deployments and CI UNVERIFIED. Do not edit files, execute shell commands, run tests, contact external APIs, install dependencies, touch credentials, or trigger CI/GitHub Actions/CircleCI/Vercel. Do not call benchmark results equipment safety guarantees or retrain models.
