# Smart Factory — release acceptance evidence (2026-10-14)

PR: #22, branch `design/nexus-analysis-workspaces-20261013`.

## Local regression
- Offline pinned `scikit-learn==1.9.0`: 139 passed, 4 opt-in AI4I tests skipped, 248 subtests passed.
- Both distinct analytical workspaces verified with Playwright at 1440, 1024 and 375 px; 6/6 chart-surface and responsive subcases passed. No horizontal document overflow or missing images.
- Full-width tablet diagnostic charts, dark source canvas, actual AI4I risk-budget scenario rows and C-MAPSS FD001 model metrics have been preserved.
- Regression suite now guards the 1024 px single-column plots and true offline benchmark framing.

## Real AI4I opt-in integration (not a synthetic fixture)
- Official UCI dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset (DOI: 10.24432/C5HS5C).
- UCI legacy ZIP download currently fails local Windows HTTPS verification because the upstream certificate has expired. Never disable TLS verification.
- Two independently hosted GitHub CSV copies were downloaded with valid HTTPS and audited against the UCI schema: `tayyabrehman96/XAI-PdMNet-Bench-Industry-5.0/data/ai4i/ai4i2020.csv` and `mandyiv/Predictive_Machine_Maintenance/ai4i2020.csv`.
- Both copies have the **same byte-for-byte SHA-256** `59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183`; exactly 10,000 records / 14 expected columns / 339 machine failures / no nulls / unique 1..10,000 UDI.
- This corroborates consistency across mirrors; it is not a SHA signed by UCI. The verified CSV and local ZIP were written **only to ignored** `analysis/.cache/`, never committed and never used to retrain/deploy the model.
- `RUN_REAL_DATA_TESTS=1 python -m pytest -q tests/test_drift_check.py`: **29 passed, 15 subtests passed**, including the four previously skipped real-dataset cases.

## Required CircleCI release acceptance
- Trigger exactly one clean CI workflow via a regular commit without `[skip ci]`.
- CircleCI `python-quality` must pass locked dependencies, Ruff, Black, compilation, committed artifact verification, offline unit/route tests and Flask import.
- CircleCI `browser-quality` must pass real Chromium tests and publish current screenshots; inspect `artifacts/gallery/`.
- No paid GitHub Actions workflow is required.
- Merge only when the exact tip is green and owner visual acceptance is resolved. Production verification must check the release's actual target commit and URL.
