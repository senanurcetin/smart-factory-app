# Smart Factory — Render release and rollback verification

> **Historical production/browser QA — 14 October 2026:** Render `smart-factory-app` in the SmartFaactory workspace deployed the current README screenshot revision `0512457` as **live**. The original app-content hotfix was `747f0ce` (PR #23); production accepted seven HTTP routes/assets and six Chromium viewport checks. The steps below remain the release and rollback checklist **for future code deployments**, not a claim that the 14 October hotfix is waiting for merge. Subsequent hosted CircleCI `container-quality` success is now verified as described below.

---

## Current release evidence — 17 October 2026 check

- Repository: `senanurcetin/smart-factory-app`, `main` SHA `61dd76cb944bdedaf7951e59f73c003817655ee3` (merge PR #31, 16 October 2026).
- Hosted CircleCI results attached to **that exact SHA**: [Python quality #88](https://circleci.com/gh/senanurcetin/smart-factory-app/88) **SUCCESS**, [browser quality #89](https://circleci.com/gh/senanurcetin/smart-factory-app/89) **SUCCESS**, [container quality #90](https://circleci.com/gh/senanurcetin/smart-factory-app/90) **SUCCESS**.
- Render workspace `SmartFaactory`, service `smart-factory-app`, deployment `dep-db911t9srm7s73a4ve0g` is marked **live**, from exactly `61dd76cb944bdedaf7951e59f73c003817655ee3` (finished 16 October 2026 11:58 UTC).
- These are GitHub CI and **Render control-plane** observations. They do not establish an independently executed live HTTP/Chromium smoke on the 16 October SHA. The previously documented 14 October live HTTP and browser checks remain historical evidence for that earlier build. Re-run the HTTP/browser checklist below before calling a new full production acceptance.
- AI4I metrics in [committed benchmark JSON](data/ai4i-case-study/summary.json) and README: ROC-AUC **0.9874**, PR-AUC **0.9019**, F1 **0.8293**; NASA C-MAPSS FD001 RUL RMSE **18.0535** from [the independent RUL study](data/cmapss-rul-case-study/summary.json). These are offline simulated-dataset benchmark metrics, **not live plant measurements**.

## Release artifact

The production process starts `gunicorn ... main:app` from the final
`Dockerfile` image. Every local Python module imported by `main.py`
must be copied into that final stage, including `decision_studio.py`.
The Flask CSS/JS/HTML files under `static/` must also be included.

## Pre-merge acceptance

- CircleCI `python-quality`: Ruff, Black, artifact, model and packaging contract tests
- CircleCI `browser-quality`: real Chromium route and responsive tests
- CircleCI `container-quality`: a real Docker build, process startup and live HTTP 200 + expected-content checks for dashboard API, maintenance, RUL, two studio routes and NEXUS styles/scripts
- Review the PR code diff and merge only the passing head SHA

## Render production verification

- Identify and explicitly select the intended Render workspace and `smart-factory-app` service; do not select an unrelated workspace
- Determine whether the service auto-deploys the `main` branch
- Confirm the Render deployment commit matches the GitHub merge commit
- Verify `/api/data`, `/case-study`, `/rul-case-study`, `/studio/maintenance` and `/studio/rul` are accessible
- Confirm HTML references and responses for `/static/nexus-factory-studio.css` and `/static/decision-studio.js` are valid
- Run desktop/tablet/mobile production Chromium smoke without asserting that synthetic AI4I telemetry is real plant data

## Rollback threshold

If the new production image fails health checks, Flask import, static resources,
or any critical control route, rollback to the last healthy Render release.
Do not claim the UI deployed merely because GitHub reports a successful merge.
