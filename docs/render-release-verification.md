# Smart Factory — Render release and rollback verification

> **Last verified production — 14 October 2026:** Render `smart-factory-app` in the SmartFaactory workspace deployed the current README screenshot revision `0512457` as **live**. The original app-content hotfix was `747f0ce` (PR #23); production accepted seven HTTP routes/assets and six Chromium viewport checks. The steps below remain the release and rollback checklist **for future code deployments**, not a claim that the 14 October hotfix is waiting for merge. Separate CircleCI `container-quality` job green status was not verified and remains a follow-up check.

---

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
