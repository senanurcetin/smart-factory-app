# CircleCI configuration safety and release rules

The repository uses CircleCI v2.1 with three separate quality gates: python-quality, browser-quality, and container-quality. All three must pass on the same PR head commit before merge.

## Prevent the prior configuration parser failure

The Docker smoke step includes a shell heredoc. An unescaped opening token is interpreted by CircleCI as a configuration parameter and can reject the whole workflow with an unclosed-template-tag error before any job starts.

To test the file without network or dependencies:

    python scripts/check_circleci_heredocs.py
    python -m unittest discover -s tests -p test_ci_heredoc_safety.py -v

The python-quality job now runs the preflight before installing large Python dependencies. The repository also offers an optional local pre-commit hook. After installing pre-commit, run:

    pre-commit install

This prevents new broken CircleCI config from being committed locally. CI cannot run its preflight at all if CircleCI refuses to parse the entire YAML, which is why the local hook matters. The regex detects raw shell heredoc openings, not normal CircleCI spaced template variables.

## Stable release practices

- Keep the container job on the real Render Docker image and verify Flask route responses.
- Keep browser-quality responsible for actual Chromium screenshots and mobile/desktop checks.
- Do not merge when any required job is failing, pending, or unreported.
- Do not replace captured application evidence with generated demo images or real-plant claims.
- Never store GCP private keys, Kaggle access tokens or production telemetry in CI YAML or repository files.
- If CI cannot provide a log, the failure cause is UNKNOWN until the job step output is inspected.
