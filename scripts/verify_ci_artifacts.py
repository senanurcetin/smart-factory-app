#!/usr/bin/env python3
"""Verify committed model and report artifacts without network access.

Used by CircleCI's offline quality job. Deliberately does not unpickle models,
retrain them, connect to production services, or require downloaded datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "analysis/artifacts/model.pkl",
    "analysis/artifacts/rul_model.pkl",
    "docs/data/ai4i-case-study/summary.json",
    "docs/data/cmapss-rul-case-study/summary.json",
    "docs/data/ai4i-case-study/drift-report.json",
)


def verify_committed_artifacts(repo_root: Path) -> None:
    for relative_name in REQUIRED_ARTIFACTS:
        artifact = repo_root / relative_name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise SystemExit(f"Missing or empty committed artifact: {relative_name}")
        if artifact.suffix == ".json":
            with artifact.open(encoding="utf-8") as stream:
                json.load(stream)
        print(f"OK: {relative_name}")


if __name__ == "__main__":
    verify_committed_artifacts(Path(__file__).resolve().parents[1])
