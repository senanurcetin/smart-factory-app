# Offline model candidate registry (manual approval only)

**Implementation:** [`analysis/model_registry.py`](../analysis/model_registry.py).
**Status:** Candidate preparation, validation and read-only comparison are
implemented. Automated training, promotion, rollback or production model swaps
are **not** implemented or authorized by this module.

## Why it exists

The existing AI4I `analysis/artifacts/model.pkl` and C-MAPSS
`analysis/artifacts/rul_model.pkl` are the active checked-in deployment
artifacts. Changing either silently would invalidate the published evidence.
The registry creates a separate, immutable **local candidate copy** named
by a SHA-256 digest, with a separate manifest that identifies the dataset
digest, evaluated UTC timestamp and held-out metrics.

The local candidate store under `analysis/artifacts/registry/` is ignored
by Git. The tests use only temporary directories with synthetic byte strings;
they **never unpickle** a model or download any dataset.

## Explicit offline staging example

```python
from pathlib import Path
from analysis.model_registry import stage_candidate, verify_candidate

root = Path.cwd()
candidate = stage_candidate(
    root,
    "ai4i_failure",
    dataset_sha256="<64-character verified lowercase dataset SHA-256>",
    metrics={"pr_auc": 0.9019, "roc_auc": 0.9874, "f1": 0.8293},
    evaluated_at="2026-10-17T18:00:00+03:00",
)
assert verify_candidate(root, candidate) == candidate
```

This example is illustrative: **never assert a dataset digest without
independently verifying the bytes**. The caller is responsible for independently
proving that the supplied evaluation metrics belong to the exact staged model
and split; the registry validates structure/integrity, not statistical
provenance. Use the committed summary JSON files for existing benchmark
context.

Supported model families:

| `model_id` | Active source | Required candidate metric | Direction |
| --- | --- | --- | --- |
| `ai4i_failure` | `model.pkl` | `pr_auc` | Higher is better |
| `cmapss_fd001_rul` | `rul_model.pkl` | `rmse` | Lower is better |

The registry rejects unexpected paths, nonfinite/boolean metrics, invalid
hashes, missing/modified bytes, symlinks, unsupported schemas and attempts
to forge candidate approval by editing metadata.

## Review gates

```python
from analysis.model_registry import assess_candidate

review = assess_candidate(root, baseline_manifest, candidate_manifest)
# review["status"] is "manual_review_required" or "rejected_no_improvement"
assert review["automatic_promotion"] is False
```

Passing the primary metric gate **does not** establish generalization,
calibration, schema compatibility or plant validity. Manual release review
must also cover the full benchmark, subgroup/failure-mode behavior,
out-of-distribution tests, model serialization compatibility and rollback
readiness. Model rollback is not offered until it is tested independently.

## Verification

```bash
python -m unittest discover -s tests -p test_model_registry.py -v
ruff check analysis/model_registry.py tests/test_model_registry.py
black --check analysis/model_registry.py tests/test_model_registry.py
```

The same unit test file is listed in CircleCI `python-quality`, but the
branch is deliberately kept as a draft with `[skip ci]` commits while the
existing PR #34 Python-quality failure and the account's CircleCI quota
constraint are unresolved. Do **not** describe it as CI-green until that
suite actually runs and passes on the exact commit.
