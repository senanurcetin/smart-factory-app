# Smart Factory App — RUL Case Study (NASA C-MAPSS)

## Positioning

This is a second, separate case study from [`docs/case-study.md`](case-study.md). AI4I 2020 is a static snapshot dataset — one row per independently sampled machine, no time axis — so it cannot support genuine remaining-useful-life (RUL) regression, a limitation disclosed throughout this repo. NASA's C-MAPSS turbofan degradation dataset is a real run-to-failure dataset, which makes real RUL regression possible, and it also has the structure (multiple cycles per engine) to support a proper grouped train/validation split — the exact rigor technique AI4I's structure could not support.

---

## Problem

Given a stream of sensor readings from an aircraft engine over its operating life, predict how many operating cycles remain before it fails — a genuine RUL regression problem, evaluated on the benchmark's own held-out test protocol.

---

## Dataset and Signal

| Property | Value |
|----------|-------|
| Dataset | NASA C-MAPSS Turbofan Engine Degradation Simulation |
| Subset used | FD001 (one operating condition, one fault mode — the simplest of the four C-MAPSS subsets) |
| Reference | Saxena, Goebel, Simon & Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, 2008 |
| Train engines | 100 (run to failure — 20,631 total cycle-rows) |
| Test engines | 100 (truncated before failure; official ground-truth RUL provided separately) |
| Raw signals | 3 operational settings + 21 sensor measurements per cycle |

Source: `analysis/run_cmapss_rul_case_study.py` downloads and caches the official NASA PHM datasets archive on first run.

---

## Feature Engineering

- **Sensor selection**: 6 of 21 sensors are ~constant under FD001's single operating condition and are dropped programmatically (variance threshold), not by memorized assumption.
- **Rolling-window features**: for each kept sensor, a per-engine rolling mean and standard deviation over the last 5 cycles (`min_periods=1`, so early cycles still get a value from partial history). A single raw reading is noisy; a short window captures trend direction and volatility that tracks degradation better than one snapshot.
- **Target**: `RUL = max_cycle_for_engine - current_cycle`, capped at 125 cycles — standard C-MAPSS practice, since early-life RUL is non-linear and largely uninformative (an engine at cycle 10 of 200 isn't meaningfully "healthier" than one at cycle 30 of 220).

Isolated impact of the rolling-window features (same tuned hyperparameters applied to raw sensors alone vs. raw + rolling-window features, so the delta is attributable to the engineered features alone): **RMSE −0.06 cycles, PHM08 score −107**. Small but real — reported honestly rather than inflated; not every feature engineering idea produces a large lift, and this one's effect is modest for this model architecture.

---

## Modeling Approach

- **Model**: `HistGradientBoostingRegressor` — consistent with the AI4I case study's choice: no feature scaling needed, trains in seconds at this data size, and supports exact/fast SHAP explanations via `TreeExplainer` (versus the approximate methods deep learning would require).
- **Hyperparameter search**: `RandomizedSearchCV` (15 iterations) over `max_depth`, `learning_rate`, `l2_regularization`, `max_leaf_nodes`, scored on negative RMSE.
- **Cross-validation**: `GroupKFold` (5 folds) grouped by `unit_number` — cycles from the same engine never appear on both sides of a fold. This is the grouped/time-aware split AI4I's one-row-per-machine structure could not support; C-MAPSS's multi-cycle-per-engine structure makes it both possible and necessary (a plain `KFold` would let the model "cheat" by seeing other cycles of the engine it's being validated on).
- **Evaluation protocol**: NASA's own — exactly one row per test engine (its final observed cycle), scored against the officially provided `RUL_FD001.txt` ground truth. Not a custom holdout.

---

## Results

| Metric | Naive baseline (median training RUL) | Tuned model |
|--------|----------------------------------------|-------------|
| RMSE (cycles) | 49.82 | **18.05** |
| PHM08 score | 166,570.54 | **837.60** |

**63.8% RMSE reduction, 99.5% PHM08-score reduction** versus the naive baseline. An RMSE of ~18 cycles is within the range typically reported for classical (non-deep-learning) approaches on FD001 in the published literature — this is a credible, literature-consistent result, not a bug producing suspiciously good numbers.

### PHM08 score

NASA's PHM08 challenge scoring function is deliberately asymmetric: overestimating remaining life (a late/optimistic prediction) is penalized more heavily than underestimating it (an early/conservative prediction) — pulling an engine from service too early costs money, but flying one too long on an over-optimistic RUL estimate is a safety risk. The naive baseline's huge PHM08 score (vs. its comparatively modest RMSE gap) reflects how badly a constant prediction performs against that asymmetric penalty on engines that failed much earlier or later than the median.

### Best hyperparameters

`max_leaf_nodes=31`, `max_depth=6`, `learning_rate=0.05`, `l2_regularization=0.5`. Full search config and CV results: `docs/data/cmapss-rul-case-study/model-selection.json` → `hyperparameter_tuning`.

Charts: `docs/assets/cmapss-model-vs-baseline.png`, `docs/assets/cmapss-predicted-vs-actual.png`, `docs/assets/cmapss-degradation-trajectories.png`, `docs/assets/cmapss-feature-importance.png`.

---

## Explainability

SHAP `TreeExplainer` (exact for tree ensembles, unlike the approximate interventional method needed for the AI4I classifier's larger background-sample estimation) on the official test set. The top signal is `sensor_4_roll_mean`, followed by `sensor_9_roll_mean`, `sensor_11_roll_mean`, and `sensor_15_roll_mean` — rolling means dominate over raw or rolling-std features, consistent with degradation being a slow trend rather than a noisy spike. Full ranking: `docs/data/cmapss-rul-case-study/feature-importance.json`.

---

## Operational Framing

1. Score every engine's latest cycle for predicted RUL.
2. Flag engines whose predicted RUL falls below a maintenance-planning threshold for inspection scheduling.
3. Because PHM08 penalizes over-optimistic predictions harder, a deployed threshold would be set conservatively (bias toward earlier flags) rather than tuned purely on RMSE.
4. Kept as a separate results page (`/rul-case-study`) rather than wired into the live bento dashboard — the dashboard simulates a different asset type entirely (AI4I-schema CNC/machine-tool telemetry, not turbofan engine sensors), and mixing the two would mean fabricating a connection between unrelated sensor schemas.

---

## Limitations

- FD001 is the simplest C-MAPSS subset (one operating condition, one fault mode); FD002/FD004 (six conditions, two fault modes) are harder and not covered here — noted as the natural next step.
- The RUL cap (125 cycles) follows common practice in the literature but is a modeling choice, not a physical constant.
- Per-cycle "true RUL" shown in the sample degradation-trajectory chart, before an engine's official final test cycle, is a linear back-projection from the single officially truthed point — illustrative only, never used for scoring.
- This is simulated benchmark data from the 2008 PHM challenge, not flight data from a real fleet.
- The rolling-window feature engineering's measured lift over raw sensors is modest (see above) — reported as found, not inflated.
