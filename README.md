# Smart Factory App

[![CI](https://github.com/senanurcetin/smart-factory-app/actions/workflows/ci.yml/badge.svg)](https://github.com/senanurcetin/smart-factory-app/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Predictive-maintenance analytics case study** — end-to-end ML pipeline on the UCI AI4I 2020 dataset, including EDA, feature engineering, model selection, SQL analysis, review-queue design, and cost framing. Packaged with a Flask plant dashboard.

Demo: [Portfolio project entry](https://senanur-cetin.vercel.app/projects/smart-factory-app)

Short video: [`docs/assets/smart-factory-dashboard.webm`](docs/assets/smart-factory-dashboard.webm)

![Smart Factory dashboard](docs/assets/smart-factory-dashboard.png)

---

## Problem

Maintenance teams cannot review every asset equally. The question is whether sensor telemetry can be translated into a **ranked work queue** that captures most failures within a limited review budget — and whether that logic can be expressed in terms of cost and business value.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| Records | 10,000 |
| Failure rate | 3.39% — 339 failures across 9,661 normal records |
| Evaluation | Deterministic 80/20 stratified holdout (seed=42) |
| Raw features | Product type, air temp, process temp, rotational speed, torque, tool wear |

---

## Exploratory Data Analysis

### Class imbalance

The dataset is heavily imbalanced at 3.39% failure rate. Accuracy is a misleading metric here; PR-AUC and recall are the correct evaluation signals.

![Class balance](docs/assets/eda-class-balance.png)

### Failure mode distribution

Heat dissipation (HDF) and overstrain (OSF) are the most frequent failure types. Random failures (RNF) are rare and poorly separable — an honest limitation of the benchmark.

![Failure modes](docs/assets/eda-failure-modes.png)

### Product type distribution

Type L (low quality) makes up 60% of records. Type H (high quality) is the smallest slice at 10%.

![Product type distribution](docs/assets/eda-type-distribution.png)

---

## Feature Engineering

Three cross-sensor features were derived to capture interaction effects that individual raw sensors cannot express:

| Derived feature | Formula | Intuition |
|-----------------|---------|-----------|
| `mechanical_load` | Torque x RPM | Proxy for power delivered to the spindle |
| `thermal_stress` | (Process temp - Air temp) / Air temp | Relative thermal load under operating conditions |
| `tool_wear_load_ratio` | Tool wear / Torque | Wear rate per unit of mechanical load |

Isolated impact (same tuned hyperparameters, raw vs. enhanced features, so the delta is attributable to the three derived features alone): ROC-AUC **+0.018**, PR-AUC **+0.077**, F1 **+0.082**.

---

## Model Selection

Four models benchmarked on the holdout using the **raw feature set** (before derived features):

![Model comparison](docs/assets/model-comparison.png)

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|----|
| Dummy baseline | 0.500 | 0.034 | 0.000 |
| Logistic regression | 0.907 | 0.382 | 0.242 |
| Random forest | 0.965 | 0.762 | 0.695 |
| **HistGradientBoosting** (selected) | **0.975** | **0.852** | **0.810** |

HistGradientBoosting selected for highest PR-AUC and F1 on the imbalanced holdout. This table uses raw features and default hyperparameters for a fair four-model comparison; the deployed model additionally adds the derived features above and a tuned hyperparameter search (see [Final Model Performance](#final-model-performance)).

---

## Feature Importance

Permutation importance (average precision scoring, 8 repeats) on the final enhanced model:

![Feature importance](docs/assets/feature-importance.png)

Rotational speed and `thermal_stress` dominate. Derived features rank in the top 5, validating the feature engineering step.

---

## Final Model Performance

HistGradientBoosting with derived features and a tuned hyperparameter search (`max_leaf_nodes=15`, `learning_rate=0.03`, `l2_regularization=1.0`, unrestricted `max_depth`; see [Hyperparameter Tuning](#hyperparameter-tuning)) achieves:

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9874** |
| PR-AUC | **0.9019** |
| Precision | **0.9273** |
| Recall | **0.75** |
| F1 | **0.8293** |
| Calibration ECE | 0.007 (well-calibrated) |

### Confusion matrix

![Confusion matrix](docs/assets/eda-confusion-matrix.png)

Out of 68 actual failures in the holdout: **51 correctly flagged (TP)**, 17 missed (FN), only 4 false alarms (FP). Tuning traded some recall for precision relative to the untuned model — see the Hyperparameter Tuning section for why PR-AUC (threshold-independent) was used as the search objective instead of F1.

---

## Hyperparameter Tuning

The deployed model's hyperparameters were selected with `RandomizedSearchCV` (20 iterations, 5-fold stratified CV, scored on PR-AUC — the same primary metric used everywhere else in this repo) over `max_depth`, `learning_rate`, `l2_regularization`, and `max_leaf_nodes`. Full search config and results: [`docs/data/ai4i-case-study/model-selection.json`](docs/data/ai4i-case-study/model-selection.json).

---

## Explainability (SHAP)

Local feature attribution for the top 10 highest-risk holdout predictions uses `shap.TreeExplainer` (interventional, probability-space) rather than a single-feature-ablation heuristic, so it correctly accounts for feature interactions (e.g. torque and tool wear jointly driving overstrain risk). Detail: [`docs/data/ai4i-case-study/feature-importance.json`](docs/data/ai4i-case-study/feature-importance.json) → `local_attributions_top10_high_risk`.

---

## Validation Robustness

AI4I 2020 has no time axis or repeated-asset grouping — each of the 10,000 rows is one independently sampled machine snapshot with a unique Product ID — so time-aware or group-based (`GroupKFold`) splitting is not applicable to this dataset. Instead, the final architecture was re-fit across 5 independent stratified 80/20 splits to confirm the headline metrics aren't an artifact of one lucky split:

**ROC-AUC 0.984 ± 0.004, PR-AUC 0.880 ± 0.032** across seeds. PR-AUC varies more than ROC-AUC across splits (expected — only ~68 failures land in each 20% holdout), so the single-split PR-AUC of 0.90 reported above sits on the higher end of that range rather than being a guaranteed number. Full per-seed results: [`docs/data/ai4i-case-study/validation-robustness.json`](docs/data/ai4i-case-study/validation-robustness.json).

---

## SQL Analysis

Analytical queries run with **DuckDB** directly on the pre-computed JSON artifacts. Results saved to [`docs/data/ai4i-case-study/sql-analysis.json`](docs/data/ai4i-case-study/sql-analysis.json).

**Q1 — Failure modes ranked by frequency and capture rate:**

| Code | Label | Holdout failures | Capture rate at top 10% |
|------|-------|-----------------|------------------------|
| HDF | Heat dissipation | 29 | 100% |
| OSF | Overstrain | 16 | 100% |
| PWF | Power failure | 13 | 100% |
| TWF | Tool wear | 10 | 70% |
| RNF | Random | 4 | 25% |

RNF has the lowest separability — acknowledged as a model limitation.

**Q2 — Models ranked by PR-AUC (correct metric for imbalanced data):**

HistGradientBoosting 0.852 > RandomForest 0.762 > LogisticRegression 0.382 > Dummy 0.034

**Q3 — Review queue ROI:**

| Budget | Failures caught | Yield lift | Assets per failure caught |
|--------|----------------|-----------|--------------------------|
| 5% — 100 assets | 89.7% | 17.9x | 1.6 |
| 10% — 200 assets | 94.1% | 9.4x | 3.1 |
| 15% — 300 assets | 95.6% | 6.4x | 4.6 |

---

## Maintenance Review Queue

![Review queue curve](docs/assets/review-queue-curve.png)

Reviewing the **top 10% of assets** (200 machines) captures **94.1% of all holdout failures** — 9.4x better than random review.

---

## Cost Impact

From [`docs/data/ai4i-case-study/cost-simulation.json`](docs/data/ai4i-case-study/cost-simulation.json):

| Scenario | Cost |
|----------|------|
| Reactive maintenance (no model) | $238,000 |
| Random review | $224,000 |
| Risk-ranked queue (top 10%) | $53,000 |
| **Savings vs reactive** | **$185,000 (77.7%)** |

Assumptions are illustrative — demonstrates how model output translates into maintenance economics.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Application | Python 3.11, Flask |
| Analytics | Pandas, NumPy, SciPy, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| SQL analysis | DuckDB (in-memory, reads JSON natively) |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, JavaScript |
| CI | GitHub Actions |

---

## Architecture

```
smart-factory-app/
├── analysis/
│   ├── run_ai4i_case_study.py   # Full ML pipeline (downloads UCI dataset on first run)
│   ├── generate_visuals.py      # Model comparison, feature importance, review queue curve
│   ├── generate_eda.py          # EDA charts: class balance, failure modes, confusion matrix
│   ├── sql_queries.py           # DuckDB SQL analysis on JSON artifacts
│   └── artifacts/model.pkl      # Persisted final pipeline — also scored live by main.py
├── docs/
│   ├── assets/                  # All PNG charts (9 files)
│   ├── data/ai4i-case-study/    # JSON artifacts + sql-analysis.json
│   ├── case-study.md
│   └── hiring-summary.md
├── tests/
│   ├── test_case_study.py       # Route and integration tests
│   ├── test_dashboard_model.py  # Live dashboard scores against the real trained pipeline
│   └── test_artifacts.py        # Data quality, metrics range, asset presence (33 tests)
├── main.py
├── case_study.py
├── requirements.txt
└── requirements-dev.txt         # + ruff, black
```

---

## Local Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements-dev.txt   # or requirements.txt if you don't need lint/format tooling

# Generate all charts from JSON artifacts (no dataset download needed)
python analysis/generate_eda.py
python analysis/generate_visuals.py

# Run SQL analysis
python analysis/sql_queries.py

# Run full ML benchmark (downloads UCI dataset ~500 KB on first run).
# This also persists the final pipeline to analysis/artifacts/model.pkl,
# which main.py loads to score the live dashboard — a pre-trained copy is
# already committed, so this step is optional unless you want to retrain.
python analysis/run_ai4i_case_study.py

# Start the dashboard (scores simulated AI4I-schema telemetry with the
# real trained pipeline above — not a separate toy model)
python main.py
```

App: `http://127.0.0.1:8080` | Case-study route: `http://127.0.0.1:8080/case-study`

---

## Tests

```bash
# All 40 tests: routes, artifact contracts, metrics thresholds, chart file
# presence, hyperparameter-tuning/validation-robustness contracts, and live
# dashboard <-> real model integration
python -m unittest discover -s tests -v

# Lint and format check
ruff check .
black --check .

# Syntax check all modules
python -m py_compile main.py case_study.py \
    analysis/run_ai4i_case_study.py \
    analysis/generate_visuals.py \
    analysis/generate_eda.py \
    analysis/sql_queries.py
```

---

## Proof Surfaces

| Document | Contents |
|----------|----------|
| [`docs/case-study.md`](docs/case-study.md) | Full methodology, results, limitations |
| [`docs/hiring-summary.md`](docs/hiring-summary.md) | Recruiter-facing one-page summary |
| [`docs/data/ai4i-case-study/summary.json`](docs/data/ai4i-case-study/summary.json) | Final model metrics and confusion matrix |
| [`docs/data/ai4i-case-study/sql-analysis.json`](docs/data/ai4i-case-study/sql-analysis.json) | DuckDB query results |
| Local route | `http://127.0.0.1:8080/case-study` |

---

## What This Proves

- Class imbalance identified up front; PR-AUC and recall drive metric selection throughout
- Feature engineering's marginal value is isolated from hyperparameter tuning (same tuned params, raw vs. enhanced features): ROC-AUC +0.018, PR-AUC +0.077, F1 +0.082
- Hyperparameters are searched (`RandomizedSearchCV`, PR-AUC objective), not hardcoded guesses
- Reported metrics are backed by a repeated-holdout robustness check across 5 seeds, not a single lucky split
- Local explanations use SHAP (interaction-aware), not a single-feature-ablation heuristic
- SQL proficiency: DuckDB analytical queries on structured JSON artifacts
- Model output translated into a ranked maintenance queue with quantified business ROI
- The live dashboard scores simulated telemetry with the same trained pipeline benchmarked in the offline case study — no separate, disconnected demo model
- End-to-end analytical thinking: EDA to feature engineering to model selection to business framing

---

## Limitations

- The UCI AI4I dataset is simulated — metrics are benchmark evidence, not plant-specific deployment claims
- The live dashboard simulates plausible AI4I-schema telemetry rather than reading a deployed production sensor stream (the risk model itself is real, not a toy)
- The cost model is illustrative; real maintenance economics vary by plant and equipment type
- Random failures (RNF) have low separability — 25% capture rate is an acknowledged limitation
- The dashboard's RUL figure is a simple heuristic derived from the model's risk score, not a calibrated survival/RUL regression
- SHAP attributions use an interventional TreeExplainer with a 200-row background sample — a close but not exact approximation of full-dataset Shapley values
- AI4I 2020 has no time axis or asset-grouping structure to split on, so split stability is checked via repeated holdouts (see [Validation Robustness](#validation-robustness)) rather than time-aware or group-based cross-validation

---

## License

MIT
