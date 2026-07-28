# Smart Factory App — Case Study

## Positioning

`smart-factory-app` is a predictive-maintenance analytics case study. It demonstrates an end-to-end analytical workflow: exploratory data analysis, feature engineering, model selection, SQL analysis, review-queue design, and cost framing — packaged with a plant-facing Flask dashboard.

---

## Problem

Maintenance teams do not benefit from a risk score alone. They need to know which assets deserve attention first, what failure modes dominate, how limited review capacity should be spent, and what the cost implications are.

---

## Dataset and Signal

| Property | Value |
|----------|-------|
| Dataset | AI4I 2020 Predictive Maintenance Dataset |
| Source | UCI Machine Learning Repository |
| DOI | 10.24432/C5HS5C |
| Rows | 10,000 |
| Failure rate | 3.39% (339 failures) |
| Failure modes | TWF, HDF, PWF, OSF, RNF |
| Raw features | Product type, air temp, process temp, rotational speed, torque, tool wear |

---

## Exploratory Data Analysis

Key findings from EDA:

- **Class imbalance**: 96.6% normal vs 3.39% failure. Accuracy misleads; PR-AUC and recall are the right metrics.
- **Failure mode distribution**: HDF (115) and OSF (98) dominate in the full dataset. RNF (19) is rare and has low model separability.
- **Product type**: Type L (60%), M (30%), H (10%). Type H is the rarest but highest-quality cohort.
- **Holdout split**: Stratified 80/20 preserves failure rate in both sets — 8,000 train rows, 2,000 holdout rows, 68 holdout failures.

EDA charts: `docs/assets/eda-class-balance.png`, `docs/assets/eda-failure-modes.png`, `docs/assets/eda-type-distribution.png`

---

## Feature Engineering

Three derived features capture cross-sensor interactions:

| Feature | Formula | Why |
|---------|---------|-----|
| `mechanical_load` | Torque x RPM | Power proxy — high load correlates with wear and overheating |
| `thermal_stress` | (Process temp - Air temp) / Air temp | Relative heat load better than absolute temperature |
| `tool_wear_load_ratio` | Tool wear / Torque | Wear rate under load — flags accelerated degradation |

Impact vs same model without derived features: ROC-AUC +0.006, PR-AUC +0.033, F1 +0.027.

---

## Modeling Approach

- **Preprocessing**: median imputation, standard scaling for numeric features, one-hot encoding for product type
- **Benchmarks**: dummy baseline, logistic regression, random forest, HistGradientBoostingClassifier
- **Evaluation**: deterministic 80/20 stratified holdout, ROC-AUC, PR-AUC, precision, recall, F1, review-queue analysis
- **Model selection criterion**: PR-AUC — correct primary metric for rare-event detection on imbalanced data

---

## Model Selection Results

All four models benchmarked with raw features (before feature engineering):

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|----|
| Dummy baseline | 0.500 | 0.034 | 0.000 |
| Logistic regression | 0.907 | 0.382 | 0.242 |
| Random forest | 0.965 | 0.762 | 0.695 |
| **HistGradientBoosting** | **0.975** | **0.852** | **0.810** |

HistGradientBoosting selected for highest PR-AUC and F1.

---

## Final Model Results

HistGradientBoosting with enhanced (derived) features:

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9807 |
| PR-AUC | 0.8848 |
| Precision | 0.8852 |
| Recall | 0.7941 |
| F1 | 0.8372 |
| Accuracy | 0.9895 |
| Calibration ECE | 0.0014 |

**Confusion matrix** (holdout, n=2,000):

|  | Predicted Normal | Predicted Failure |
|--|-----------------|------------------|
| **Actual Normal** | 1,925 (TN) | 7 (FP) |
| **Actual Failure** | 14 (FN) | 54 (TP) |

Chart: `docs/assets/eda-confusion-matrix.png`

---

## SQL Analysis

Four analytical queries run with DuckDB on JSON artifacts. Full results: `docs/data/ai4i-case-study/sql-analysis.json`.

Key finding: reviewing the top 10% of assets (200 machines) catches 94.1% of failures at 3.1 assets reviewed per failure caught — 9.4x better yield than random review.

---

## Operational Framing

1. Score failure risk for every asset using the trained model
2. Route the top-N highest-risk assets to maintenance engineers
3. Use the dashboard for KPI context and shift-level visibility
4. Use the benchmark artifacts to justify prioritization logic in stakeholder conversations

---

## Cost Model

From `docs/data/ai4i-case-study/cost-simulation.json`:

| Scenario | Estimated cost |
|----------|---------------|
| Reactive maintenance (no model) | $238,000 |
| Random review | $224,000 |
| Risk-ranked queue (top 10%) | $53,000 |
| Savings vs reactive | **$185,000 (77.7%)** |

Assumptions: $3,500 unplanned failure cost, $500 scheduled maintenance cost, $35 review cost per asset. All figures are illustrative.

---

## Limitations

- The UCI AI4I dataset is simulated — metrics are benchmark evidence, not plant-specific deployment claims
- The cost model is illustrative; real maintenance economics vary significantly by plant, equipment, and failure type
- Random failures (RNF) have low model separability — 25% capture rate at 10% budget is an acknowledged limitation
- Local attributions use median-ablation, not SHAP — may underestimate interaction effects
- The live dashboard simulates plausible AI4I-schema telemetry and scores it with the real trained pipeline (not a separate toy model); it is not connected to a deployed production sensor stream
