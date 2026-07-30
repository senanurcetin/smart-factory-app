# Hiring Summary

## One-line summary

Predictive-maintenance analytics case study covering EDA, feature engineering, hyperparameter tuning, SHAP explainability, model selection, DuckDB SQL analysis, review-queue design, and cost framing — packaged with a Flask dashboard that scores live telemetry with the same trained model, and 40 passing tests.

## Headline metrics

- Final model (HistGradientBoosting + engineered features + tuned hyperparameters): ROC-AUC **0.9874**, PR-AUC **0.9019**, F1 **0.8293**
- Robustness-checked across 5 independent splits: ROC-AUC 0.984 ± 0.004, PR-AUC 0.880 ± 0.032 — not a single lucky split
- Reviewing the top **10%** of assets captures **94.1%** of holdout failures — **9.4x** better yield than random review
- Estimated cost savings vs reactive maintenance: **$185,000 (77.7%)**
- Feature engineering impact, isolated from hyperparameter tuning: ROC-AUC +0.018, PR-AUC +0.077, F1 +0.082

## Skills demonstrated

- **Exploratory data analysis**: class imbalance identification, failure mode profiling, product type distribution
- **Feature engineering**: derived cross-sensor features (mechanical_load, thermal_stress, tool_wear_load_ratio), with impact isolated from hyperparameter tuning via a controlled comparison
- **Hyperparameter tuning**: `RandomizedSearchCV` with PR-AUC as the search objective, results logged transparently
- **Explainability**: SHAP TreeExplainer for interaction-aware local attribution, not a single-feature heuristic
- **Validation rigor**: repeated stratified holdouts to confirm metric stability where time-aware/group splitting isn't applicable
- **Model selection**: four-model benchmark with PR-AUC as primary metric for imbalanced data
- **SQL**: DuckDB analytical queries on JSON artifacts — failure mode ranking, model comparison, queue ROI
- **Business framing**: ranked review queue, cost model, operational maintenance economics
- **Engineering practice**: pinned dependencies, lint/format CI gate, model persisted via joblib and served (not retrained ad hoc) by the live app
- **Python stack**: pandas, NumPy, scikit-learn, SHAP, matplotlib, DuckDB, Flask
- **Testing**: 40 tests covering routes, artifact contracts, metrics thresholds, chart presence, and dashboard-model integration

## Interview-ready talking points

1. Class imbalance is identified in EDA and drives every downstream decision — PR-AUC and recall over accuracy throughout.
2. Feature engineering's value is isolated from hyperparameter tuning with a controlled comparison (same tuned params, raw vs. enhanced features) — a common conflation this project deliberately avoids.
3. Headline metrics are backed by a 5-seed robustness check, not reported from a single split; the writeup shows the metric that varies more (PR-AUC) and by how much.
4. SQL analysis is built directly into the pipeline: DuckDB queries on JSON artifacts show failure mode ranking, model comparison, and queue ROI in a reproducible, testable format.
5. The project frames predictive maintenance as a **prioritization problem**, not a chart demo — connecting model output to a ranked queue and maintenance cost model.
6. The live dashboard scores the same persisted model used in the offline benchmark (via joblib), not a disconnected demo model trained on random labels — a deliberate fix for a common portfolio-project credibility gap.
7. Limitations are documented honestly: random failures (RNF) have 25% capture rate; the UCI dataset is simulated; the cost model is illustrative; SHAP and dashboard RUL are both explicitly labeled as approximations, not exact/calibrated outputs.
