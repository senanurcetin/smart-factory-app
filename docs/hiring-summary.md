# Hiring Summary

## One-line summary

Predictive-maintenance analytics case study covering EDA, feature engineering, model selection, DuckDB SQL analysis, review-queue design, and cost framing — packaged with a Flask dashboard and 27 passing tests.

## Headline metrics

- Final model (HistGradientBoosting + engineered features): ROC-AUC **0.9819**, PR-AUC **0.8855**, F1 **0.8372**
- Reviewing the top **10%** of assets captures **94.1%** of holdout failures — **9.4x** better yield than random review
- Estimated cost savings vs reactive maintenance: **$185,000 (77.7%)**
- Feature engineering impact: ROC-AUC +0.013, PR-AUC +0.033 vs same model without derived features

## Skills demonstrated

- **Exploratory data analysis**: class imbalance identification, failure mode profiling, product type distribution
- **Feature engineering**: derived cross-sensor features (mechanical_load, thermal_stress, tool_wear_load_ratio)
- **Model selection**: four-model benchmark with PR-AUC as primary metric for imbalanced data
- **SQL**: DuckDB analytical queries on JSON artifacts — failure mode ranking, model comparison, queue ROI
- **Business framing**: ranked review queue, cost model, operational maintenance economics
- **Python stack**: pandas, NumPy, scikit-learn, matplotlib, DuckDB, Flask
- **Testing**: 27 tests covering routes, artifact contracts, metrics thresholds, and chart presence

## Interview-ready talking points

1. Class imbalance is identified in EDA and drives every downstream decision — PR-AUC and recall over accuracy throughout.
2. Feature engineering is quantified: three derived features improve PR-AUC by 0.033 and are ranked in the top 5 by permutation importance.
3. SQL analysis is built directly into the pipeline: DuckDB queries on JSON artifacts show failure mode ranking, model comparison, and queue ROI in a reproducible, testable format.
4. The project frames predictive maintenance as a **prioritization problem**, not a chart demo — connecting model output to a ranked queue and maintenance cost model.
5. Limitations are documented honestly: random failures (RNF) have 25% capture rate; the UCI dataset is simulated; the cost model is illustrative.
