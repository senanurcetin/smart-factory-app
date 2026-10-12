# Reviewer brief — Smart Factory Predictive Maintenance

**Role:** Manufacturing analytics, predictive-maintenance ML, risk-prioritized work queues and model validation. **Evidence source:** UCI AI4I 2020 and NASA C-MAPSS research datasets.

## Decision problem

With limited maintenance review capacity, how should a plant rank equipment snapshots for investigation? The project compares models, translates risk into a review-queue/cost scenario and adds a separate RUL regression case study.

| Assessment | Evidence |
|---|---|
| Is model performance benchmarked against baselines? | Final AI4I HistGradientBoosting: **ROC-AUC 0.9874**, **PR-AUC 0.9019**, **F1 0.8293**; [full README](../README.md#final-model-performance) |
| Is the chosen split robust? | Across five stratified splits: ROC-AUC **0.984 ± 0.004**, PR-AUC **0.880 ± 0.032**; [robustness discussion](../README.md#validation-robustness) |
| Can I inspect source outputs? | [AI4I metrics JSON](data/ai4i-case-study/summary.json), [full AI4I study](case-study.md), [RUL study](rul-case-study.md) |
| Does a UI exist? | [Captured dashboard](assets/smart-factory-dashboard.png) and [hosted demo](https://smart-factory-app.onrender.com) |
| How does it handle operations decisions? | [Review-queue analysis](../README.md#maintenance-review-queue) and [cost model](assets/cost-model.png) |

## Engineering evidence

- Derived process features, explicit model comparisons, repeat-split checks and explainability analysis.
- Separation of binary failure risk from C-MAPSS **remaining-useful-life regression**.
- Flask application and persisted scoring artifacts for demonstrable integration.

## Non-claims

AI4I is simulated and dashboard readings are not live sensors. The dashboard's simplified RUL indicator is **not** the NASA RUL regression model. Maintenance costs are illustrative, not a measured plant ROI. No safety or maintenance action should be automated based on these results.

**Detailed methodology:** [AI4I study](case-study.md) · [C-MAPSS study](rul-case-study.md).
