# Smart Factory App Case Study

## Positioning

`smart-factory-app` is a predictive-maintenance support case study. It combines a plant-facing KPI dashboard with a measurable failure-risk benchmark on a public maintenance dataset.

## Problem

Maintenance teams do not benefit from a risk score alone. They need to know which assets deserve attention first, what failure modes dominate, and how limited review capacity should be spent.

## Dataset and signal

- Dataset: `AI4I 2020 Predictive Maintenance Dataset`
- Source: UCI Machine Learning Repository
- Rows: `10,000`
- Failure rate: `3.39%`
- Features used:
  - product type
  - air temperature
  - process temperature
  - rotational speed
  - torque
  - tool wear

## Modeling approach

- Preprocessing:
  - median imputation
  - feature scaling for numeric inputs
  - one-hot encoding for product type
- Benchmarks:
  - dummy baseline
  - logistic regression
  - random forest
  - HistGradientBoostingClassifier
- Evaluation:
  - deterministic `80/20` stratified holdout
  - ROC-AUC
  - PR-AUC
  - precision, recall, F1 at `0.5`
  - maintenance review queue analysis

## Why the final model was selected

HistGradientBoosting was selected because it delivered the strongest PR-AUC and F1 on the imbalanced holdout without introducing GPU assumptions or non-free dependencies.

## Current results

- ROC-AUC: `0.9686`
- PR-AUC: `0.8522`
- Precision: `0.9074`
- Recall: `0.7206`
- F1: `0.8033`
- Queue result: reviewing the top `10%` highest-risk assets captures `92.6%` of holdout failures with `9.26x` better failure yield than random review

## Operational framing

The public story is maintenance triage:

- score failure risk
- route a small queue of high-risk assets to engineers
- use the dashboard for KPI and context
- use the benchmark layer to justify the prioritization logic

## What this proves

- predictive-maintenance work can be presented as a measured operational decision system, not only as a dashboard
- imbalanced classification can be tied to review-budget logic
- business-facing maintenance decisions can be translated into a zero-cost, reproducible portfolio artifact

## Limitations

- the UCI dataset is simulated rather than plant-specific
- the cost model is illustrative
- the application UI remains a lightweight Flask demo instead of a production monitoring system
