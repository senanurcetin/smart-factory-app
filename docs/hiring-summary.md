# Hiring Summary

## One-line summary

Predictive-maintenance support case study that combines a plant-facing dashboard with a real failure-risk benchmark, review-budget logic, and an illustrative maintenance cost model.

## Headline metrics

- ROC-AUC: `0.9686`
- PR-AUC: `0.8522`
- F1: `0.8033`
- Top `10%` maintenance queue captures `92.6%` of holdout failures with `9.26x` better failure yield than random review

## What the project proves

- I can turn predictive-maintenance data into a prioritization system rather than a chart-only demo.
- I can work with imbalanced failure data and choose metrics that fit the operational problem.
- I can translate model output into queue design and maintenance economics that a plant reviewer can understand.

## Interview-ready talking points

1. The repo started as a synthetic Industry 4.0 dashboard and was upgraded with a real UCI predictive-maintenance benchmark.
2. I used PR-AUC and queue lift, not just accuracy, because the failure class is rare and maintenance teams care about prioritization quality.
3. I framed the product around high-risk asset review and cost tradeoffs instead of overselling the model as a plant-ready deployment.
