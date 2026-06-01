# Analysis

This folder upgrades `smart-factory-app` from a synthetic dashboard demo into a measurable predictive-maintenance support case study.

## Dataset

- Dataset: `AI4I 2020 Predictive Maintenance Dataset`
- Source: UCI Machine Learning Repository
- DOI: `10.24432/C5HS5C`
- Scope: 10,000 machine records with failure labels and failure-mode annotations

## What the pipeline does

`run_ai4i_case_study.py` downloads the official UCI dataset into `analysis/.cache/`, benchmarks several models on an imbalanced failure target, builds a review-queue analysis, and writes recruiter-facing artifacts into `docs/data/ai4i-case-study/`.

## Run it

```bash
python analysis/run_ai4i_case_study.py
```

## Why this matters

The dashboard already shows KPI framing and plant context. This analysis layer adds a real predictive-maintenance benchmark, budgeted maintenance triage, and an illustrative cost model that is easier to defend in DS and applied-AI interviews.
