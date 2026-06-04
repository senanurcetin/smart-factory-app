# Smart Factory App

[![CI](https://github.com/senanurcetin/smart-factory-app/actions/workflows/ci.yml/badge.svg)](https://github.com/senanurcetin/smart-factory-app/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Predictive-maintenance support case study** for manufacturing teams. Combines two portfolio layers in one repo:

- a Flask dashboard for plant KPIs, machine-health context, and maintenance-oriented UI packaging
- a reproducible ML benchmark on the UCI AI4I 2020 dataset with measurable review-queue and cost logic

Demo: [Portfolio project entry](https://senanur-cetin.vercel.app/projects/smart-factory-app)

Short video: [`docs/assets/smart-factory-dashboard.webm`](docs/assets/smart-factory-dashboard.webm)

![Smart Factory dashboard](docs/assets/smart-factory-dashboard.png)

---

## Problem

Maintenance teams cannot review every asset equally. The useful question is whether sensor telemetry can be translated into a **ranked work queue** that captures most failures inside a small review budget.

A dashboard is only valuable if it changes maintenance prioritization — supporting scheduling and escalation decisions, not just visualizing telemetry.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| Records | 10,000 |
| Failure rate | 3.39% (339 failures) |
| Evaluation | Deterministic 80/20 stratified holdout |
| Features | Product type, air temp, process temp, rotational speed, torque, tool wear |
| Derived | `mechanical_load`, `thermal_stress`, `tool_wear_load_ratio` |

---

## Model Benchmark

Four models benchmarked on the imbalanced holdout:

![Model comparison](docs/assets/model-comparison.png)

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|----|
| Dummy baseline | 0.500 | 0.034 | 0.000 |
| Logistic regression | 0.907 | 0.382 | 0.242 |
| Random forest | 0.961 | 0.772 | 0.717 |
| **HistGradientBoosting** ✓ | **0.982** | **0.886** | **0.837** |

HistGradientBoosting selected for best PR-AUC and F1 on the imbalanced holdout without GPU assumptions or non-free dependencies.

---

## Feature Importance

Permutation importance (average precision scoring, 8 repeats):

![Feature importance](docs/assets/feature-importance.png)

Rotational speed and thermal stress dominate. The three derived features (`mechanical_load`, `thermal_stress`, `tool_wear_load_ratio`) capture cross-sensor interactions that individual raw features cannot express.

---

## Maintenance Review Queue

The model output is a **ranked risk queue** — reviewing the top-scoring assets first captures far more failures than random review:

![Review queue curve](docs/assets/review-queue-curve.png)

| Review budget | Assets reviewed | Failures captured | Yield lift vs random |
|--------------|----------------|-------------------|---------------------|
| Top 5% | 100 | 88.2% | **17.6×** |
| Top 10% | 200 | 94.1% | **9.4×** |
| Top 15% | 300 | 95.6% | **6.4×** |

Reviewing the top 10% of highest-risk assets captures 94% of holdout failures.

---

## Cost Impact (Illustrative)

From [`docs/data/ai4i-case-study/cost-simulation.json`](docs/data/ai4i-case-study/cost-simulation.json):

- Illustrative savings vs reactive maintenance: **76.47%**

The cost model is illustrative — meant to show maintenance economics framing, not audited savings.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Application | Python, Flask |
| Analytics | Pandas, NumPy, SciPy, scikit-learn |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, JavaScript |
| CI | GitHub Actions |

---

## Architecture

- **Application shell:** Flask dashboard for plant-facing KPI and maintenance context
- **Signal layer:** Synthetic telemetry and MES-style operational data
- **Analysis layer:** Reproducible AI4I benchmark, maintenance queue logic, permutation importance
- **Visuals layer:** Pre-generated PNG charts from JSON artifacts (`analysis/generate_visuals.py`)
- **Proof layer:** JSON artifacts, hiring summary, and public case-study route

---

## Local Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# (optional) generate/refresh visualization assets
python analysis/generate_visuals.py

# run full analysis benchmark (downloads UCI dataset ~500KB on first run)
python analysis/run_ai4i_case_study.py

# start the dashboard
python main.py
```

App runs on `http://127.0.0.1:8080`  |  Case-study route: `http://127.0.0.1:8080/case-study`

---

## Tests

```bash
# full test suite (23 tests: route checks + artifact contracts + metrics range)
python -m unittest discover -s tests -v

# syntax check
python -m py_compile main.py case_study.py analysis/run_ai4i_case_study.py analysis/generate_visuals.py

# import check
python -c "import main; print(main.app.name)"
```

---

## Public Proof Surfaces

- Case-study notes: [`docs/case-study.md`](docs/case-study.md)
- Hiring summary: [`docs/hiring-summary.md`](docs/hiring-summary.md)
- Analysis notes: [`analysis/README.md`](analysis/README.md)
- Metrics artifacts: [`docs/data/ai4i-case-study/`](docs/data/ai4i-case-study)

---

## What This Proves

- Predictive maintenance can be framed as an **imbalanced ranking and prioritization problem**, not only as a dashboard feature
- Model output can be connected to limited maintenance capacity through **review-budget logic and cost tradeoffs**
- Operations language and DS language can be packaged into one readable, reproducible portfolio artifact

---

## Limitations

- The live dashboard uses synthetic telemetry, not a deployed production stream
- The UCI dataset is simulated — treat as benchmark evidence, not plant-ready
- The cost model is illustrative and not audited

---

## License

MIT
