# Smart Factory App

Smart Factory App is a predictive-maintenance support case study for manufacturing teams. It combines two portfolio layers in one repo:

- a Flask dashboard for plant KPIs, machine-health context, and maintenance-oriented UI packaging
- a real predictive-maintenance benchmark on the UCI AI4I 2020 dataset with measurable review-queue logic

Demo: [Portfolio project entry](https://senanur-cetin.vercel.app/projects/smart-factory-app)

Short video demo: [`docs/assets/smart-factory-dashboard.webm`](docs/assets/smart-factory-dashboard.webm)

![Smart Factory dashboard](docs/assets/smart-factory-dashboard.png)

Portfolio role: `support case study`

## Why this project exists

Predictive-maintenance demos often stop at a generic chart or a model score. Plant teams need something more usable:

- machine context
- failure prioritization
- queue logic for limited engineering capacity
- an explanation of why the score matters operationally

This repository exists to show both the product layer and the DS layer.

## Case-study frame

### Problem

Maintenance teams cannot review every asset equally. The useful question is whether telemetry can be translated into a ranked work queue that captures most failures inside a small review budget.

### Business context

Plant analytics is only valuable if it changes maintenance prioritization. A dashboard should support scheduling and escalation decisions, not just visualize telemetry.

### Data or signal source

- Workflow layer: synthetic telemetry, OEE, RUL, event context, and shift-level plant KPIs
- DS layer: `AI4I 2020 Predictive Maintenance Dataset` from the UCI Machine Learning Repository with `10,000` records and a `3.39%` failure rate

### Methodology

- Features:
  - product type
  - air temperature
  - process temperature
  - rotational speed
  - torque
  - tool wear
- Benchmarks:
  - dummy baseline
  - logistic regression
  - random forest
  - HistGradientBoostingClassifier
- Evaluation:
  - deterministic `80/20` stratified holdout
  - ROC-AUC
  - PR-AUC
  - precision, recall, and F1
  - top-risk maintenance queue analysis
  - illustrative maintenance cost model

## Key results

From [`docs/data/ai4i-case-study/summary.json`](docs/data/ai4i-case-study/summary.json):

- Final model: `HistGradientBoostingClassifier`
- ROC-AUC: `0.9686`
- PR-AUC: `0.8522`
- Precision: `0.9074`
- Recall: `0.7206`
- F1: `0.8033`
- Queue result: reviewing the top `10%` highest-risk assets captures `92.6%` of holdout failures with `9.26x` better failure yield than random review

From [`docs/data/ai4i-case-study/cost-simulation.json`](docs/data/ai4i-case-study/cost-simulation.json):

- Illustrative savings vs reactive maintenance: `76.47%`

## What it does

- Simulates live machine telemetry and plant KPIs
- Estimates failure risk with a lightweight model layer
- Shows OEE, RUL, and cost context in one dashboard
- Publishes a real maintenance case-study route at `/case-study`
- Exposes review-budget and cost-tradeoff artifacts for interviews and portfolio review

## Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- HTML, CSS, JavaScript

## Architecture snapshot

- **Application shell:** Flask dashboard for plant-facing KPI and maintenance context
- **Signal layer:** synthetic telemetry and MES-style operational data
- **Analysis layer:** reproducible AI4I benchmark and maintenance queue logic
- **Proof layer:** JSON artifacts, hiring summary, and public case-study route
- **Deployment shape:** lightweight local or free-tier Flask demo

## Public proof surfaces

- Case-study notes: [`docs/case-study.md`](docs/case-study.md)
- Hiring summary: [`docs/hiring-summary.md`](docs/hiring-summary.md)
- Analysis notes: [`analysis/README.md`](analysis/README.md)
- Metrics artifacts: [`docs/data/ai4i-case-study`](docs/data/ai4i-case-study)
- Local route: `http://127.0.0.1:8080/case-study`

## What this proves

- You can frame predictive maintenance as an imbalanced ranking and prioritization problem, not only as a dashboard feature.
- You can connect model output to limited maintenance capacity through queue design and cost tradeoffs.
- You can package operations language and DS language into one readable portfolio surface.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python analysis/run_ai4i_case_study.py
python main.py
```

The app runs on `http://127.0.0.1:8080`.

## Quality checks

```bash
python analysis/run_ai4i_case_study.py
python -m py_compile main.py case_study.py analysis/run_ai4i_case_study.py
python -m unittest discover -s tests -v
python -c "import main; print(main.app.name)"
```

## Limitations

- The live dashboard still uses synthetic telemetry rather than a deployed production stream.
- The UCI dataset is simulated, so it should be treated as benchmark evidence rather than a plant-ready claim.
- The cost model is illustrative and meant to show maintenance economics framing, not audited savings.

## Portfolio note

This repository now sits above a generic dashboard prototype but below a production monitoring platform. The correct framing is: portfolio-grade predictive-maintenance support case study with measurable prioritization logic.

## License

MIT
