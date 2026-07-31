from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template_string

ARTIFACT_DIR = Path(__file__).resolve().parent / "docs" / "data" / "cmapss-rul-case-study"

RUL_CASE_STUDY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Factory RUL Case Study</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a1120;
            --panel: rgba(15, 25, 41, 0.92);
            --panel-soft: rgba(22, 37, 61, 0.92);
            --border: #223856;
            --text: #edf4ff;
            --muted: #9fb4d1;
            --accent: #5eead4;
            --accent-2: #60a5fa;
            --danger: #fca5a5;
            --warning: #fde68a;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 30%),
                radial-gradient(circle at bottom right, rgba(94, 234, 212, 0.12), transparent 28%),
                var(--bg);
        }
        a { color: var(--accent); text-decoration: none; }
        .page {
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 20px 64px;
        }
        .hero, .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            backdrop-filter: blur(14px);
        }
        .hero {
            padding: 28px;
            display: grid;
            gap: 20px;
            margin-bottom: 22px;
        }
        .eyebrow {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        h1, h2, h3, p { margin: 0; }
        h1 { font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.05; }
        .lede {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .metrics, .grid {
            display: grid;
            gap: 16px;
        }
        .metrics { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
        .metric {
            padding: 18px;
            background: var(--panel-soft);
            border: 1px solid rgba(96, 165, 250, 0.16);
            border-radius: 16px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .metric-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.7rem;
            font-weight: 500;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 8px;
        }
        .grid {
            grid-template-columns: 1.15fr 0.85fr;
            align-items: start;
        }
        .panel { padding: 22px; }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .section-copy, .bullet-list li {
            color: var(--muted);
            line-height: 1.62;
        }
        .bullet-list {
            margin: 0;
            padding-left: 18px;
            display: grid;
            gap: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        th, td {
            padding: 10px 0;
            border-bottom: 1px solid rgba(159, 180, 209, 0.18);
            text-align: left;
        }
        th {
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .mono { font-family: 'IBM Plex Mono', monospace; }
        .footer-note {
            margin-top: 18px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .tag {
            display: inline-block;
            padding: 5px 10px;
            border: 1px solid rgba(94, 234, 212, 0.24);
            border-radius: 999px;
            color: var(--accent);
            font-size: 0.78rem;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        @media (max-width: 920px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <div>
                <div class="eyebrow">Second Case Study — Genuine RUL Regression</div>
                <h1>{{ summary.project }}</h1>
            </div>
            <p class="lede">{{ summary.operational_takeaway }}</p>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Test RMSE</div>
                    <div class="metric-value">{{ summary.final_model.rmse }} cycles</div>
                    <div class="metric-note">On NASA's official held-out test protocol — one row per engine, its final observed cycle.</div>
                </div>
                <div class="metric">
                    <div class="metric-label">PHM08 Score</div>
                    <div class="metric-value">{{ summary.final_model.phm08_score }}</div>
                    <div class="metric-note">Asymmetric scoring — over-optimistic (late) predictions penalized harder than conservative ones.</div>
                </div>
                <div class="metric">
                    <div class="metric-label">RMSE Lift vs Baseline</div>
                    <div class="metric-value">{{ summary.lift_vs_baseline.rmse_reduction_pct }}%</div>
                    <div class="metric-note">vs. always predicting the median training RUL ({{ summary.baseline.rmse }} cycles RMSE).</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Dataset</div>
                    <div class="metric-value">{{ summary.dataset.train_units }} / {{ summary.dataset.test_units }}</div>
                    <div class="metric-note">Train / test engines, {{ summary.dataset.subset }}.</div>
                </div>
            </div>
        </section>

        <section class="grid">
            <div class="panel">
                <h2 class="section-title">Why this exists alongside the AI4I case study</h2>
                <ul class="bullet-list">
                    <li>AI4I 2020 is a static snapshot dataset — one row per independently sampled machine, no time axis — so genuine remaining-useful-life regression isn't possible there. C-MAPSS is a real run-to-failure dataset, which is.</li>
                    <li>Because engines are tracked over many cycles, rows can be grouped by engine for cross-validation — the grouped/time-aware split AI4I's structure could not support is used here.</li>
                    <li>Evaluation uses the dataset's own designed protocol (NASA's official test split + truthed RUL labels), not a custom holdout.</li>
                </ul>
                <p class="footer-note">Dataset: {{ summary.dataset.name }}. {{ summary.dataset.reference }}.</p>
            </div>
            <div class="panel">
                <h2 class="section-title">Model vs. Naive Baseline</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Approach</th>
                            <th>RMSE</th>
                            <th>PHM08 Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{{ baseline.name }}</td>
                            <td class="mono">{{ baseline.rmse }}</td>
                            <td class="mono">{{ baseline.phm08_score }}</td>
                        </tr>
                        <tr>
                            <td><strong>{{ final_model.name }}</strong></td>
                            <td class="mono">{{ final_model.rmse }}</td>
                            <td class="mono">{{ final_model.phm08_score }}</td>
                        </tr>
                    </tbody>
                </table>
                <p class="footer-note">{{ model_selection.selection_reason }}</p>
            </div>
        </section>

        <section class="grid" style="margin-top: 18px;">
            <div class="panel">
                <h2 class="section-title">Hyperparameter Tuning</h2>
                <table>
                    <tbody>
                        <tr><th>Search method</th><td class="mono">{{ tuning.search_method }} ({{ tuning.n_iter }} iter)</td></tr>
                        <tr><th>CV strategy</th><td>{{ tuning.cv_strategy }}</td></tr>
                        <tr><th>Scoring</th><td class="mono">{{ tuning.scoring }}</td></tr>
                        <tr><th>Best CV RMSE</th><td class="mono">{{ tuning.best_cv_rmse }}</td></tr>
                    </tbody>
                </table>
                <p class="footer-note">Best params: {% for key, value in tuning.best_params.items() %}<span class="tag">{{ key }}={{ value }}</span>{% endfor %}</p>
            </div>
            <div class="panel">
                <h2 class="section-title">Rolling-Window Feature Impact</h2>
                <table>
                    <thead>
                        <tr><th>Feature set</th><th>RMSE</th><th>PHM08 Score</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Raw sensors only</td><td class="mono">{{ fe_impact.raw_features.rmse }}</td><td class="mono">{{ fe_impact.raw_features.phm08_score }}</td></tr>
                        <tr><td>+ Rolling window (mean/std)</td><td class="mono">{{ fe_impact.enhanced_features.rmse }}</td><td class="mono">{{ fe_impact.enhanced_features.phm08_score }}</td></tr>
                    </tbody>
                </table>
                <p class="footer-note">{{ fe_impact.note }}</p>
            </div>
        </section>

        <section class="grid" style="margin-top: 18px;">
            <div class="panel">
                <h2 class="section-title">Signal Importance (SHAP)</h2>
                {% for row in feature_importance %}
                    <span class="tag">{{ row.feature }}: {{ row.mean_abs_shap }}</span>
                {% endfor %}
                <p class="footer-note">{{ feature_importance_method }}</p>
            </div>
            <div class="panel">
                <h2 class="section-title">Sensors</h2>
                <table>
                    <tbody>
                        <tr><th>Total sensors</th><td class="mono">{{ summary.sensors.total }}</td></tr>
                        <tr><th>Kept (informative)</th><td class="mono">{{ summary.sensors.kept_informative }}</td></tr>
                        <tr><th>Dropped (near-constant)</th><td>{{ summary.sensors.dropped_near_constant | join(', ') }}</td></tr>
                    </tbody>
                </table>
                <p class="footer-note">{{ summary.target.definition }}</p>
            </div>
        </section>

        <section class="panel" style="margin-top: 18px;">
            <h2 class="section-title">Limitations and next step</h2>
            <ul class="bullet-list">
                {% for item in summary.limitations %}
                    <li>{{ item }}</li>
                {% endfor %}
            </ul>
            <p class="footer-note"><span style="color: var(--warning);">Next step:</span> {{ model_selection.next_step }}</p>
        </section>
    </main>
</body>
</html>
"""

rul_case_study_bp = Blueprint("rul_case_study", __name__)


def _load_json(filename: str):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_rul_case_study_artifacts() -> dict:
    summary = _load_json("summary.json")
    if summary is None:
        raise FileNotFoundError(
            "RUL case-study artifacts are missing. Run "
            "python analysis/run_cmapss_rul_case_study.py."
        )
    model_selection = _load_json("model-selection.json") or {}
    feature_importance_payload = _load_json("feature-importance.json") or {}
    return {
        "summary": summary,
        "model_selection": model_selection,
        "baseline": summary.get("baseline", {}),
        "final_model": summary.get("final_model", {}),
        "tuning": model_selection.get("hyperparameter_tuning", {}),
        "fe_impact": model_selection.get("feature_engineering_impact", {}),
        "feature_importance": feature_importance_payload.get("features", [])[:15],
        "feature_importance_method": feature_importance_payload.get("method", ""),
    }


@rul_case_study_bp.route("/api/rul-case-study")
def rul_case_study_api():
    try:
        return jsonify(load_rul_case_study_artifacts())
    except FileNotFoundError as exc:
        abort(503, description=str(exc))


@rul_case_study_bp.route("/rul-case-study")
def rul_case_study_page():
    try:
        artifacts = load_rul_case_study_artifacts()
    except FileNotFoundError as exc:
        abort(503, description=str(exc))
    return render_template_string(RUL_CASE_STUDY_TEMPLATE, **artifacts)
