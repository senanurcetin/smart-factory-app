"""Flask blueprint for the AI4I case-study page (`/case-study`).

Reads the JSON artifacts `analysis/run_ai4i_case_study.py` writes to
`docs/data/ai4i-case-study/` and renders them into a single page — no
database, no request-time computation. If `summary.json` (the anchor
artifact) is missing, both routes return 503 rather than a stack trace.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template_string

from layout import SIDEBAR_CSS, render_sidebar

ARTIFACT_DIR = Path(__file__).resolve().parent / "docs" / "data" / "ai4i-case-study"

CASE_STUDY_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Factory Predictive Maintenance | AI4I Case Study</title>
    <link rel="icon" type="image/svg+xml" href="/static/nexus-official-x.svg">
    <meta name="description" content="Source-backed AI4I predictive maintenance model and risk-ranked holdout review queues. Offline evaluation, not deployed plant alerts.">
    <meta name="robots" content="index,follow,max-image-preview:large">
    <meta name="theme-color" content="#071829">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="NEXUS">
    <meta property="og:locale" content="en_US">
    <meta property="og:title" content="Smart Factory Predictive Maintenance | AI4I Case Study">
    <meta property="og:description" content="Source-backed AI4I predictive maintenance model and risk-ranked holdout review queues. Offline evaluation, not deployed plant alerts.">
    <meta property="og:url" content="https://smart-factory-app.onrender.com/case-study">
    <meta property="og:image" content="https://smart-factory-app.onrender.com/assets/seo/maintenance-social.jpg">
    <meta property="og:image:secure_url" content="https://smart-factory-app.onrender.com/assets/seo/maintenance-social.jpg">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:image:type" content="image/jpeg">
    <meta property="og:image:alt" content="AI4I maintenance evaluation and review queue">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Smart Factory Predictive Maintenance | AI4I Case Study">
    <meta name="twitter:description" content="Source-backed AI4I predictive maintenance model and risk-ranked holdout review queues. Offline evaluation, not deployed plant alerts.">
    <meta name="twitter:image" content="https://smart-factory-app.onrender.com/assets/seo/maintenance-social.jpg">
    <link rel="canonical" href="https://smart-factory-app.onrender.com/case-study">
    <script type="application/ld+json">{"@context":"https://schema.org","@type":"WebPage","name":"Smart Factory Predictive Maintenance | AI4I Case Study","description":"Source-backed AI4I predictive maintenance model and risk-ranked holdout review queues. Offline evaluation, not deployed plant alerts.","url":"https://smart-factory-app.onrender.com/case-study","isPartOf":{"@type":"WebSite","name":"NEXUS","url":"https://smart-factory-app.onrender.com/"},"primaryImageOfPage":{"@type":"ImageObject","url":"https://smart-factory-app.onrender.com/assets/seo/maintenance-social.jpg"}}</script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>"""
    + SIDEBAR_CSS
    + """
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
            margin-left: 250px;
            box-sizing: border-box;
        }
        @media (max-width: 900px) {
            .page { margin-left: 0; padding: 20px 16px 48px; padding-top: 4.5rem; }
        }
        .hero, .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            backdrop-filter: blur(14px);
            transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
        }
        .panel:hover {
            border-color: rgba(96, 165, 250, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 10px 24px -8px rgba(0, 0, 0, 0.35);
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
            transition: border-color 0.2s, transform 0.2s;
        }
        .metric:hover {
            border-color: rgba(94, 234, 212, 0.5);
            transform: translateY(-2px);
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
            padding: 10px 8px;
            border-bottom: 1px solid rgba(159, 180, 209, 0.18);
            text-align: left;
        }
        th {
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        tbody tr { transition: background-color 0.15s; }
        tbody tr:nth-child(even) { background: rgba(96, 165, 250, 0.05); }
        tbody tr:hover { background: rgba(94, 234, 212, 0.08); }
        .mono { font-family: 'IBM Plex Mono', monospace; }
        .chart-caption {
            display: block;
            margin-top: 8px;
            color: var(--muted);
            font-size: 0.8rem;
            text-align: center;
        }
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
        .chart-frame {
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px;
            margin-top: 16px;
        }
        .chart-frame img {
            display: block;
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .chart-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
        }
        .chart-gallery .chart-frame { margin-top: 0; }
        @media (max-width: 920px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
    <link rel="stylesheet" href="/static/nexus-design.css">
    <link rel="stylesheet" href="/static/nexus-analysis-workspace.css">
    <link rel="stylesheet" href="/static/nexus-factory-studio.css">
</head>
<body>
    {{ sidebar_html|safe }}
    <main class="page nexus-workspace nexus-maintenance-studio" id="main-content">
        <nav class="workspace-nav" aria-label="AI4I workspace chapters">
            <a href="#workspace-overview">Overview</a>
            <a href="#workspace-exploration">Signals</a>
            <a href="#workspace-benchmark">Benchmark</a>
            <a href="#workspace-validation">Validation</a>
            <a href="#workspace-queue">Maintenance queue</a>
            <a href="#workspace-drift">Drift</a>
        </nav>
        <section class="maintenance-console" id="workspace-overview" aria-label="Predictive maintenance decision workspace">
            <div class="maintenance-console-intro">
                <div class="studio-overline">NEXUS / OPERATIONS INTELLIGENCE <span>01 — MAINTENANCE STRATEGY</span></div>
                <div class="studio-research-title">CASE FILE · {{ summary.project }}</div>
                <div class="maintenance-console-index"><span class="pulse-mark" aria-hidden="true"></span> VALIDATED BENCHMARK · NOT LIVE EQUIPMENT DATA</div>
                <h1>Maintenance, <em>before failure.</em></h1>
                <p class="maintenance-console-lede">{{ summary.operational_takeaway }}</p>
                <div class="maintenance-console-caption"><span>UCI AI4I 2020</span><span>Failure prevalence: {{ "%.2f"|format((summary.dataset.failure_rate | float) * 100) }}%</span><span>Offline, reproducible evaluation</span></div>
            </div>
            <aside class="maintenance-command-score" aria-label="Selected review queue outcome">
                <span class="maintenance-score-eyebrow">PRIORITIZED REVIEW / HOLDOUT</span>
                <div class="maintenance-score-main" id="maintenance-score-main">{{ "%.1f"|format((summary.review_queue.failure_capture_rate | float) * 100) }}%</div>
                <strong>Failures caught by the selected queue</strong>
                <p id="maintenance-score-context">{{ summary.review_queue.reviewed_assets }} assets reviewed · {{ summary.review_queue.selected_budget_label }}</p>
                <div class="maintenance-score-line"></div>
                <span class="maintenance-score-foot">Measured evaluation result. Do not interpret as a live alarm or a deployed work-order system.</span>
            </aside>
        </section>
        <section class="maintenance-budget-lab" aria-labelledby="maintenance-budget-title">
            <div class="maintenance-budget-intro">
                <span>02 / REVIEW CAPACITY</span>
                <h2 id="maintenance-budget-title">Choose the maintenance review budget.</h2>
                <p>Compare three measured holdout queues. These are fixed benchmark scenarios; selecting a budget does not run a live prediction or schedule maintenance.</p>
            </div>
            <div class="maintenance-budget-options" role="group" aria-label="Review budget scenarios">
            {% for row in review_queue.review_budgets %}
                <button type="button" class="maintenance-budget-option{{ ' is-active' if row.review_fraction == summary.review_queue.selected_budget_fraction else '' }}"
                    data-queue-capture="{{ row.failure_capture_rate }}"
                    data-queue-review-count="{{ row.reviewed_assets }}"
                    data-queue-lift="{{ row.yield_lift_vs_random }}"
                    data-queue-budget="{{ (row.review_fraction * 100) | int }}"
                    aria-pressed="{{ 'true' if row.review_fraction == summary.review_queue.selected_budget_fraction else 'false' }}">
                    <span class="maintenance-budget-heading">{{ (row.review_fraction * 100) | int }}% reviewed</span>
                    <strong>{{ "%.1f"|format(row.failure_capture_rate * 100) }}%</strong>
                    <small>{{ row.captured_failures }} failures captured · {{ row.reviewed_assets }} assets</small>
                </button>
            {% endfor %}
            </div>
            <div class="maintenance-budget-summary" id="maintenance-budget-summary" aria-live="polite">Selected: top {{ (summary.review_queue.selected_budget_fraction * 100) | int }}% queue · {{ summary.review_queue.reviewed_assets }} assets reviewed · {{ "%.2f"|format(summary.review_queue.yield_lift_vs_random | float) }}× yield versus random.</div>
        </section>
        <section class="maintenance-metrics" aria-label="Predictive maintenance model evidence">
            <div class="maintenance-metric primary"><span>QUEUE YIELD VS RANDOM</span><strong>{{ summary.review_queue.yield_lift_vs_random }}×</strong><small>Observed triage leverage</small></div>
            <div class="maintenance-metric"><span>PRECISION–RECALL AUC</span><strong>{{ summary.final_model.pr_auc }}</strong><small>Imbalanced failure classification</small></div>
            <div class="maintenance-metric"><span>ROC–AUC</span><strong>{{ summary.final_model.roc_auc }}</strong><small>Holdout ranking performance</small></div>
            <div class="maintenance-metric"><span>F1 AT 0.5</span><strong>{{ summary.final_model.f1 }}</strong><small>Fixed operating threshold</small></div>
        </section>
        <div class="maintenance-chapter-title"><span>01 / TRIAGE WORKSPACE</span><h2>Put the right assets first.</h2><p>Review budget and maintenance economics are the main operational decisions; exploratory plots are supporting evidence.</p></div>

        <section class="panel maintenance-evidence-lab" id="workspace-exploration">
            <h2 class="section-title">Exploratory Data Analysis</h2>
            <div class="chart-gallery">
                <figure class="chart-frame" style="margin: 0;">
                    <img src="/assets/eda-class-balance.png" alt="Class balance: 96.6% normal vs 3.39% failure">
                    <figcaption class="chart-caption">Class balance</figcaption>
                </figure>
                <figure class="chart-frame" style="margin: 0;">
                    <img src="/assets/eda-failure-modes.png" alt="Failure mode distribution across TWF, HDF, PWF, OSF, RNF">
                    <figcaption class="chart-caption">Failure mode distribution</figcaption>
                </figure>
                <figure class="chart-frame" style="margin: 0;">
                    <img src="/assets/eda-type-distribution.png" alt="Product type distribution: Type L, M, H">
                    <figcaption class="chart-caption">Product type distribution</figcaption>
                </figure>
            </div>
            <p class="footer-note">96.6% of the 10,000 rows are normal operation — 3.39% failure rate drives the PR-AUC-first evaluation strategy used throughout this case study.</p>
        </section>

        <section class="grid maintenance-benchmark-studio" id="workspace-benchmark">
            <div class="panel">
                <h2 class="section-title">What this proves</h2>
                <ul class="bullet-list">
                    <li>The repo now has real predictive-maintenance evidence from the official UCI AI4I dataset instead of relying only on synthetic telemetry.</li>
                    <li>The useful product story is maintenance prioritization: route a small set of high-risk assets to engineers, not merely display a score on a chart.</li>
                    <li>The analysis stays free-tier friendly and reproducible: scikit-learn pipelines, JSON artifacts, and a Flask case-study route with no paid infrastructure assumptions.</li>
                </ul>
                <p class="footer-note">Dataset: <a href="{{ summary.dataset.source_url }}">{{ summary.dataset.name }}</a>. The raw CSV is downloaded on demand by the analysis script and excluded from Git.</p>
            </div>
            <div class="panel">
                <h2 class="section-title">Benchmark Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>PR-AUC</th>
                            <th>ROC-AUC</th>
                            <th>F1</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in benchmarks %}
                        <tr>
                            <td><strong>{{ row.model }}</strong></td>
                            <td class="mono">{{ row.pr_auc }}</td>
                            <td class="mono">{{ row.roc_auc }}</td>
                            <td class="mono">{{ row.f1 }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
                <p class="footer-note">{{ model_selection.selection_reason }}</p>
            </div>
        </section>

        <section class="panel maintenance-validation-studio" id="workspace-validation">
            <h2 class="section-title">Model Performance</h2>
            <div class="chart-gallery">
                <figure class="chart-frame" style="margin: 0;">
                    <img src="/assets/model-comparison.png" alt="PR-AUC, ROC-AUC and F1 comparison across the four benchmarked models">
                    <figcaption class="chart-caption">Benchmark comparison</figcaption>
                </figure>
                <figure class="chart-frame" style="margin: 0;">
                    <img src="/assets/eda-confusion-matrix.png" alt="Confusion matrix for the final tuned model on the holdout set">
                    <figcaption class="chart-caption">Confusion matrix (holdout)</figcaption>
                </figure>
            </div>
        </section>

        <section class="grid maintenance-priority-board" id="workspace-queue">
            <div class="panel">
                <h2 class="section-title">Review Capacity / Failure Capture</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Budget</th>
                            <th>Reviewed</th>
                            <th>Failures Captured</th>
                            <th>Lift</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in review_queue.review_budgets %}
                        <tr>
                            <td class="mono">{{ (row.review_fraction * 100) | int }}%</td>
                            <td class="mono">{{ row.reviewed_assets }}</td>
                            <td class="mono">{{ row.captured_failures }}</td>
                            <td class="mono">{{ row.yield_lift_vs_random }}x</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
                <figure class="chart-frame">
                    <img src="/assets/review-queue-curve.png" alt="Failure capture rate vs. review budget curve">
                    <figcaption class="chart-caption">Capture rate vs. review budget</figcaption>
                </figure>
                <p class="footer-note">At the selected `10%` queue, the model captures {{ "%.1f"|format((summary.review_queue.failure_capture_rate | float) * 100) }}% of holdout failures while reviewing only {{ summary.review_queue.reviewed_assets }} assets.</p>
            </div>
            <div class="panel">
                <h2 class="section-title">Illustrative Cost Model</h2>
                <table>
                    <tbody>
                        <tr><th>Reactive baseline</th><td class="mono">${{ cost_simulation.reactive_baseline_cost }}</td></tr>
                        <tr><th>Random review</th><td class="mono">${{ cost_simulation.random_review_cost }}</td></tr>
                        <tr><th>Risk queue</th><td class="mono">${{ cost_simulation.risk_queue_cost }}</td></tr>
                        <tr><th>Savings vs reactive</th><td class="mono">${{ cost_simulation.savings_vs_reactive }}</td></tr>
                    </tbody>
                </table>
                <figure class="chart-frame">
                    <img src="/assets/cost-model.png" alt="Per-failure-mode cost breakdown across scenarios">
                    <figcaption class="chart-caption">Cost breakdown by scenario</figcaption>
                </figure>
                <p class="footer-note">This is an illustrative planning model, not a production claim. It exists to show how a classifier can be translated into maintenance economics.</p>
            </div>
        </section>

        <section class="grid" style="margin-top: 18px;">
            <div class="panel">
                <h2 class="section-title">Failure Mode Coverage</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Failure Mode</th>
                            <th>Holdout</th>
                            <th>Captured in Queue</th>
                            <th>Capture Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in failure_modes %}
                        <tr>
                            <td>{{ row.label }}</td>
                            <td class="mono">{{ row.holdout_failures }}</td>
                            <td class="mono">{{ row.captured_in_top_10_percent_queue }}</td>
                            <td class="mono">{{ row.capture_rate }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="panel">
                <h2 class="section-title">Signal Importance</h2>
                {% set signal_rows = feature_importance.features if feature_importance is mapping else feature_importance %}
                {% set scale_max = (signal_rows[0].importance | float) if signal_rows else 0 %}
                {% if signal_rows %}
                <div class="signal-list" role="list" aria-label="Top permutation importance scores">
                    {% for row in signal_rows[:7] %}
                    {% set score = row.importance | float %}
                    {% set width_pct = (score / scale_max * 100) if scale_max > 0 and score > 0 else 0 %}
                    <div class="signal-row" role="listitem">
                        <span class="signal-name">{{ row.feature }}</span>
                        <span class="signal-track" aria-hidden="true"><span class="signal-fill" style="width: {{ width_pct | round(1) }}%"></span></span>
                        <strong class="signal-value">{{ "%.4f"|format(score) }}</strong>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <p class="footer-note">Feature attribution scores were not included in this analysis build.</p>
                {% endif %}
                <p class="footer-note">Permutation importance uses average precision as the scoring function to stay aligned with the imbalanced maintenance target. Positive scores show the measured loss of ranking quality when a feature is shuffled.</p>
            </div>
        </section>

        <section class="panel" style="margin-top: 18px;">
            <figure class="chart-frame" style="margin: 0 auto; max-width: 720px;">
                <img src="/assets/feature-importance.png" alt="Permutation importance ranking of model features">
                <figcaption class="chart-caption">Permutation importance</figcaption>
            </figure>
        </section>

        {% if drift_report %}
        <section class="panel" style="margin-top: 18px;">
            <h2 class="section-title" id="workspace-drift">Drift Check (Demonstrated)</h2>
            <p class="section-copy">{{ drift_report.scenario }}</p>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>PSI</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
                {% for row in drift_report.features %}
                    <tr>
                        <td>{{ row.feature }}</td>
                        <td class="mono">{{ "%.2f"|format(row.psi) }}</td>
                        <td class="mono">{{ row.verdict }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            <p class="footer-note">{{ drift_report.method }}. This is a real, runnable check (not just a documented idea) — regenerate with <code>python analysis/check_drift.py</code>.</p>
        </section>
        {% endif %}

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
<script>
document.querySelectorAll('.maintenance-budget-option').forEach(function(button) {
  button.addEventListener('click', function () {
    var group = document.querySelectorAll('.maintenance-budget-option');
    group.forEach(function(item) {
      var active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    var fraction = Number(button.dataset.queueCapture);
    var budget = Number(button.dataset.queueBudget);
    var count = Number(button.dataset.queueReviewCount);
    var lift = Number(button.dataset.queueLift);
    document.getElementById('maintenance-score-main').textContent = (fraction * 100).toFixed(1) + '%';
    document.getElementById('maintenance-score-context').textContent = count.toLocaleString('en-US') + ' assets reviewed · top ' + budget + '% queue';
    document.getElementById('maintenance-budget-summary').textContent = 'Selected: top ' + budget + '% queue · ' + count.toLocaleString('en-US') + ' assets reviewed · ' + lift.toFixed(2) + '× yield versus random.';
  });
});
</script>
</body>
</html>
"""
)

case_study_bp = Blueprint("case_study", __name__)


def _load_json(filename: str):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_study_artifacts() -> dict:
    summary = _load_json("summary.json")
    if summary is None:
        raise FileNotFoundError(
            "Case-study artifacts are missing. Run python analysis/run_ai4i_case_study.py."
        )
    return {
        "summary": summary,
        "benchmarks": _load_json("benchmark-comparison.json") or [],
        "model_selection": _load_json("model-selection.json") or {},
        "feature_importance": _load_json("feature-importance.json") or [],
        "review_queue": _load_json("review-queue.json") or {"review_budgets": []},
        "failure_modes": _load_json("failure-mode-breakdown.json") or [],
        "cost_simulation": _load_json("cost-simulation.json") or {},
        "dataset_profile": _load_json("dataset-profile.json") or {},
        "drift_report": _load_json("drift-report.json") or {},
    }


@case_study_bp.route("/api/case-study")
def case_study_api():
    try:
        return jsonify(load_case_study_artifacts())
    except FileNotFoundError as exc:
        abort(503, description=str(exc))


@case_study_bp.route("/case-study")
def case_study_page():
    try:
        artifacts = load_case_study_artifacts()
    except FileNotFoundError as exc:
        abort(503, description=str(exc))
    return render_template_string(
        CASE_STUDY_TEMPLATE, sidebar_html=render_sidebar("case-study"), **artifacts
    )
