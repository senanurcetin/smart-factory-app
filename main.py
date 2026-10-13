"""
A single-file Flask application with a Modern "Bento Grid" Dashboard.

This version specifically addresses layout and scaling issues, ensuring all cards
and charts render correctly without deformation, while maintaining the existing
data simulation and backend logic.
"""

import csv
import io
import os
import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, send_from_directory

from analysis.run_ai4i_case_study import (
    MODEL_FEATURES_ENHANCED,
    MODEL_PATH,
    add_derived_features,
)
from case_study import case_study_bp
from demo_state import advance_production_counts, synchronized_demo_view
from layout import SIDEBAR_CSS, render_sidebar
from rul_case_study import rul_case_study_bp
from settings import settings_bp

ASSETS_DIR = Path(__file__).resolve().parent / "docs" / "assets"

# =============================================================================
# 1. BACKEND: APPLICATION & LIVE RISK SCORING
# =============================================================================
#
# The dashboard scores live (simulated) telemetry against the same
# HistGradientBoosting pipeline that is benchmarked in the offline case study
# (see analysis/run_ai4i_case_study.py). Feature names/engineering are
# imported from that module so the dashboard can never drift out of sync
# with what the model was actually trained on.

app = Flask(__name__)
app.register_blueprint(case_study_bp)
app.register_blueprint(rul_case_study_bp)
app.register_blueprint(settings_bp)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run `python analysis/run_ai4i_case_study.py` first to train and persist it."
    )
risk_model = joblib.load(MODEL_PATH)

# Approximate AI4I 2020 feature ranges/distribution, used to simulate plausible
# live sensor readings that the real model can score meaningfully.
MACHINE_TYPES = ["L", "M", "H"]
MACHINE_TYPE_WEIGHTS = [0.6, 0.3, 0.1]
MAX_RUL_HOURS = 48

historical_data = deque(maxlen=240)
event_log = deque(maxlen=10)
production_actual = 0
scrap_count = 0
downtime_seconds = 0.0
tool_wear_minutes = 0.0
app_start_time = time.time()
IDEAL_CYCLE_TIME_SECONDS = 12.0  # design-spec cycle time used for the OEE Performance component

# =============================================================================
# 2. FRONTEND: BENTO GRID TEMPLATE (Layout & Style FIXES)
# =============================================================================

HTML_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Factory | Industrial Analytics Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>
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
            --accent: #8F9BFF;
            --accent-2: #4EA8FF;
        }

        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 30%),
                radial-gradient(circle at bottom right, rgba(143, 155, 255, 0.12), transparent 28%),
                var(--bg);
        }
        .text-muted { color: var(--muted) !important; }

        .main-content {
            margin-left: 250px;
            padding: 2rem;
        }
        @media (max-width: 900px) {
            .main-content { margin-left: 0; padding: 1.25rem; padding-top: 4.5rem; }
        }

        .cold-start-overlay {
            position: fixed;
            inset: 0;
            z-index: 40;
            background-color: var(--bg);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
            text-align: center;
            padding: 2rem;
            transition: opacity 0.3s ease;
        }
        .cold-start-overlay.hidden {
            opacity: 0;
            pointer-events: none;
        }
        .cold-start-overlay p {
            margin: 0;
            font-weight: 600;
            color: var(--text);
        }
        .cold-start-overlay .cold-start-hint {
            font-weight: 400;
            font-size: 0.85rem;
            color: var(--muted);
            max-width: 340px;
        }
        .spinner {
            width: 38px;
            height: 38px;
            border: 4px solid var(--border);
            border-top-color: var(--accent-2);
            border-radius: 50%;
            animation: sf-spin 0.8s linear infinite;
        }
        @keyframes sf-spin {
            to { transform: rotate(360deg); }
        }

        .bento-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            backdrop-filter: blur(14px);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            height: 100%;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .bento-card:hover {
            transform: translateY(-2px);
            border-color: rgba(96, 165, 250, 0.4);
            box-shadow: 0 10px 24px -8px rgba(0, 0, 0, 0.45);
        }

        .kpi-card {
            height: 130px; /* Fixed height for KPI cards */
        }

        .kpi-card-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--muted);
        }
        .kpi-card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text);
            margin: auto 0;
            line-height: 1.2;
        }
        .kpi-card-caption {
            font-size: 0.72rem;
            font-weight: 600;
            align-self: flex-start;
            letter-spacing: 0.02em;
        }
        .kpi-card-caption.is-model { color: var(--accent-2); }
        .kpi-card-caption.is-sim { color: var(--muted); }

        .risk-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-left: 8px;
            vertical-align: middle;
        }
        .risk-badge.risk-low { background: #dcfce7; color: #166534; }
        .risk-badge.risk-medium { background: #fef9c3; color: #854d0e; }
        .risk-badge.risk-high { background: #fee2e2; color: #991b1b; }

        .dashboard-intro {
            background: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-left: 4px solid var(--accent-2);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
        }
        .dashboard-intro i {
            color: var(--accent-2);
            font-size: 1.2rem;
            margin-top: 0.15rem;
        }
        .dashboard-intro p {
            margin: 0;
            font-size: 0.92rem;
            color: var(--muted);
            line-height: 1.55;
        }
        .dashboard-intro a { color: var(--accent-2); font-weight: 600; }

        .page-header { margin-bottom: 1.25rem; }
        .page-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 4px 0;
        }
        .page-subtitle {
            font-size: 0.92rem;
            color: var(--muted);
            margin: 0;
        }

        .chart-card {
            height: 400px;
            min-height: 0;
        }

        /* Flex allocates remaining card space after the visible chart headings. */
        .chart-container {
            position: relative;
            flex: 1 1 auto;
            min-height: 0;
            width: 100%;
        }
        .chart-container canvas { display: block; }

        .detail-card {
            height: 300px; /* Fixed height for detail cards */
            overflow: hidden;
        }

        .log-box {
            background-color: var(--panel-soft);
            color: var(--muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding: 1rem;
            border-radius: 12px;
            height: 100%;
            overflow-y: auto;
        }

        .info-list-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }
        .info-list-item:last-child { border-bottom: none; }
        .info-list-item .label { color: var(--muted); }
        .info-list-item .value { font-weight: 500; font-family: 'JetBrains Mono', monospace; color: var(--text); }

        .scrollable-content {
            overflow-y: auto;
        }

        .progress { background-color: var(--panel-soft) !important; }
        .report-actions { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin-top:14px; }
        .report-actions .btn { border-color: #4EA8FF; color: #eaf1fa; font-size: 0.85rem; font-weight: 600; }
        .report-actions .btn:hover, .report-actions .btn:focus-visible { background: #214f7f; color: #ffffff; }
        .source-label { display:inline-flex; align-items:center; gap:7px; color:#cbd9ed; font-size:0.78rem; font-weight:700; letter-spacing:0.04em; }
        .source-label::before { content:""; width:8px; height:8px; border-radius:50%; background:#8F9BFF; }
        .metric-breakdown { display:grid; gap:18px; padding-top:12px; }
        .metric-breakdown .metric-row { display:flex; justify-content:space-between; margin-bottom:7px; color:var(--text); font-size:0.85rem; }
        .metric-breakdown .progress { height:9px; }
        .metric-breakdown .progress-bar { background:#4EA8FF; }
        .metric-breakdown .metric:nth-child(2) .progress-bar { background:#8F9BFF; }
        .metric-breakdown .metric:nth-child(3) .progress-bar { background:#F6B95F; }
        .evidence-caption { font-size:0.8rem; line-height:1.5; color:var(--muted); margin-top:auto; padding-top:12px; }
        @media(max-width:575px) {
            .kpi-card { min-height:150px; height:auto; }
            .kpi-card-value { font-size:1.7rem; }
            .bento-card { padding:1rem; }
            .chart-card { height:360px; }
        }

    </style>
    <link rel="stylesheet" href="/static/nexus-design.css">
</head>
<body>
    {{ sidebar_html|safe }}

    <div class="cold-start-overlay" id="cold-start-overlay" role="status" aria-live="polite">
        <div class="spinner" aria-hidden="true"></div>
        <p>Waking up the simulation&hellip;</p>
        <p class="cold-start-hint">This demo runs on a free-tier server that sleeps when idle — first load can take up to a minute.</p>
    </div>

    <main class="main-content" id="main-content">
        <div class="page-header">
            <div class="nexus-eyebrow">NEXUS / INDUSTRIAL INTELLIGENCE <span>01 — OPERATIONS</span></div>
            <h1 class="page-title">Industrial Analytics Dashboard</h1>
            <p class="page-subtitle">AI4I-simulated telemetry · trained model risk scoring · transparent KPI definitions</p>
            <div class="nexus-hero-chips" aria-label="Data classification"><span>SIMULATED SENSOR STREAM</span><span>TRAINED ML CLASSIFIER</span><span>NOT REAL PLANT DATA</span></div>
            <div class="report-actions"><span class="nexus-stream-badge" id="nexus-stream-badge" data-state="connecting" role="status" aria-live="polite"><span class="nexus-stream-dot" aria-hidden="true"></span><span id="nexus-stream-label">CONNECTING TO DEMO STREAM</span></span><span class="source-label">SIMULATED PLANT DATA + TRAINED ML MODEL · SHARED DEMO STREAM</span><a class="btn btn-outline-primary" href="/api/report.csv" download="smart-factory-demo-telemetry.csv"><i class="fas fa-file-csv me-2" aria-hidden="true"></i>Export shared demo readings (CSV)</a><span id="telemetry-clock" class="text-muted small" aria-live="polite">Awaiting a simulated reading</span></div>
        </div>
        <div class="dashboard-intro">
            <i class="fas fa-circle-info"></i>
            <p>
                <strong>What you're looking at:</strong> synthetic AI4I-schema sensor readings are generated every 2 seconds to imitate a running machine, then scored live by the same trained pipeline validated in the
                <a href="/case-study">Case Study</a> (PR-AUC 0.90 on held-out data). The <strong>Failure Risk Score</strong> is computed by the trained classifier on synthetic sensors; the <strong>RUL Proxy</strong> is a risk-derived heuristic, <em>not</em> a calibrated remaining-life regression. OEE, production and costs use simulated plant counters. For actual evaluated regression, open the RUL Case Study.
            </p>
        </div>
        <div class="nexus-section-label">01 / LIVE OPERATIONS OVERVIEW</div>
        <div class="row g-4">
            <!-- KPI ROW with fixed height cards -->
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">Failure Risk Score <span class="risk-badge" id="kpi-risk-badge">--</span></div><div class="kpi-card-value" id="kpi-risk">—</div><div class="kpi-card-caption is-model"><i class="fas fa-robot"></i> Live model output</div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">RUL Proxy (heuristic)</div><div class="kpi-card-value" id="kpi-rul" title="Heuristic estimate derived from the model's risk score — not a calibrated survival/RUL model. See the RUL Case Study for genuine RUL regression.">—</div><div class="kpi-card-caption is-model"><i class="fas fa-robot"></i> Derived from risk score</div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">OEE</div><div class="kpi-card-value" id="kpi-oee">—</div><div class="kpi-card-caption is-sim"><i class="fas fa-shuffle"></i> Simulated telemetry</div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">Net Production</div><div class="kpi-card-value" id="kpi-prod">—</div><div class="kpi-card-caption is-sim"><i class="fas fa-shuffle"></i> Simulated telemetry</div></div></div>

            <!-- CHART ROW with fixed height cards and containers -->
            <div class="col-lg-6"><div class="bento-card chart-card"><h5>Real-time Temperature Analysis <span class="nexus-data-chip">SIMULATED INPUT · °C</span></h5><p class="small text-muted mb-2">Process temperature (model input)</p><div class="chart-container"><canvas id="tempChart" role="img" aria-label="Temperature over simulated time"></canvas></div></div></div>
            <div class="col-lg-6"><div class="bento-card chart-card"><h5>Real-time Tool Wear Analysis <span class="nexus-data-chip">SIMULATED INPUT · MIN</span></h5><p class="small text-muted mb-2">Tool wear accumulation (model input)</p><div class="chart-container"><canvas id="vibrationChart" role="img" aria-label="Tool wear over simulated time"></canvas></div></div></div>

            <!-- Model outputs and OEE calculation are bound to the same API stream -->
            <div class="col-lg-8"><div class="bento-card chart-card"><h5>Model Risk Score Trend <span class="nexus-data-chip nexus-model-chip">MODEL OUTPUT · 0–100</span></h5><p class="small text-muted mb-2">Trained AI4I classifier score on synthetic sensors. Higher means more model-assessed risk; not a calibrated failure probability.</p><div class="chart-container"><canvas id="riskChart" aria-label="Model risk score trend over simulated timestamps" role="img"></canvas></div></div></div>
            <div class="col-lg-4"><div class="bento-card chart-card"><h5>OEE Decomposition</h5><p class="small text-muted">Availability × Performance × Quality</p><div class="metric-breakdown">
              <div class="metric"><div class="metric-row"><span>Availability</span><strong id="oee-availability">—</strong></div><div class="progress"><div class="progress-bar" id="oee-availability-bar" style="width:0%"></div></div></div>
              <div class="metric"><div class="metric-row"><span>Performance</span><strong id="oee-performance">—</strong></div><div class="progress"><div class="progress-bar" id="oee-performance-bar" style="width:0%"></div></div></div>
              <div class="metric"><div class="metric-row"><span>Quality</span><strong id="oee-quality">—</strong></div><div class="progress"><div class="progress-bar" id="oee-quality-bar" style="width:0%"></div></div></div>
            </div><p class="evidence-caption">Calculated from simulated production/downtime/scrap. Demonstrates the standard OEE formula; does not describe a live factory.</p></div></div>
            <!-- DETAIL ROW with fixed height cards -->
            <div class="col-lg-4"><div class="bento-card detail-card scrollable-content"><h6 class="mb-3">Shift Information</h6><div id="shift-info-panel"></div></div></div>
            <div class="col-lg-4"><div class="bento-card detail-card"><h6 class="mb-3">System Health</h6><div id="health-panel" class="mt-4"></div></div></div>
            <div class="col-lg-4"><div class="bento-card detail-card p-0"><div class="log-box" id="log-box"></div></div></div>
        </div>
    </main>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const chartDefaults = {
            responsive: true,
            maintainAspectRatio: false, // CRITICAL: Allows chart to fill container height
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#091c31',
                    titleColor: '#f1f7ff',
                    bodyColor: '#c3d9f1',
                    borderColor: 'rgba(78, 168, 255, 0.42)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false
                }
            },
            interaction: { intersect: false, mode: 'index' },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#b5c9e1', maxTicksLimit: 7, maxRotation: 0 } },
                y: { grid: { color: 'rgba(135, 179, 224, 0.13)' }, ticks: { color: '#b5c9e1', maxTicksLimit: 6 } }
            }
        };
        const tempChart = new Chart(document.getElementById('tempChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                data: [],
                borderColor: '#4EA8FF',
                tension: 0.4,           // Makes lines curved
                borderWidth: 2,         // Sets line thickness
                pointRadius: 0,
                fill: true,
                backgroundColor: 'rgba(78, 168, 255, 0.07)'
            }] },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { callback: value => value + '°C' } } } }
        });
        const vibrationChart = new Chart(document.getElementById('vibrationChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                data: [],
                borderColor: '#8F9BFF',
                tension: 0.4,           // Makes lines curved
                borderWidth: 2,         // Sets line thickness
                pointRadius: 0,
                fill: true,
                backgroundColor: 'rgba(143, 155, 255, 0.07)'
            }] },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { callback: value => value + ' min' } } } }
        });

        const riskChart = new Chart(document.getElementById('riskChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Model risk score (%)', data: [],
                borderColor: '#F6B95F', borderWidth: 2.5, pointRadius: 0,
                tension: 0.25, fill: true, backgroundColor: 'rgba(246,185,95,0.10)'
            }] },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales,
                y: { ...chartDefaults.scales.y, min: 0, max: 100,
                    ticks: { color: '#9fb4d1', callback: value => value + '%' } }
            } }
        });

        // Text produced by the simulated API is data, not executable markup.
        // Never insert operator names or event descriptions as raw HTML.
        const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[ch]));

        async function fetchData() {
            const response = await fetch('/api/data', { cache: 'no-store' });
            if (!response.ok) throw new Error('Telemetry HTTP ' + response.status);
            const data = await response.json();
            if (!data || !data.current_reading || !data.production_stats ||
                !Array.isArray(data.event_log)) {
                throw new Error('Incomplete telemetry payload');
            }
            const current = data.current_reading;
            document.getElementById('telemetry-clock').textContent =
                'Last model-scored sample: ' + current.RecordedAtUTC + ' (UTC)';
            const prod = data.production_stats;
            const events = data.event_log;

            // --- Update KPIs ---
            document.getElementById('kpi-oee').textContent = prod.oee.toFixed(1) + '%';
            ['availability', 'performance', 'quality'].forEach(function (key) {
                const pct = Math.max(0, Math.min(100, Number(prod[key]) || 0));
                document.getElementById('oee-' + key).textContent = pct.toFixed(1) + '%';
                document.getElementById('oee-' + key + '-bar').style.width = pct + '%';
            });
            document.getElementById('kpi-prod').textContent = prod.actual;
            document.getElementById('kpi-rul').textContent = `${current.RUL} Hr`;

            const riskPct = current.RiskScore * 100;
            document.getElementById('kpi-risk').textContent = riskPct.toFixed(1) + '%';
            const riskBadge = document.getElementById('kpi-risk-badge');
            if (riskPct < 30) {
                riskBadge.textContent = 'LOW';
                riskBadge.className = 'risk-badge risk-low';
            } else if (riskPct < 60) {
                riskBadge.textContent = 'MEDIUM';
                riskBadge.className = 'risk-badge risk-medium';
            } else {
                riskBadge.textContent = 'HIGH';
                riskBadge.className = 'risk-badge risk-high';
            }

            // --- Update Shift Info Panel ---
            const shiftPanel = document.getElementById('shift-info-panel');
            shiftPanel.innerHTML = `
                <div class="info-list-item"><span class="label">Shift ID</span><span class="value">${escapeHtml(current.Shift_ID)}</span></div>
                <div class="info-list-item"><span class="label">Operator</span><span class="value">${escapeHtml(current.Operator)}</span></div>
                <div class="info-list-item"><span class="label">Cycle Time</span><span class="value">${current.Cycle_Time.toFixed(1)}s</span></div>
                <div class="info-list-item"><span class="label">Power Factor</span><span class="value">${current.Power_Factor.toFixed(2)}</span></div>
                <div class="info-list-item"><span class="label">Machine Type</span><span class="value">${escapeHtml(current.Type)}</span></div>
                <div class="info-list-item"><span class="label">Energy Cost</span><span class="value">$${current.Energy_Cost_Per_Hour.toFixed(2)}/hr</span></div>
            `;

            // --- Update Health Progress Bars with conditional colors ---
            const healthPanel = document.getElementById('health-panel');
            const spindleLoad = current.Spindle_Load;
            const pressure = current.Hydraulic_Pressure;
            let spindleColor = spindleLoad > 90 ? 'bg-danger' : spindleLoad > 75 ? 'bg-warning' : 'bg-success';
            let pressureColor = pressure > 140 ? 'bg-danger' : pressure > 120 ? 'bg-warning' : 'bg-primary';

            healthPanel.innerHTML = `
                <div class="mb-4">
                    <div class="d-flex justify-content-between small text-muted"><span>Spindle Load</span><span>${spindleLoad.toFixed(1)}%</span></div>
                    <div class="progress" style="height:10px;"><div class="progress-bar ${spindleColor}" style="width: ${spindleLoad}%;"></div></div>
                </div>
                <div>
                    <div class="d-flex justify-content-between small text-muted"><span>Hydraulic Pressure</span><span>${pressure.toFixed(1)} Bar</span></div>
                    <div class="progress" style="height:10px;"><div class="progress-bar ${pressureColor}" style="width: ${pressure/150*100}%;"></div></div>
                </div>
            `;

            // --- Update Charts ---
            const label = current.Timestamp;
            [tempChart, vibrationChart, riskChart].forEach(chart => {
                if (chart.data.labels.length >= 30) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
                chart.data.labels.push(label);
            });
            tempChart.data.datasets[0].data.push(current.Temperature);
            vibrationChart.data.datasets[0].data.push(current.Tool_Wear);
            riskChart.data.datasets[0].data.push(current.RiskScore * 100);
            tempChart.update();
            vibrationChart.update();
            riskChart.update();

            // --- Update Log Box ---
            const logBox = document.getElementById('log-box');
            logBox.replaceChildren();
            events.forEach(event => {
                const line = document.createElement('p');
                const when = document.createElement('span');
                when.className = 'text-muted';
                when.textContent = '[' + String(event.timestamp ?? '') + ']';
                line.append(when, document.createTextNode(' ' + String(event.message ?? '')));
                logBox.appendChild(line);
            });
            logBox.scrollTop = logBox.scrollHeight;
        }

        function hideColdStartOverlay() {
            const overlay = document.getElementById('cold-start-overlay');
            if (overlay) overlay.classList.add('hidden');
        }

        // Reflect transport health separately from the model failure-risk score:
        // cached measurements are not proof that the stream is still connected.
        function setStreamState(state, text) {
            const badge = document.getElementById('nexus-stream-badge');
            const label = document.getElementById('nexus-stream-label');
            if (badge) badge.dataset.state = state;
            if (label) label.textContent = text;
        }
        // Slow free-tier responses cannot race and overwrite newer readings.
        let dashboardRefreshing = false;
        async function refreshDashboard() {
            if (dashboardRefreshing) return;
            dashboardRefreshing = true;
            try {
                await fetchData();
                setStreamState('active', 'SIMULATED STREAM CONNECTED');
            } catch (_error) {
                setStreamState('unavailable', 'STREAM UNAVAILABLE · RETRYING');
                document.getElementById('telemetry-clock').textContent =
                    'Telemetry temporarily unavailable — retrying automatically (prior values may be stale)';
            } finally {
                dashboardRefreshing = false;
                hideColdStartOverlay();
            }
        }

        const refreshMs = parseInt(localStorage.getItem('sf_refresh_ms'), 10) || 2000;
        setInterval(refreshDashboard, refreshMs);
        document.addEventListener('DOMContentLoaded', refreshDashboard);
    </script>
</body>
</html>
"""
)

# =============================================================================
# 3. BACKEND API & ROUTES (UNCHANGED)
# =============================================================================


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, sidebar_html=render_sidebar("dashboard"))


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.route("/api/data")
@synchronized_demo_view
def get_data():
    global production_actual, scrap_count, downtime_seconds, tool_wear_minutes

    # Simulate Production & MES Data; rejects require a completed unit.
    production_actual, scrap_count = advance_production_counts(
        production_actual, scrap_count, random.randint(0, 3), random.random() < 0.02
    )
    if random.random() < 0.05:
        downtime_seconds += random.uniform(1.0, 4.0)  # simulated micro-stoppage (jam, adjustment)

    # OEE = Availability x Performance x Quality (real formula, not a random number)
    elapsed_seconds = max(time.time() - app_start_time, 1.0)
    run_time_seconds = max(elapsed_seconds - downtime_seconds, 1.0)
    availability = run_time_seconds / elapsed_seconds
    performance = min((IDEAL_CYCLE_TIME_SECONDS * production_actual) / run_time_seconds, 1.0)
    good_count = max(production_actual - scrap_count, 0)
    quality = good_count / production_actual if production_actual else 1.0
    oee = availability * performance * quality

    production_stats = {
        "target": 1000,
        "actual": production_actual,
        "scrap": scrap_count,
        "availability": round(availability * 100, 2),
        "performance": round(performance * 100, 2),
        "quality": round(quality * 100, 2),
        "oee": round(oee * 100, 2),
    }

    # ── Simulate AI4I-schema telemetry (the actual model inputs) ─────────────
    # Tool wear accumulates across ticks and resets when a preventive
    # maintenance / tool-change event fires, so RiskScore genuinely tracks
    # a degrading-then-reset wear cycle instead of pure IID noise.
    tool_wear_minutes += random.uniform(0.5, 2.5)
    if tool_wear_minutes > 230 and random.random() < 0.15:
        tool_wear_minutes = 0.0
        event_log.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": "INFO: Tool replaced during preventive maintenance",
            }
        )

    machine_type = random.choices(MACHINE_TYPES, weights=MACHINE_TYPE_WEIGHTS, k=1)[0]
    air_temp_k = random.uniform(295.0, 304.0)
    process_temp_k = air_temp_k + random.uniform(8.0, 12.0)
    rotational_speed_rpm = random.uniform(1168.0, 2886.0)
    torque_nm = random.uniform(3.0, 77.0)

    raw_row = pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temp_k,
                "Process temperature [K]": process_temp_k,
                "Rotational speed [rpm]": rotational_speed_rpm,
                "Torque [Nm]": torque_nm,
                "Tool wear [min]": tool_wear_minutes,
            }
        ]
    )
    feature_row = add_derived_features(raw_row)[MODEL_FEATURES_ENHANCED]
    risk_score = float(risk_model.predict_proba(feature_row)[:, 1][0])

    # Heuristic RUL derived from the real model's risk score (linear decay from
    # MAX_RUL_HOURS at risk=0 to 0 at risk=1). This is NOT a calibrated
    # survival/RUL regression — it is an illustrative, monotonic proxy meant
    # to be honest about what it is while still being driven by the real model.
    rul_hours = max(1, round(MAX_RUL_HOURS * (1 - risk_score)))

    # Simulate Enriched Sensor & Metadata (plant-context flavor telemetry,
    # not fed to the risk model)
    current_val = random.uniform(3.0, 4.9)
    voltage_val = random.uniform(218.0, 223.0)

    reading = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "RecordedAtUTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Type": machine_type,
        "Temperature": round(process_temp_k - 273.15, 1),  # process temp, Kelvin to C
        "Tool_Wear": round(tool_wear_minutes, 1),
        "Torque": round(torque_nm, 1),
        "Rotational_Speed": round(rotational_speed_rpm, 0),
        "Current": round(current_val, 2),
        "Pressure": round(random.uniform(30.0, 42.0), 1),
        "Energy": round(random.uniform(12.0, 17.5), 2),
        "Spindle_Load": round(random.uniform(60.0, 95.0), 1),
        "Voltage": round(voltage_val, 1),
        "Hydraulic_Pressure": round(random.uniform(115.0, 148.0), 1),
        "Power_Factor": round(random.uniform(0.88, 0.99), 2),
        "Cycle_Time": round(random.uniform(11.5, 14.2), 1),
        "Shift_ID": "SHIFT-A" if 8 <= datetime.now().hour < 20 else "SHIFT-B",
        "Operator": "John Doe" if 8 <= datetime.now().hour < 20 else "Jane Smith",
        "Energy_Cost_Per_Hour": (current_val * voltage_val * 0.12) / 1000,
        "RUL": rul_hours,
        "RiskScore": round(risk_score, 4),
    }

    historical_data.append(reading)

    # Simulate System Events
    if random.random() < 0.2:
        event_log.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": random.choice(
                    ["INFO: Cycle OK", "WARN: Temp high", "INFO: Pressure nominal"]
                ),
            }
        )

    return jsonify(
        {
            "current_reading": reading,
            "production_stats": production_stats,
            "event_log": list(event_log),
        }
    )


@app.route("/api/report.csv")
@synchronized_demo_view
def export_shared_demo_report():
    """Export actual API-generated demo readings and trained classifier scores.

    The input sensor readings are synthetic AI4I-like signals; this is a
    process-local shared demo stream, not a private visitor session or real
    factory telemetry. Readings are not mutated, and different workers can
    have different windows.
    """
    stream = io.StringIO()
    writer = csv.writer(stream)
    columns = [
        ("RecordedAtUTC", "Recorded at (UTC, ISO 8601)"),
        ("Timestamp", "Timestamp (server local time of day)"),
        ("Type", "Machine type"),
        ("Temperature", "Process temperature (C)"),
        ("Tool_Wear", "Tool wear (min)"),
        ("Torque", "Torque (Nm)"),
        ("Rotational_Speed", "Rotational speed (rpm)"),
        ("RiskScore", "Model risk score (0-1)"),
        ("RUL", "Heuristic RUL proxy (hours)"),
    ]
    writer.writerow([header for _, header in columns])
    for reading in list(historical_data):
        writer.writerow([reading.get(key, "") for key, _ in columns])
    return Response(
        stream.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=smart-factory-demo-telemetry.csv",
            "X-Data-Provenance": "simulated-inputs;trained-model-scores",
            "X-Data-Scope": "shared-worker-buffer;last-240-readings",
            "Cache-Control": "no-store",
        },
    )


# =============================================================================
# 4. APPLICATION RUNNER (UNCHANGED)
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
