"""
A single-file Flask application with a Modern "Bento Grid" Dashboard.

This version specifically addresses layout and scaling issues, ensuring all cards
and charts render correctly without deformation, while maintaining the existing
data simulation and backend logic.
"""

import os
import random
import time
from collections import deque
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string

from analysis.run_ai4i_case_study import (
    MODEL_FEATURES_ENHANCED,
    MODEL_PATH,
    add_derived_features,
)
from case_study import case_study_bp
from layout import render_sidebar
from rul_case_study import rul_case_study_bp

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

historical_data = deque(maxlen=8)
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartFactory - Bento Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>
    <style>
        :root {
            --sidebar-bg: #111827;
            --main-bg: #f3f4f6;
            --card-bg: #ffffff;
            --text-dark: #1f2937;
            --text-light: #f9fafb;
            --text-muted: #6b7280;
            --border-color: #e5e7eb;
        }

        body {
            background-color: var(--main-bg);
            font-family: 'Inter', sans-serif;
            color: var(--text-dark);
            margin: 0;
        }

        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 250px;
            height: 100vh;
            background-color: var(--sidebar-bg);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
        }

        .sidebar-logo {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-light);
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2.5rem;
        }
        .sidebar-logo i { color: #3b82f6; }

        .sidebar-nav a {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            color: #d1d5db;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            margin-bottom: 0.5rem;
            transition: background-color 0.2s;
        }
        .sidebar-nav a.active, .sidebar-nav a:hover {
            background-color: #1f2937;
            color: var(--text-light);
        }
        .sidebar-nav a i {
            width: 1.25em;
            margin-right: 0.75rem;
        }
        .sidebar-footer { margin-top: auto; }

        .main-content {
            margin-left: 250px;
            padding: 2rem;
        }

        .bento-card {
            background-color: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .kpi-card {
            height: 130px; /* Fixed height for KPI cards */
        }

        .kpi-card-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-muted);
        }
        .kpi-card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-dark);
            margin: auto 0;
            line-height: 1.2;
        }
        .kpi-card-trend {
            font-size: 0.85rem;
            font-weight: 500;
            align-self: flex-end;
        }

        .chart-card {
             height: 400px; /* Fixed height for chart cards */
        }

        .chart-container {
            position: relative;
            height: 100%;
            width: 100%;
        }

        .detail-card {
            height: 300px; /* Fixed height for detail cards */
            overflow: hidden;
        }

        .log-box {
            background-color: #1f2937;
            color: #d1d5db;
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
            border-bottom: 1px solid var(--border-color);
            font-size: 0.9rem;
        }
        .info-list-item:last-child { border-bottom: none; }
        .info-list-item .label { color: var(--text-muted); }
        .info-list-item .value { font-weight: 500; font-family: 'JetBrains Mono', monospace; }

        .scrollable-content {
            overflow-y: auto;
        }

    </style>
</head>
<body>
    {{ sidebar_html|safe }}

    <main class="main-content">
        <div class="row g-4">
            <!-- KPI ROW with fixed height cards -->
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">OEE</div><div class="kpi-card-value" id="kpi-oee">0%</div><div class="kpi-card-trend text-success">+0.1% <i class="fas fa-arrow-up"></i></div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">Net Production</div><div class="kpi-card-value" id="kpi-prod">0</div><div class="kpi-card-trend text-success">+5 <i class="fas fa-arrow-up"></i></div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">Instant Cost</div><div class="kpi-card-value" id="kpi-cost">$0/hr</div><div class="kpi-card-trend text-danger">+0.5 <i class="fas fa-arrow-up"></i></div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">AI Forecast (RUL)</div><div class="kpi-card-value" id="kpi-rul" title="Heuristic estimate derived from the model's risk score — not a calibrated survival/RUL model. See the RUL Case Study for genuine RUL regression.">0 Hr</div><div class="kpi-card-trend text-secondary">-1hr <i class="fas fa-arrow-down"></i></div></div></div>

            <!-- CHART ROW with fixed height cards and containers -->
            <div class="col-lg-6"><div class="bento-card chart-card"><h5>Real-time Temperature Analysis</h5><p class="small text-muted mb-2">Process temperature (model input)</p><div class="chart-container"><canvas id="tempChart"></canvas></div></div></div>
            <div class="col-lg-6"><div class="bento-card chart-card"><h5>Real-time Tool Wear Analysis</h5><p class="small text-muted mb-2">Tool wear accumulation (model input)</p><div class="chart-container"><canvas id="vibrationChart"></canvas></div></div></div>

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
            plugins: { legend: { display: false } },
            scales: { x: { grid: { display: false }, ticks: { color: '#9ca3af'} }, y: { grid: { color: '#e5e7eb' }, ticks: { color: '#9ca3af' } } }
        };
        const tempChart = new Chart(document.getElementById('tempChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                data: [],
                borderColor: '#ef4444',
                tension: 0.4,           // Makes lines curved
                borderWidth: 2,         // Sets line thickness
                pointRadius: 0,
                fill: true,
                backgroundColor: 'rgba(239, 68, 68, 0.05)'
            }] },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { callback: value => value + '°C' } } } }
        });
        const vibrationChart = new Chart(document.getElementById('vibrationChart'), {
            type: 'line',
            data: { labels: [], datasets: [{
                data: [],
                borderColor: '#3b82f6',
                tension: 0.4,           // Makes lines curved
                borderWidth: 2,         // Sets line thickness
                pointRadius: 0,
                fill: true,
                backgroundColor: 'rgba(59, 130, 246, 0.05)'
            }] },
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { callback: value => value + ' min' } } } }
        });

        async function fetchData() {
            const response = await fetch('/api/data');
            const data = await response.json();
            const current = data.current_reading;
            const prod = data.production_stats;
            const events = data.event_log;

            // --- Update KPIs ---
            document.getElementById('kpi-oee').textContent = prod.oee.toFixed(1) + '%';
            document.getElementById('kpi-prod').textContent = prod.actual;
            document.getElementById('kpi-cost').textContent = `$${current.Energy_Cost_Per_Hour.toFixed(2)}/hr`;
            document.getElementById('kpi-rul').textContent = `${current.RUL} Hr`;

            // --- Update Shift Info Panel ---
            const shiftPanel = document.getElementById('shift-info-panel');
            shiftPanel.innerHTML = `
                <div class="info-list-item"><span class="label">Shift ID</span><span class="value">${current.Shift_ID}</span></div>
                <div class="info-list-item"><span class="label">Operator</span><span class="value">${current.Operator}</span></div>
                <div class="info-list-item"><span class="label">Cycle Time</span><span class="value">${current.Cycle_Time.toFixed(1)}s</span></div>
                <div class="info-list-item"><span class="label">Power Factor</span><span class="value">${current.Power_Factor.toFixed(2)}</span></div>
                <div class="info-list-item"><span class="label">Machine Type</span><span class="value">${current.Type}</span></div>
                <div class="info-list-item"><span class="label">Model Risk Score</span><span class="value">${current.RiskScore.toFixed(4)}</span></div>
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
            [tempChart, vibrationChart].forEach(chart => {
                if (chart.data.labels.length > 20) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
                chart.data.labels.push(label);
            });
            tempChart.data.datasets[0].data.push(current.Temperature);
            vibrationChart.data.datasets[0].data.push(current.Tool_Wear);
            tempChart.update();
            vibrationChart.update();

            // --- Update Log Box ---
            const logBox = document.getElementById('log-box');
            logBox.innerHTML = '';
            events.forEach(event => {
                logBox.innerHTML += `<p><span class="text-muted">[${event.timestamp}]</span> ${event.message}</p>`;
            });
            logBox.scrollTop = logBox.scrollHeight;
        }

        setInterval(fetchData, 2000);
        document.addEventListener('DOMContentLoaded', fetchData);
    </script>
</body>
</html>
"""

# =============================================================================
# 3. BACKEND API & ROUTES (UNCHANGED)
# =============================================================================


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, sidebar_html=render_sidebar("dashboard"))


@app.route("/api/data")
def get_data():
    global production_actual, scrap_count, downtime_seconds, tool_wear_minutes

    # Simulate Production & MES Data
    production_actual += random.randint(0, 3)
    if random.random() < 0.02:
        scrap_count += 1
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
        "Type": machine_type,
        "Temperature": round(process_temp_k - 273.15, 1),  # process temp, Kelvin -> Celsius
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
        "Energy_Cost_Per_Hour": (current_val * voltage_val * 0.12) / 1000,  # Example cost calc
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


# =============================================================================
# 4. APPLICATION RUNNER (UNCHANGED)
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
