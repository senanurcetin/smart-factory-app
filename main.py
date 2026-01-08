
"""
A single-file Flask application with a Modern "Bento Grid" Dashboard.

This version specifically addresses layout and scaling issues, ensuring all cards
and charts render correctly without deformation, while maintaining the existing
data simulation and backend logic.
"""

import os
import random
from collections import deque
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. BACKEND: APPLICATION & ENRICHED DATA SIMULATION (UNCHANGED)
# =============================================================================

app = Flask(__name__)

features_for_model = ['Temperature', 'Vibration', 'Current', 'Pressure', 'RPM', 'Energy']
data = {feat: [random.uniform(1, 100) for _ in range(100)] for feat in features_for_model}
data['Failure'] = [random.choice([0, 1]) for _ in range(100)]
df = pd.DataFrame(data)
X = df[features_for_model]
y = df['Failure']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
model.fit(X_train, y_train)

historical_data = deque(maxlen=8)
event_log = deque(maxlen=10)
production_actual = 0
scrap_count = 0

# =============================================================================
# 2. FRONTEND: BENTO GRID TEMPLATE (Layout & Style FIXES)
# =============================================================================

HTML_TEMPLATE = '''
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
    <div class="sidebar">
        <div class="sidebar-logo"><i class="fas fa-bolt"></i> SmartFactory</div>
        <nav class="sidebar-nav">
            <a href="#" class="active"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
            <a href="#"><i class="fas fa-chart-line"></i> Analytics</a>
            <a href="#"><i class="fas fa-cogs"></i> Settings</a>
        </nav>
         <div class="sidebar-footer">
            <nav class="sidebar-nav">
                <a href="#"><i class="fas fa-user-circle"></i> Profile</a>
            </nav>
        </div>
    </div>

    <main class="main-content">
        <div class="row g-4">
            <!-- KPI ROW with fixed height cards -->
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">OEE</div><div class="kpi-card-value" id="kpi-oee">0%</div><div class="kpi-card-trend text-success">+0.1% <i class="fas fa-arrow-up"></i></div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">Net Production</div><div class="kpi-card-value" id="kpi-prod">0</div><div class="kpi-card-trend text-success">+5 <i class="fas fa-arrow-up"></i></div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">Instant Cost</div><div class="kpi-card-value" id="kpi-cost">$0/hr</div><div class="kpi-card-trend text-danger">+0.5 <i class="fas fa-arrow-up"></i></div></div></div>
            <div class="col-lg-3 col-6"><div class="bento-card kpi-card text-center"><div class="kpi-card-title">AI Forecast (RUL)</div><div class="kpi-card-value" id="kpi-rul">0 Hr</div><div class="kpi-card-trend text-secondary">-1hr <i class="fas fa-arrow-down"></i></div></div></div>

            <!-- CHART ROW with fixed height cards and containers -->
            <div class="col-lg-6"><div class="bento-card chart-card"><h5>Real-time Temperature Analysis</h5><p class="small text-muted mb-2">Sensor: THERM-L5</p><div class="chart-container"><canvas id="tempChart"></canvas></div></div></div>
            <div class="col-lg-6"><div class="bento-card chart-card"><h5>Real-time Vibration Analysis</h5><p class="small text-muted mb-2">Sensor: PIEZO-V1</p><div class="chart-container"><canvas id="vibrationChart"></canvas></div></div></div>

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
            options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { callback: value => value + ' mm/s' } } } }
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
            vibrationChart.data.datasets[0].data.push(current.Vibration);
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
'''

# =============================================================================
# 3. BACKEND API & ROUTES (UNCHANGED)
# =============================================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/data")
def get_data():
    global production_actual, scrap_count

    # Simulate Production & MES Data
    production_actual += random.randint(0, 3)
    if random.random() < 0.02: scrap_count += 1
    production_stats = {
        'target': 1000, 'actual': production_actual, 'scrap': scrap_count, 'oee': random.uniform(92.0, 98.5)
    }

    # Simulate Enriched Sensor & Metadata
    current_val = random.uniform(3.0, 4.9)
    voltage_val = random.uniform(218.0, 223.0)

    reading = {
        'Timestamp': datetime.now().strftime("%H:%M:%S"),
        'Temperature': round(random.uniform(65.0, 85.0), 1),
        'Vibration': round(random.uniform(0.7, 2.5), 2),
        'Current': round(current_val, 2),
        'Pressure': round(random.uniform(30.0, 42.0), 1),
        'RPM': int(random.uniform(1800, 2500)),
        'Energy': round(random.uniform(12.0, 17.5), 2),
        'Spindle_Load': round(random.uniform(60.0, 95.0), 1),
        'Voltage': round(voltage_val, 1),
        'Hydraulic_Pressure': round(random.uniform(115.0, 148.0), 1),
        'Power_Factor': round(random.uniform(0.88, 0.99), 2),
        'Cycle_Time': round(random.uniform(11.5, 14.2), 1),
        'Shift_ID': "SHIFT-A" if 8 <= datetime.now().hour < 20 else "SHIFT-B",
        'Operator': "John Doe" if 8 <= datetime.now().hour < 20 else "Jane Smith",
        'Energy_Cost_Per_Hour': (current_val * voltage_val * 0.12) / 1000, # Example cost calc
        'RUL': int(48 - (production_actual / 250)), # Remaining Useful Life in hours
    }

    # AI Risk Prediction
    df_live = pd.DataFrame([reading], columns=features_for_model)
    risk_score = model.predict_proba(df_live)[:, 1][0]
    reading['RiskScore'] = round(risk_score, 4)

    historical_data.append(reading)
    
    # Simulate System Events
    if random.random() < 0.2: 
        event_log.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': random.choice(['INFO: Cycle OK', 'WARN: Temp high', 'INFO: Pressure nominal'])
        })

    return jsonify({
        'current_reading': reading,
        'production_stats': production_stats,
        'event_log': list(event_log)
    })

# =============================================================================
# 4. APPLICATION RUNNER (UNCHANGED)
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
