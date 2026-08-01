from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, render_template_string

from layout import SIDEBAR_CSS, render_sidebar

AI4I_DIR = Path(__file__).resolve().parent / "docs" / "data" / "ai4i-case-study"
RUL_DIR = Path(__file__).resolve().parent / "docs" / "data" / "cmapss-rul-case-study"

SETTINGS_TEMPLATE = (
    """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Factory Settings</title>
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
            max-width: 900px;
            margin: 0 auto;
            padding: 32px 20px 64px;
            margin-left: 250px;
            box-sizing: border-box;
        }
        .eyebrow {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        h1, h2, h3, p { margin: 0; }
        h1 { font-size: clamp(1.8rem, 4vw, 2.6rem); line-height: 1.1; margin-bottom: 8px; }
        .lede { color: var(--muted); font-size: 1rem; line-height: 1.6; margin-bottom: 28px; }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 22px;
            margin-bottom: 18px;
            transition: border-color 0.2s, transform 0.2s;
        }
        .panel:hover { border-color: rgba(96, 165, 250, 0.4); }
        .section-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 6px; }
        .section-copy { color: var(--muted); font-size: 0.88rem; line-height: 1.55; margin-bottom: 16px; }
        .setting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 14px 0;
            border-bottom: 1px solid rgba(159, 180, 209, 0.14);
        }
        .setting-row:last-child { border-bottom: none; }
        .setting-label { font-weight: 600; font-size: 0.92rem; }
        .setting-hint { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
        select.setting-control {
            background: var(--panel-soft);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
        }
        .theme-switch {
            position: relative;
            width: 52px;
            height: 28px;
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            cursor: pointer;
            flex-shrink: 0;
        }
        .theme-switch .knob {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--muted);
            transition: transform 0.2s, background 0.2s;
        }
        .theme-switch.is-on .knob {
            transform: translateX(24px);
            background: var(--accent-2);
        }
        .save-toast {
            display: inline-block;
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.3s;
            margin-left: 10px;
        }
        .save-toast.show { opacity: 1; }
        table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
        th, td { padding: 9px 8px; border-bottom: 1px solid rgba(159, 180, 209, 0.16); text-align: left; }
        th { color: var(--muted); font-weight: 500; width: 45%; }
        tbody tr { transition: background-color 0.15s; }
        tbody tr:hover { background: rgba(94, 234, 212, 0.08); }
        .mono { font-family: 'IBM Plex Mono', monospace; }
        .model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .links-list { display: flex; flex-direction: column; gap: 10px; }
        .links-list a { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 0.92rem; transition: transform 0.15s; }
        .links-list a:hover { transform: translateX(4px); }
        .links-list i { width: 1.2em; color: var(--muted); }
        @media (max-width: 760px) {
            .model-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    {{ sidebar_html|safe }}
    <main class="page">
        <div class="eyebrow">Settings</div>
        <h1>Preferences &amp; Model Info</h1>
        <p class="lede">Controls for the live dashboard, plus the model card metadata for both trained pipelines — training date, library versions, and the exact dataset hash each model was fit on.</p>

        <section class="panel">
            <h2 class="section-title">Dashboard Preferences</h2>
            <p class="section-copy">These apply to the live <a href="/">Dashboard</a> and are saved in your browser (no account, no server round-trip).</p>
            <div class="setting-row">
                <div>
                    <div class="setting-label">Refresh interval</div>
                    <div class="setting-hint">How often the dashboard polls <code>/api/data</code> for a new simulated reading.</div>
                </div>
                <div>
                    <select class="setting-control" id="refresh-rate-select">
                        <option value="500">0.5s</option>
                        <option value="1000">1s</option>
                        <option value="2000" selected>2s (default)</option>
                        <option value="5000">5s</option>
                    </select>
                    <span class="save-toast" id="refresh-toast">Saved</span>
                </div>
            </div>
            <div class="setting-row">
                <div>
                    <div class="setting-label">Dark mode</div>
                    <div class="setting-hint">Switch the dashboard from its default light bento theme to dark.</div>
                </div>
                <div style="display:flex; align-items:center;">
                    <div class="theme-switch" id="theme-switch" role="switch" aria-checked="false" tabindex="0">
                        <div class="knob"></div>
                    </div>
                    <span class="save-toast" id="theme-toast">Saved</span>
                </div>
            </div>
        </section>

        <section class="panel">
            <h2 class="section-title">Model Card</h2>
            <p class="section-copy">Metadata recorded automatically at training time by <code>analysis/run_ai4i_case_study.py</code> and <code>analysis/run_cmapss_rul_case_study.py</code> — regenerated on every CI run, never hand-edited.</p>
            <div class="model-grid">
                <div>
                    <h3 style="font-size: 0.95rem; margin-bottom: 8px;">AI4I Failure Classifier</h3>
                    <table>
                        <tbody>
                            <tr><th>Trained</th><td class="mono">{{ ai4i_card.trained_at_utc }}</td></tr>
                            <tr><th>scikit-learn</th><td class="mono">{{ ai4i_card.scikit_learn_version }}</td></tr>
                            <tr><th>SHAP</th><td class="mono">{{ ai4i_card.shap_version }}</td></tr>
                            <tr><th>Dataset rows</th><td class="mono">{{ ai4i_card.dataset.rows }}</td></tr>
                            <tr><th>Dataset SHA-256</th><td class="mono" title="{{ ai4i_card.dataset.sha256 }}">{{ ai4i_card.dataset.sha256[:16] }}&hellip;</td></tr>
                            <tr><th>Random seed</th><td class="mono">{{ ai4i_card.random_seed }}</td></tr>
                        </tbody>
                    </table>
                </div>
                <div>
                    <h3 style="font-size: 0.95rem; margin-bottom: 8px;">C-MAPSS RUL Regressor</h3>
                    <table>
                        <tbody>
                            <tr><th>Trained</th><td class="mono">{{ rul_card.trained_at_utc }}</td></tr>
                            <tr><th>scikit-learn</th><td class="mono">{{ rul_card.scikit_learn_version }}</td></tr>
                            <tr><th>SHAP</th><td class="mono">{{ rul_card.shap_version }}</td></tr>
                            <tr><th>Dataset rows</th><td class="mono">{{ rul_card.dataset.rows }}</td></tr>
                            <tr><th>Dataset SHA-256</th><td class="mono" title="{{ rul_card.dataset.sha256 }}">{{ rul_card.dataset.sha256[:16] }}&hellip;</td></tr>
                            <tr><th>Random seed</th><td class="mono">{{ rul_card.random_seed }}</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <section class="panel">
            <h2 class="section-title">Links</h2>
            <div class="links-list">
                <a href="https://github.com/senanurcetin/smart-factory-app" target="_blank" rel="noopener"><i class="fab fa-github"></i> Source code on GitHub</a>
                <a href="https://senanur-cetin.vercel.app/projects/smart-factory-app" target="_blank" rel="noopener"><i class="fas fa-id-badge"></i> Portfolio project entry</a>
                <a href="/case-study"><i class="fas fa-chart-line"></i> AI4I Case Study</a>
                <a href="/rul-case-study"><i class="fas fa-hourglass-half"></i> RUL Case Study</a>
            </div>
        </section>
    </main>

    <script>
        const REFRESH_KEY = 'sf_refresh_ms';
        const THEME_KEY = 'sf_dashboard_theme';

        const refreshSelect = document.getElementById('refresh-rate-select');
        const refreshToast = document.getElementById('refresh-toast');
        refreshSelect.value = localStorage.getItem(REFRESH_KEY) || '2000';
        refreshSelect.addEventListener('change', () => {
            localStorage.setItem(REFRESH_KEY, refreshSelect.value);
            flashToast(refreshToast);
        });

        const themeSwitch = document.getElementById('theme-switch');
        const themeToast = document.getElementById('theme-toast');
        function setThemeSwitch(isDark) {
            themeSwitch.classList.toggle('is-on', isDark);
            themeSwitch.setAttribute('aria-checked', String(isDark));
        }
        setThemeSwitch(localStorage.getItem(THEME_KEY) === 'dark');
        function toggleTheme() {
            const isDark = !themeSwitch.classList.contains('is-on');
            setThemeSwitch(isDark);
            localStorage.setItem(THEME_KEY, isDark ? 'dark' : 'light');
            flashToast(themeToast);
        }
        themeSwitch.addEventListener('click', toggleTheme);
        themeSwitch.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleTheme(); }
        });

        function flashToast(el) {
            el.classList.add('show');
            clearTimeout(el._timer);
            el._timer = setTimeout(() => el.classList.remove('show'), 1400);
        }
    </script>
</body>
</html>
"""
)

settings_bp = Blueprint("settings", __name__)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_settings_context() -> dict:
    ai4i_card = _load_json(AI4I_DIR / "model-selection.json").get("model_card", {})
    rul_card = _load_json(RUL_DIR / "model-selection.json").get("model_card", {})
    return {"ai4i_card": ai4i_card, "rul_card": rul_card}


@settings_bp.route("/settings")
def settings_page():
    context = load_settings_context()
    return render_template_string(
        SETTINGS_TEMPLATE, sidebar_html=render_sidebar("settings"), **context
    )
