"""Shared sidebar navigation, reused by main.py, case_study.py, and
rul_case_study.py so the same nav appears on every page instead of only on
the dashboard.

Colors are hardcoded (not CSS custom properties) so this renders correctly
regardless of which page's own `:root` variables are defined — the bento
dashboard and the two case-study pages use different theme variables.
"""

from __future__ import annotations

SIDEBAR_CSS = """
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 250px;
            height: 100vh;
            background-color: #111827;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            z-index: 10;
        }
        .sidebar-logo {
            font-family: 'JetBrains Mono', monospace;
            color: #f9fafb;
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
            color: #f9fafb;
        }
        .sidebar-nav a i {
            width: 1.25em;
            margin-right: 0.75rem;
        }
        .sidebar-footer { margin-top: auto; }
"""

NAV_ITEMS = [
    ("dashboard", "/", "fa-tachometer-alt", "Dashboard"),
    ("case-study", "/case-study", "fa-chart-line", "Case Study"),
    ("rul-case-study", "/rul-case-study", "fa-hourglass-half", "RUL Case Study"),
]


def render_sidebar(active: str) -> str:
    """Render the sidebar nav HTML, marking `active` (one of the NAV_ITEMS
    keys) as the current page."""
    links = []
    for key, href, icon, label in NAV_ITEMS:
        active_class = ' class="active"' if key == active else ""
        links.append(f'<a href="{href}"{active_class}><i class="fas {icon}"></i> {label}</a>')
    links_html = "\n            ".join(links)
    return f"""<div class="sidebar">
        <div class="sidebar-logo"><i class="fas fa-bolt"></i> SmartFactory</div>
        <nav class="sidebar-nav">
            {links_html}
            <a href="#"><i class="fas fa-cogs"></i> Settings</a>
        </nav>
        <div class="sidebar-footer">
            <nav class="sidebar-nav">
                <a href="#"><i class="fas fa-user-circle"></i> Profile</a>
            </nav>
        </div>
    </div>"""
