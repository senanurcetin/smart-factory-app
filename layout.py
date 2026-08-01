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
            position: relative;
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            color: #d1d5db;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            margin-bottom: 0.5rem;
            transition: background-color 0.15s, color 0.15s, padding-left 0.15s;
        }
        .sidebar-nav a.active, .sidebar-nav a:hover {
            background-color: #1f2937;
            color: #f9fafb;
            padding-left: 1.15rem;
        }
        .sidebar-nav a.active {
            box-shadow: inset 3px 0 0 0 #3b82f6;
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

FOOTER_NAV_ITEMS = [
    ("settings", "/settings", "fa-cogs", "Settings"),
]


def render_sidebar(active: str) -> str:
    """Render the sidebar nav HTML, marking `active` (one of the NAV_ITEMS
    or FOOTER_NAV_ITEMS keys) as the current page."""

    def _link(key: str, href: str, icon: str, label: str) -> str:
        active_class = ' class="active"' if key == active else ""
        return f'<a href="{href}"{active_class}><i class="fas {icon}"></i> {label}</a>'

    links_html = "\n            ".join(_link(*item) for item in NAV_ITEMS)
    footer_links_html = "\n                ".join(_link(*item) for item in FOOTER_NAV_ITEMS)
    return f"""<div class="sidebar">
        <div class="sidebar-logo"><i class="fas fa-bolt"></i> SmartFactory</div>
        <nav class="sidebar-nav">
            {links_html}
        </nav>
        <div class="sidebar-footer">
            <nav class="sidebar-nav">
                {footer_links_html}
            </nav>
        </div>
    </div>"""
