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
        .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: #111827;
            color: #f9fafb;
            padding: 10px 16px;
            z-index: 100;
            border-radius: 0 0 8px 0;
            transition: top 0.15s;
        }
        .skip-link:focus { top: 0; }
        .sidebar-toggle {
            display: none;
            position: fixed;
            top: 14px;
            left: 14px;
            z-index: 30;
            width: 42px;
            height: 42px;
            align-items: center;
            justify-content: center;
            background-color: #111827;
            color: #f9fafb;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
        }
        .sidebar-backdrop {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 15;
        }
        @media (max-width: 900px) {
            .sidebar-toggle { display: flex; }
            .sidebar {
                transform: translateX(-100%);
                transition: transform 0.25s ease;
            }
            .sidebar.open { transform: translateX(0); }
            .sidebar-backdrop.open { display: block; }
        }
"""

NAV_ITEMS = [
    ("dashboard", "/", "fa-tachometer-alt", "Dashboard"),
    ("case-study", "/case-study", "fa-chart-line", "Predictive Maintenance"),
    ("rul-case-study", "/rul-case-study", "fa-hourglass-half", "RUL Prognostics"),
]

FOOTER_NAV_ITEMS = [
    ("settings", "/settings", "fa-cogs", "Settings"),
]


def render_sidebar(active: str) -> str:
    """Render the sidebar nav HTML, marking `active` (one of the NAV_ITEMS
    or FOOTER_NAV_ITEMS keys) as the current page."""

    def _link(key: str, href: str, icon: str, label: str) -> str:
        active_class = ' class="active" aria-current="page"' if key == active else ""
        return f'<a href="{href}"{active_class}><i class="fas {icon}"></i> {label}</a>'

    links_html = "\n            ".join(_link(*item) for item in NAV_ITEMS)
    footer_links_html = "\n                ".join(_link(*item) for item in FOOTER_NAV_ITEMS)
    return f"""<a class="skip-link" href="#main-content">Skip to content</a>
    <button class="sidebar-toggle" id="sidebar-toggle" aria-label="Open navigation" aria-controls="sidebar" aria-expanded="false">
        <i class="fas fa-bars"></i>
    </button>
    <div class="sidebar-backdrop" id="sidebar-backdrop"></div>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-logo">
            <a class="nexus-brand-link" href="/" aria-label="NEXUS Smart Factory home">
                <img class="nexus-brand-wordmark" src="/static/nexus-official-wordmark.png"
                     alt="NEXUS" width="196" height="59" decoding="async">
                <span class="nexus-product">Smart Factory</span>
                <span class="nexus-caption">INDUSTRIAL INTELLIGENCE</span>
            </a>
        </div>
        <nav class="sidebar-nav" aria-label="Primary">
            {links_html}
        </nav>
        <div class="sidebar-footer">
            <nav class="sidebar-nav" aria-label="Secondary">
                {footer_links_html}
            </nav>
        </div>
    </div>
    <script>
        (function () {{
            const sidebar = document.getElementById('sidebar');
            const toggle = document.getElementById('sidebar-toggle');
            const backdrop = document.getElementById('sidebar-backdrop');
            function setSidebarOpen(open) {{
                const mobile = window.matchMedia('(max-width: 900px)').matches;
                const visible = Boolean(open && mobile);
                sidebar.classList.toggle('open', visible);
                backdrop.classList.toggle('open', visible);
                document.body.classList.toggle('sf-nav-open', visible);
                toggle.setAttribute('aria-expanded', String(visible));
                toggle.setAttribute('aria-label', visible ? 'Close navigation' : 'Open navigation');
                if (visible) {{
                    const first = sidebar.querySelector('a');
                    if (first) first.focus();
                }}
            }}
            function closeSidebar() {{ setSidebarOpen(false); }}
            function toggleSidebar() {{
                setSidebarOpen(!sidebar.classList.contains('open'));
            }}
            toggle.addEventListener('click', toggleSidebar);
            backdrop.addEventListener('click', function () {{
                closeSidebar();
                toggle.focus();
            }});
            document.addEventListener('keydown', function (event) {{
                if (event.key === 'Escape' && sidebar.classList.contains('open')) {{
                    closeSidebar();
                    toggle.focus();
                }}
            }});
            window.addEventListener('resize', function () {{
                if (!window.matchMedia('(max-width: 900px)').matches) closeSidebar();
            }});
            sidebar.querySelectorAll('a').forEach(function (a) {{
                a.addEventListener('click', closeSidebar);
            }});
        }})();
    </script>"""
