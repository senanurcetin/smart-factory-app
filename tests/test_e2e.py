"""Real-browser smoke tests for all four routes.

Every other test in this suite exercises the backend (Flask `test_client()`,
artifact contracts) — none of them render a page in an actual browser, so a
broken script tag, a 404'ing asset, or a JS error would never fail CI. These
tests catch that class of bug: a real Chromium instance loads each route and
asserts on console errors and key DOM elements.

The Flask app under test already has `model.pkl`/`rul_model.pkl` and all
`docs/data/**/*.json` artifacts committed to git (see Engineering Decisions
in README.md), so no retraining is needed to serve real pages here.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import unittest
import urllib.request
from urllib.error import URLError

from werkzeug.serving import make_server

from main import app

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - only hit if requirements-dev.txt wasn't installed
    sync_playwright = None

HOST = "127.0.0.1"
PORT = 8099
BASE_URL = f"http://{HOST}:{PORT}"


class _ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.server = make_server(HOST, PORT, app)

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def _wait_until_ready(timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE_URL, timeout=1)
            return
        except URLError:
            time.sleep(0.2)
    raise RuntimeError(f"Server never became ready at {BASE_URL}")


@unittest.skipIf(sync_playwright is None, "playwright not installed (see requirements-dev.txt)")
class E2ESmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_thread = _ServerThread()
        cls.server_thread.start()
        _wait_until_ready()

        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server_thread.shutdown()

    def _open(self, path: str):
        page = self.browser.new_page()
        console_errors = []
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        # Runtime exceptions can occur without a console-error message.
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.goto(f"{BASE_URL}{path}", wait_until="networkidle")
        return page, console_errors

    def test_dashboard_loads(self):
        page, console_errors = self._open("/")
        try:
            self.assertTrue(page.is_visible(".sidebar"))
            self.assertEqual(
                page.text_content(".page-title").strip(), "Industrial Analytics Dashboard"
            )
            self.assertTrue(page.is_visible("#riskChart"))
            self.assertTrue(page.is_visible("#oee-availability"))
            self.assertTrue(page.is_visible('a[href="/api/report.csv"]'))
            page.wait_for_function(
                "() => document.querySelector('#riskChart') && "
                "document.querySelector('#oee-availability').textContent !== '—'"
            )
            self.assertTrue(page.is_visible("#kpi-risk"))
            self.assertEqual(console_errors, [])
        finally:
            page.close()

    def test_nexus_executive_design_at_1440_and_optional_screenshot(self):
        page, console_errors = self._open("/")
        try:
            page.set_viewport_size({"width": 1440, "height": 900})
            page.wait_for_function(
                "() => document.querySelector('#nexus-stream-badge')"
                ".dataset.state === 'active'", timeout=15000
            )
            self.assertTrue(page.is_visible(".nexus-hero"))
            self.assertTrue(page.is_visible("#riskChart"))
            self.assertTrue(page.is_visible("#nexus-stream-badge"))
            self.assertTrue(page.is_visible('a[href="/api/report.csv"]'))
            output = os.environ.get("E2E_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(Path(output) / "smart-factory-desktop-1440.png"), full_page=True)
            self.assertEqual(console_errors, [])
        finally:
            page.close()

    def test_nexus_mobile_menu_keyboard_navigation_and_overflow(self):
        page, console_errors = self._open("/")
        try:
            page.set_viewport_size({"width": 375, "height": 812})
            page.wait_for_timeout(300)
            self.assertLessEqual(
                page.evaluate("document.documentElement.scrollWidth"),
                page.evaluate("document.documentElement.clientWidth") + 1
            )
            page.click("#sidebar-toggle")
            self.assertEqual(page.locator("#sidebar-toggle").get_attribute("aria-expanded"), "true")
            self.assertTrue(page.evaluate("document.body.classList.contains('sf-nav-open')"))
            self.assertTrue(page.is_visible(".sidebar.open"))
            page.keyboard.press("Escape")
            self.assertEqual(page.locator("#sidebar-toggle").get_attribute("aria-expanded"), "false")
            self.assertFalse(page.evaluate("document.body.classList.contains('sf-nav-open')"))
            self.assertFalse(page.is_visible(".sidebar.open"))
            page.locator("#riskChart").scroll_into_view_if_needed()
            self.assertTrue(page.is_visible("#riskChart"))
            output = os.environ.get("E2E_SCREENSHOT_DIR")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(Path(output) / "smart-factory-mobile-375.png"), full_page=True)
            self.assertEqual(console_errors, [])
        finally:
            page.close()

    def test_nexus_stale_metrics_are_visibly_distinct_on_api_failure(self):
        page = self.browser.new_page(viewport={"width": 1440, "height": 900})
        self.addCleanup(page.close)
        page.route("**/api/data", lambda route: route.abort("failed"))
        page.goto(BASE_URL + "/", wait_until="load")
        page.wait_for_function(
            "() => document.getElementById('nexus-stream-badge')"
            ".dataset.state === 'unavailable'", timeout=15000
        )
        self.assertTrue(page.evaluate(
            "document.body.classList.contains('sf-telemetry-stale')"
        ))
        self.assertIn(
            "STREAM UNAVAILABLE",
            page.locator("#nexus-stream-label").inner_text()
        )
        output = os.environ.get("E2E_SCREENSHOT_DIR")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(Path(output) / "smart-factory-telemetry-outage.png"),
                full_page=True,
            )

    def test_case_study_loads_with_charts(self):
        page, console_errors = self._open("/case-study")
        try:
            self.assertTrue(page.is_visible(".sidebar"))
            broken_images = page.eval_on_selector_all(
                "img", "imgs => imgs.filter(i => i.naturalWidth === 0).map(i => i.src)"
            )
            self.assertEqual(broken_images, [])
            image_count = page.eval_on_selector_all("img", "imgs => imgs.length")
            self.assertGreaterEqual(image_count, 8)
            self.assertEqual(console_errors, [])
        finally:
            page.close()

    def test_rul_case_study_loads_with_charts(self):
        page, console_errors = self._open("/rul-case-study")
        try:
            self.assertTrue(page.is_visible(".sidebar"))
            broken_images = page.eval_on_selector_all(
                "img", "imgs => imgs.filter(i => i.naturalWidth === 0).map(i => i.src)"
            )
            self.assertEqual(broken_images, [])
            image_count = page.eval_on_selector_all("img", "imgs => imgs.length")
            self.assertGreaterEqual(image_count, 4)
            self.assertEqual(console_errors, [])
        finally:
            page.close()

    def test_settings_loads_with_working_controls(self):
        page, console_errors = self._open("/settings")
        try:
            self.assertTrue(page.is_visible("#refresh-rate-select"))
            page.select_option("#refresh-rate-select", "500")
            refresh_ms = page.evaluate("localStorage.getItem('sf_refresh_ms')")
            self.assertEqual(refresh_ms, "500")
            self.assertEqual(console_errors, [])
        finally:
            page.close()

    def test_sidebar_collapses_on_mobile_viewport(self):
        page, console_errors = self._open("/")
        try:
            page.set_viewport_size({"width": 375, "height": 812})
            page.wait_for_timeout(350)  # let the media-query/transition settle after resize
            closed_transform = page.eval_on_selector(
                ".sidebar", "el => getComputedStyle(el).transform"
            )
            self.assertNotEqual(closed_transform, "matrix(1, 0, 0, 1, 0, 0)")
            self.assertFalse(page.is_visible(".sidebar.open"))

            page.click("#sidebar-toggle")
            page.wait_for_timeout(350)  # let the 0.25s slide transition settle
            open_transform = page.eval_on_selector(
                ".sidebar", "el => getComputedStyle(el).transform"
            )
            self.assertEqual(open_transform, "matrix(1, 0, 0, 1, 0, 0)")
            self.assertTrue(page.is_visible(".sidebar.open"))

            page.click(".sidebar-backdrop")
            self.assertFalse(page.is_visible(".sidebar.open"))
            self.assertEqual(console_errors, [])
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
