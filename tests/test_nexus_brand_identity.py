"""NEXUS brand integration contracts: files, accessibility and page parity.

Pure offline checks; no Flask server, Google Drive requests or network traffic.
"""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from layout import render_sidebar

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
NAMESPACE = "{http://www.w3.org/2000/svg}"
FAVICON = 'rel="icon" type="image/svg+xml" href="/static/nexus-official-x.svg"'


class NexusBrandIdentityTests(unittest.TestCase):
    def test_approved_logos_are_real_parseable_svg(self):
        for name in ("nexus-official-wordmark.svg", "nexus-official-x.svg"):
            with self.subTest(asset=name):
                root = ET.parse(STATIC / name).getroot()
                self.assertEqual(root.tag, NAMESPACE + "svg")
                self.assertTrue(root.get("viewBox"))
                self.assertTrue(root.get("role") == "img")
                tags = {element.tag.rsplit("}", 1)[-1] for element in root.iter()}
                self.assertNotIn("script", tags)
                self.assertNotIn("foreignObject", tags)

    def test_approved_wordmark_png_is_full_size_rgba(self):
        image = (STATIC / "nexus-official-wordmark.png").read_bytes()
        self.assertTrue(image.startswith(b"\\x89PNG\\r\\n\\x1a\\n"))
        self.assertEqual(int.from_bytes(image[16:20], "big"), 1200)
        self.assertEqual(int.from_bytes(image[20:24], "big"), 360)
        self.assertEqual(image[25], 6, "Official wordmark needs RGBA transparency")

    def test_brand_gradient_matches_approved_nexus_palette(self):
        for name in ("nexus-official-wordmark.svg", "nexus-official-x.svg"):
            with self.subTest(asset=name):
                source = (STATIC / name).read_text(encoding="utf-8")
                for color in ("#00E0FF", "#4EA8FF", "#8F9BFF", "#6E55C6"):
                    self.assertIn(color, source)

    def test_shared_sidebar_has_real_logo_and_keyboard_accessible_home(self):
        for page in ("dashboard", "case-study", "rul-case-study", "settings"):
            with self.subTest(page=page):
                html = render_sidebar(page)
                self.assertIn('src="/static/nexus-official-wordmark.png"', html)
                self.assertIn('alt="NEXUS"', html)
                self.assertIn('class="nexus-brand-link" href="/"', html)
                self.assertIn("Skip to content", html)
                self.assertIn('aria-label="Primary"', html)

    def test_favicon_is_consistent_across_server_pages(self):
        for name in ("main.py", "case_study.py", "rul_case_study.py", "settings.py"):
            with self.subTest(page=name):
                self.assertIn(FAVICON, (ROOT / name).read_text(encoding="utf-8"))

    def test_decision_studio_uses_official_icon_and_dark_brand_theme(self):
        html = (STATIC / "decision-studio.html").read_text(encoding="utf-8")
        css = (STATIC / "decision-studio.css").read_text(encoding="utf-8")
        self.assertIn(FAVICON, html)
        self.assertIn('src="/static/nexus-official-x.svg"', html)
        self.assertIn('alt="NEXUS"', html)
        self.assertIn("#0b0f19", css)
        self.assertIn("color-scheme: dark", css)

    def test_brand_styles_do_not_hide_telemetry_provenance(self):
        css = (STATIC / "nexus-design.css").read_text(encoding="utf-8")
        self.assertIn("nexus-brand-wordmark", css)
        self.assertIn("body.sf-telemetry-stale", css)
        self.assertIn("prefers-reduced-motion", css)

    def test_docker_image_contains_brand_assets(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY static/ ./static/", dockerfile)
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertNotIn("static/", dockerignore)


if __name__ == "__main__":
    unittest.main()
