import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import settings
from main import app
from settings import load_settings_context


class SettingsRouteTests(unittest.TestCase):
    def test_context_has_both_model_cards(self):
        context = load_settings_context()
        self.assertIn("ai4i_card", context)
        self.assertIn("rul_card", context)
        self.assertIn("trained_at_utc", context["ai4i_card"])
        self.assertIn("trained_at_utc", context["rul_card"])

    def test_settings_route(self):
        client = app.test_client()
        response = client.get("/settings")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Preferences", response.data)
        self.assertIn(b"Model Card", response.data)

    def test_sidebar_settings_link_is_wired(self):
        client = app.test_client()
        response = client.get("/")
        self.assertIn(b'href="/settings"', response.data)
        self.assertNotIn(b'href="#"', response.data)


class SettingsMissingArtifactsTests(unittest.TestCase):
    """Unlike case_study.py/rul_case_study.py, settings.py has nothing that
    requires the model-selection.json artifacts to exist - it should degrade
    to an empty model card, not 503, since the rest of the page (refresh
    interval, links) is still fully usable without them."""

    def setUp(self):
        self.empty_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.empty_dir.cleanup)
        empty_path = Path(self.empty_dir.name)
        self.ai4i_patcher = patch.object(settings, "AI4I_DIR", empty_path)
        self.rul_patcher = patch.object(settings, "RUL_DIR", empty_path)
        self.ai4i_patcher.start()
        self.rul_patcher.start()
        self.addCleanup(self.ai4i_patcher.stop)
        self.addCleanup(self.rul_patcher.stop)

    def test_context_degrades_to_empty_cards(self):
        context = load_settings_context()
        self.assertEqual(context["ai4i_card"], {})
        self.assertEqual(context["rul_card"], {})

    def test_settings_route_still_returns_200(self):
        client = app.test_client()
        response = client.get("/settings")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Preferences", response.data)


if __name__ == "__main__":
    unittest.main()
