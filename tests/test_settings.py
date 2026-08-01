import unittest

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


if __name__ == "__main__":
    unittest.main()
