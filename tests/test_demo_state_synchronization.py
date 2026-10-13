"""Ensure demo API and CSV export use a single process-local lock."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from demo_state import synchronized_demo_view


class DemoStateLockTests(unittest.TestCase):
    def test_different_views_share_the_same_exclusion_lock(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        @synchronized_demo_view
        def update_state():
            first_entered.set()
            if not release_first.wait(timeout=5):
                raise AssertionError("first view was never released")
            return "updated"

        @synchronized_demo_view
        def read_snapshot():
            second_entered.set()
            return "snapshot"

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(update_state)
                self.assertTrue(first_entered.wait(timeout=2))
                second = executor.submit(read_snapshot)
                self.assertFalse(second_entered.wait(timeout=0.05))
                release_first.set()
                self.assertEqual(first.result(timeout=5), "updated")
                self.assertEqual(second.result(timeout=5), "snapshot")
                self.assertTrue(second_entered.is_set())
        finally:
            release_first.set()

    def test_decorator_preserves_view_identity_and_return_value(self):
        @synchronized_demo_view
        def handler(value):
            """Example docstring."""
            return value + 1

        self.assertEqual(handler.__name__, "handler")
        self.assertEqual(handler.__doc__, "Example docstring.")
        self.assertEqual(handler(6), 7)
        self.assertTrue(handler.demo_state_synchronized)

    def test_api_and_export_routes_are_serialized(self):
        from pathlib import Path

        if not (Path(__file__).resolve().parents[1] / "main.py").exists():
            self.skipTest("full application checkout is needed for Flask route integration")
        from main import app

        for url in ("/api/data", "/api/report.csv"):
            with self.subTest(url=url):
                endpoint = next(rule.endpoint for rule in app.url_map.iter_rules() if rule.rule == url)
                view = app.view_functions[endpoint]
                self.assertTrue(getattr(view, "demo_state_synchronized", False))

    def test_docker_runtime_contains_state_module(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("demo_state.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
