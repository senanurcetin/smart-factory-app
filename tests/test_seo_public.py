"""Public search and social metadata regression tests."""

import runpy
import unittest
from pathlib import Path


class SeoContractTests(unittest.TestCase):
    def test_public_search_and_social_contract(self):
        checker = Path(__file__).resolve().parents[1] / "scripts" / "check_seo_assets.py"
        runpy.run_path(str(checker), run_name="seo_checker")["check"]()
