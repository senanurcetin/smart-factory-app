"""Validate the intended Docker build-context boundaries without starting Docker."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DockerContextHygieneTests(unittest.TestCase):
    def test_local_environment_files_excluded_at_root_and_nested_levels(self):
        rules = {
            line.strip()
            for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue({".env", ".env.*", "**/.env", "**/.env.*"}.issubset(rules))
        self.assertIn(".circleci/", rules)
        self.assertNotIn("!.env.example", rules)

    def test_runtime_dockerfile_does_not_copy_whole_context(self):
        lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
        copies = [line.strip() for line in lines if line.strip().upper().startswith("COPY ")]
        self.assertTrue(copies)
        self.assertFalse(any(line.startswith(("COPY . ", "COPY ./ ")) for line in copies))


if __name__ == "__main__":
    unittest.main()
