"""Protect local environment secrets without hiding shareable example config."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GitignoreHygieneTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("Git is required to verify Git's actual ignore behavior")
        self.repo = tempfile.TemporaryDirectory()
        self.addCleanup(self.repo.cleanup)
        self.workdir = Path(self.repo.name)
        subprocess.run(
            ["git", "init", "-q", str(self.workdir)],
            check=True,
            capture_output=True,
            text=True,
        )
        (self.workdir / ".gitignore").write_text(
            (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
        )

    def _ignored(self, relative_path):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", relative_path],
            cwd=self.workdir,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return result.returncode == 0

    def test_environment_files_are_ignored_at_any_depth(self):
        for path in (".env", ".env.local", ".env.production", "config/.env", "analysis/.env.dev"):
            with self.subTest(path=path):
                self.assertTrue(self._ignored(path))

    def test_shareable_example_files_remain_trackable(self):
        for path in (".env.example", "analysis/.env.example"):
            with self.subTest(path=path):
                self.assertFalse(self._ignored(path))


if __name__ == "__main__":
    unittest.main()
