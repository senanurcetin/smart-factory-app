"""Guard the Render Docker image's Flask modules and NEXUS static assets."""

from __future__ import annotations

import ast
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def docker_copies() -> list[tuple[list[str], str]]:
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    result: list[tuple[list[str], str]] = []
    for line in lines:
        if not line.startswith("COPY "):
            continue
        arguments = shlex.split(line.removeprefix("COPY "))
        if arguments and arguments[0].startswith("--from="):
            continue
        result.append((arguments[:-1], arguments[-1]))
    return result


class DockerImageContractTests(unittest.TestCase):
    def test_every_local_main_module_is_in_runtime_image(self):
        source_files = {
            name for sources, _ in docker_copies() for name in sources if name.endswith(".py")
        }
        imported_locals = {
            node.module + ".py"
            for node in ast.walk(ast.parse((ROOT / "main.py").read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom)
            and node.module
            and "." not in node.module
            and (ROOT / (node.module + ".py")).is_file()
        }
        missing = imported_locals - source_files
        self.assertFalse(missing, f"Docker image missing imported local modules: {sorted(missing)}")

    def test_nexus_static_assets_are_in_runtime_image(self):
        directories = {
            (name, destination)
            for sources, destination in docker_copies()
            for name in sources
            if name.endswith("/")
        }
        self.assertIn(("static/", "./static/"), directories)
        for filename in (
            "nexus-design.css",
            "nexus-official-wordmark.svg",
            "nexus-official-wordmark.png",
            "nexus-official-x.svg",
            "nexus-analysis-workspace.css",
            "nexus-factory-studio.css",
            "decision-studio.html",
            "decision-studio.js",
            "decision-studio.css",
        ):
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / "static" / filename).is_file(), filename)

    def test_static_directory_is_not_excluded_from_context(self):
        ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertNotIn("static/", ignored)
        self.assertNotIn("decision_studio.py", ignored)


if __name__ == "__main__":
    unittest.main()
