"""Prevent CircleCI v2.1 from interpreting shell heredocs as template tags.

CircleCI parses the YAML before any job starts. Bare heredoc opening tokens
must be escaped in config text; source files and Python test strings are safe.
This module uses only the Python standard library and never contacts a service.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".circleci" / "config.yml"
BAD_HEREDOC = re.compile(r"(?<!\\)<<(['\"]?)[A-Za-z_][A-Za-z0-9_]*\1")


def invalid_lines(source: str) -> list[int]:
    return [
        number
        for number, line in enumerate(source.splitlines(), 1)
        if not line.lstrip().startswith("#") and BAD_HEREDOC.search(line)
    ]


def main() -> int:
    if not CONFIG.is_file():
        print("FAIL: CircleCI configuration missing", file=sys.stderr)
        return 1
    bad = invalid_lines(CONFIG.read_text(encoding="utf-8"))
    if bad:
        print(
            f"FAIL: .circleci/config.yml contains unescaped shell heredoc on line(s): {bad}",
            file=sys.stderr,
        )
        return 1
    print("PASS: no unescaped shell heredoc markers in CircleCI v2.1 configuration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
