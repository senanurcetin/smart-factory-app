"""Fast regression for the CircleCI parser problem fixed in PR #29."""

import unittest

from scripts.check_circleci_heredocs import invalid_lines, main


class CircleCIHeredocTests(unittest.TestCase):
    def test_unescaped_shell_heredoc_rejected(self):
        self.assertEqual(invalid_lines("docker exec app python - <<'PY'"), [1])
        self.assertEqual(invalid_lines("cat <<EOF"), [1])

    def test_escaped_shell_heredoc_and_circleci_parameter_accepted(self):
        self.assertEqual(invalid_lines("docker exec app python - \\<<'PY'"), [])
        self.assertEqual(invalid_lines("echo '<< parameters.foo >>'"), [])
        self.assertEqual(invalid_lines("# example cat <<EOF"), [])

    def test_repository_config_is_safe(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
