import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import case_study
from case_study import load_case_study_artifacts
from main import app


class CaseStudyRouteTests(unittest.TestCase):
    def test_artifacts_load(self):
        artifacts = load_case_study_artifacts()
        self.assertIn("summary", artifacts)
        self.assertIn("benchmarks", artifacts)

    def test_case_study_route(self):
        client = app.test_client()
        response = client.get("/case-study")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predictive maintenance triage benchmark", response.data)

    def test_case_study_api(self):
        client = app.test_client()
        response = client.get("/api/case-study")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("summary", payload)
        self.assertIn("review_queue", payload)


class CaseStudyMissingArtifactsTests(unittest.TestCase):
    """The pipeline hasn't run yet (fresh clone, wiped artifacts dir, etc.) -
    every route touching these artifacts should fail loud with a 503 and a
    helpful message, never a raw 500 stack trace."""

    def setUp(self):
        self.empty_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.empty_dir.cleanup)
        self.patcher = patch.object(case_study, "ARTIFACT_DIR", Path(self.empty_dir.name))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_load_artifacts_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_case_study_artifacts()
        self.assertIn("run_ai4i_case_study.py", str(ctx.exception))

    def test_case_study_route_returns_503(self):
        client = app.test_client()
        response = client.get("/case-study")
        self.assertEqual(response.status_code, 503)

    def test_case_study_api_returns_503(self):
        client = app.test_client()
        response = client.get("/api/case-study")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
