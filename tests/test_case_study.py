import unittest

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


if __name__ == "__main__":
    unittest.main()
