import unittest

from main import app
from rul_case_study import load_rul_case_study_artifacts


class RulCaseStudyRouteTests(unittest.TestCase):
    def test_artifacts_load(self):
        artifacts = load_rul_case_study_artifacts()
        self.assertIn("summary", artifacts)
        self.assertIn("final_model", artifacts)

    def test_rul_case_study_route(self):
        client = app.test_client()
        response = client.get("/rul-case-study")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"RUL regression on a genuine run-to-failure dataset", response.data)

    def test_rul_case_study_api(self):
        client = app.test_client()
        response = client.get("/api/rul-case-study")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("summary", payload)
        self.assertIn("final_model", payload)
        self.assertIn("baseline", payload)


if __name__ == "__main__":
    unittest.main()
