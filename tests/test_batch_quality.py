"""Offline tests for the AI4I batch data contract. No network/model required."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.validate_ai4i_batch import main, validate_batch


def valid_data():
    return pd.DataFrame(
        {
            "Type": ["L", "H"],
            "Air temperature [K]": [298.0, 301.0],
            "Process temperature [K]": [308.0, 310.0],
            "Rotational speed [rpm]": [1450, 1700],
            "Torque [Nm]": [43.2, 60.5],
            "Tool wear [min]": [0, 200],
        }
    )


class BatchContractTests(unittest.TestCase):
    def test_good_input_passes_without_moving_values(self):
        frame = valid_data()
        before = frame.copy(deep=True)
        report = validate_batch(frame)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["valid_rows"], 2)
        self.assertEqual(report["invalid_rows"], 0)
        pd.testing.assert_frame_equal(frame, before)

    def test_additional_columns_allowed(self):
        frame = valid_data()
        frame["Product ID"] = ["A1", "B2"]
        frame["Target"] = [0, 1]
        self.assertEqual(validate_batch(frame)["status"], "pass")

    def test_missing_column_fails_closed(self):
        frame = valid_data().drop(columns=["Torque [Nm]"])
        report = validate_batch(frame)
        self.assertEqual(report["missing_columns"], ["Torque [Nm]"])
        self.assertEqual(report["invalid_rows"], len(frame))
        self.assertEqual(report["status"], "fail")

    def test_empty_batch_rejected(self):
        result = validate_batch(valid_data().iloc[:0])
        self.assertEqual(result["status"], "fail")
        self.assertIn("empty_batch", result["errors_by_field"]["_batch"])

    def test_bad_numeric_and_categories(self):
        frame = valid_data().astype({"Torque [Nm]": "object"})
        frame.loc[0, "Torque [Nm]"] = "not-a-number"
        frame.loc[1, "Type"] = "X"
        result = validate_batch(frame)
        self.assertEqual(result["invalid_rows"], 2)
        self.assertEqual(result["errors_by_field"]["Torque [Nm]"]["not_numeric"], 1)
        self.assertEqual(result["errors_by_field"]["Type"]["unknown_category"], 1)
        self.assertEqual(result["error_examples"][0]["row_number"], 3)

    def test_missing_non_finite_and_physical_constraints(self):
        frame = valid_data()
        frame.loc[0, "Air temperature [K]"] = 0
        frame.loc[1, "Process temperature [K]"] = float("inf")
        frame.loc[0, "Tool wear [min]"] = -1
        frame.loc[1, "Torque [Nm]"] = float("nan")
        result = validate_batch(frame)
        self.assertEqual(result["invalid_rows"], 2)
        self.assertEqual(
            result["errors_by_field"]["Process temperature [K]"]["non_finite"], 1
        )
        self.assertEqual(
            result["errors_by_field"]["Air temperature [K]"]["physically_invalid"], 1
        )
        self.assertEqual(
            result["errors_by_field"]["Tool wear [min]"]["physically_invalid"], 1
        )
        self.assertEqual(result["errors_by_field"]["Torque [Nm]"]["missing_value"], 1)

    def test_sample_examples_cap(self):
        frame = valid_data()
        frame.loc[0, "Type"] = "Z"
        frame.loc[1, "Type"] = "?"
        report = validate_batch(frame, sample_limit=1)
        self.assertEqual(len(report["error_examples"]), 1)
        self.assertTrue(report["error_examples_truncated"])
        with self.assertRaises(ValueError):
            validate_batch(frame, sample_limit=-1)

    def test_cli_success_and_failure_exit_codes_with_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "batch.csv"
            report_path = Path(tmp) / "reports" / "quality.json"
            valid_data().to_csv(csv, index=False)
            self.assertEqual(main([str(csv), "--report", str(report_path)]), 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")

            bad = valid_data()
            bad.loc[0, "Tool wear [min]"] = -1
            bad.to_csv(csv, index=False)
            self.assertEqual(main([str(csv), "--report", str(report_path)]), 2)
            self.assertEqual(json.loads(report_path.read_text())["invalid_rows"], 1)

    def test_cli_missing_file_returns_io_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main([str(Path(tmp) / "missing.csv")]), 1)


if __name__ == "__main__":
    unittest.main()
