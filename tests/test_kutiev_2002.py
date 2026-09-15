import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from baselines.kutiev_2002 import (
    l_shell_from_invariant_latitude,
    predict_kutiev_2002,
)
from baselines.evaluate_kutiev import (
    calculate_metrics,
    evaluate_dataset,
    write_comparison_rows,
)


class FakeDataset:
    def __init__(self, columns):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, column):
        return self._columns[column]

    def __len__(self):
        return len(next(iter(self._columns.values())))


class Kutiev2002Tests(unittest.TestCase):
    def test_l_shell_from_invariant_latitude(self):
        actual = l_shell_from_invariant_latitude([0.0, 45.0, 54.735610317])
        np.testing.assert_allclose(actual, [1.0, 2.0, 3.0], rtol=1e-9)

    def test_daytime_north_equatorial_equation(self):
        result = predict_kutiev_2002(3000.0, 10.0, 30.0, 12.0)
        self.assertTrue(result.eligible.item())
        self.assertEqual(result.zone.item(), "DNE")
        self.assertAlmostEqual(result.temperature_k.item(), 4888.0)

    def test_nighttime_south_equatorial_equation(self):
        result = predict_kutiev_2002(2000.0, -10.0, -30.0, 23.0)
        self.assertTrue(result.eligible.item())
        self.assertEqual(result.zone.item(), "NSE")
        self.assertAlmostEqual(result.temperature_k.item(), 2026.0)

    def test_daytime_north_midlatitude_equation_at_l_two(self):
        result = predict_kutiev_2002(3000.0, 35.0, 45.0, 9.0)
        self.assertTrue(result.eligible.item())
        self.assertEqual(result.zone.item(), "DNM")
        self.assertAlmostEqual(result.temperature_k.item(), 6046.0)

    def test_vectorized_input_and_applicability_mask(self):
        result = predict_kutiev_2002(
            [3000.0, 3000.0, 3000.0, 900.0],
            [10.0, 10.0, 10.0, 10.0],
            [30.0, 60.0, 30.0, 30.0],
            [12.0, 12.0, 18.0, 12.0],
        )
        np.testing.assert_array_equal(result.eligible, [True, False, False, False])
        self.assertTrue(np.isfinite(result.temperature_k[0]))
        self.assertTrue(np.all(np.isnan(result.temperature_k[1:])))

    def test_night_sector_wraps_across_midnight(self):
        result = predict_kutiev_2002(
            [3000.0, 3000.0],
            [10.0, 10.0],
            [30.0, 30.0],
            [23.0, 2.0],
        )
        np.testing.assert_array_equal(result.zone, ["NNE", "NNE"])

    def test_invalid_local_times_are_not_wrapped_into_valid_sectors(self):
        result = predict_kutiev_2002(
            [3000.0, 3000.0, 3000.0],
            [10.0, 10.0, 10.0],
            [30.0, 30.0, 30.0],
            [-1.0, 24.0, 25.0],
        )
        np.testing.assert_array_equal(result.eligible, [False, False, False])
        self.assertTrue(np.all(np.isnan(result.temperature_k)))

    def test_midlatitude_rows_above_6370_km_are_excluded(self):
        result = predict_kutiev_2002(
            [6370.0, 6370.1],
            [35.0, 35.0],
            [45.0, 45.0],
            [12.0, 12.0],
        )
        np.testing.assert_array_equal(result.eligible, [True, False])
        self.assertTrue(np.isfinite(result.temperature_k[0]))
        self.assertTrue(np.isnan(result.temperature_k[1]))

    def test_reported_metrics(self):
        metrics = calculate_metrics([1000.0, 2000.0], [1050.0, 2300.0])
        self.assertEqual(metrics["accuracy_within_10_percent"], 50.0)
        self.assertAlmostEqual(metrics["rmse_k"], np.sqrt((50.0**2 + 300.0**2) / 2.0))
        self.assertEqual(metrics["sample_count"], 2)

    def test_dataset_report_includes_coverage(self):
        dataset = FakeDataset(
            {
                "Altitude": [3000.0, 3000.0, 3000.0],
                "GLAT": [10.0, 10.0, 10.0],
                "ILAT": [30.0, 30.0, 30.0],
                "GMLT": [12.0, 23.0, 18.0],
                "Te1": [4888.0, 2176.0, 3000.0],
            }
        )
        report = evaluate_dataset(dataset)
        self.assertEqual(report["dataset_rows"], 3)
        self.assertEqual(report["eligible_rows"], 2)
        self.assertAlmostEqual(report["coverage_percent"], 200.0 / 3.0)

    def test_paired_comparison_uses_identical_finite_rows(self):
        dataset = FakeDataset(
            {
                "Altitude": [3000.0, 3000.0, 3000.0],
                "GLAT": [10.0, 10.0, 10.0],
                "ILAT": [30.0, 30.0, 30.0],
                "GMLT": [12.0, 23.0, 18.0],
                "Te1": [4888.0, 2176.0, 3000.0],
            }
        )
        report = evaluate_dataset(dataset, [4888.0, 2000.0, 3000.0], "clare")
        paired = report["paired_comparison"]
        self.assertEqual(paired["sample_count"], 2)
        self.assertEqual(paired["kutiev_2002"]["sample_count"], 2)
        self.assertEqual(paired["clare"]["sample_count"], 2)

    def test_comparison_row_export_preserves_dataset_indices(self):
        dataset = FakeDataset(
            {
                "Altitude": [3000.0, 3000.0, 3000.0],
                "GLAT": [10.0, 10.0, 10.0],
                "ILAT": [30.0, 30.0, 30.0],
                "GMLT": [12.0, 18.0, 23.0],
                "Te1": [4888.0, 3000.0, 2176.0],
                "DateTimeFormatted": ["t0", "t1", "t2"],
            }
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "rows.csv"
            write_comparison_rows(
                output_path,
                dataset,
                [4888.0, 3000.0, np.nan],
                "clare",
            )
            rows = output_path.read_text(encoding="utf-8").splitlines()
        self.assertIn("clare_te_k", rows[0])
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1].startswith("0,t0,"))


if __name__ == "__main__":
    unittest.main()
