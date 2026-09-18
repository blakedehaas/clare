"""Checks against Kutiev et al. (2002), equations (1)-(2), Tables 2-3."""

import unittest

import numpy as np

from baselines.kutiev_2002 import (
    EARTH_RADIUS_KM,
    EQUATORIAL,
    MAX_REGRESSION_ALTITUDE_KM,
    MIDLATITUDE,
    evaluate_kutiev_metrics,
    predict_kutiev_2002,
)


class TestBaselines(unittest.TestCase):
    def test_published_tables_are_transcribed_exactly(self):
        self.assertEqual(EQUATORIAL, {
            "DNE": (2778.0, 0.63, 2.2), "DSE": (3114.0, 0.50, 0.4),
            "NNE": (1536.0, 0.17, 1.3), "NSE": (1676.0, 0.14, 0.7),
        })
        self.assertEqual(MIDLATITUDE, {
            "DNM": (5326.0, 0.24, 18.9), "DSM": (4558.0, 0.37, 20.6),
            "NNM": (2161.0, 0.10, 19.0), "NSM": (2077.0, 0.11, 19.2),
        })

    def test_published_equatorial_equation_and_coefficients(self):
        # DNE, Table 2: Te = (2778 + 0.63 alt) + 2.2 glat^2
        self.assertAlmostEqual(float(predict_kutiev_2002(3000, 10, 12)), 4888.0)
        # NSE, Table 2: Te = (1676 + 0.14 alt) + 0.7 glat^2
        self.assertAlmostEqual(float(predict_kutiev_2002(3000, -10, 23)), 2166.0)

        cases = [
            (10, 12, 2778.0 + 0.63 * 3000 + 2.2 * 100),
            (-10, 12, 3114.0 + 0.50 * 3000 + 0.4 * 100),
            (10, 23, 1536.0 + 0.17 * 3000 + 1.3 * 100),
            (-10, 23, 1676.0 + 0.14 * 3000 + 0.7 * 100),
        ]
        for glat, gmlt, expected in cases:
            self.assertAlmostEqual(float(predict_kutiev_2002(3000, glat, gmlt)), expected)

    def test_published_midlatitude_equation_and_coefficients(self):
        alt, glat = 1000.0, 45.0
        l_shell = (EARTH_RADIUS_KM + alt) / (EARTH_RADIUS_KM * np.cos(np.deg2rad(glat)) ** 2)
        expected = 5326.0 + 0.24 * alt + 18.9 * (l_shell ** 5 - 32.0)
        self.assertAlmostEqual(float(predict_kutiev_2002(alt, glat, 12)), expected, places=3)

        cases = [
            (45, 12, 5326.0 + 0.24 * alt + 18.9 * (l_shell ** 5 - 32.0)),
            (-45, 12, 4558.0 + 0.37 * alt + 20.6 * (l_shell ** 5 - 32.0)),
            (45, 23, 2161.0 + 0.10 * alt + 19.0 * (l_shell ** 5 - 32.0)),
            (-45, 23, 2077.0 + 0.11 * alt + 19.2 * (l_shell ** 5 - 32.0)),
        ]
        for glat, gmlt, expected in cases:
            self.assertAlmostEqual(float(predict_kutiev_2002(alt, glat, gmlt)), expected, places=3)

    def test_rejects_regions_not_defined_by_the_paper(self):
        predictions = predict_kutiev_2002(
            [999, 6371, 3000, 3000, 3000, 3000, 3000],
            [0, 0, 0, 70, 0, 0, 0],
            [12, 12, 18, 12, np.nan, -1, 24],
        )
        self.assertTrue(np.isnan(predictions).all())
        self.assertEqual(MAX_REGRESSION_ALTITUDE_KM, 6370.0)

    def test_explicit_extrapolation_covers_full_sensitivity_cohort(self):
        predictions = predict_kutiev_2002(
            [7000, 3000, 3000], [0, 0, 75], [18, 18, 12], extrapolate=True
        )
        self.assertTrue(np.isfinite(predictions).all())
        # 18 MLT is nearer the published daytime sector than the nighttime sector.
        self.assertAlmostEqual(float(predictions[1]), 2778.0 + 0.63 * 3000)

    def test_published_l_shell_boundaries(self):
        alt = 1000.0
        radius_ratio = (EARTH_RADIUS_KM + alt) / EARTH_RADIUS_KM
        glat_l2 = np.rad2deg(np.arccos(np.sqrt(radius_ratio / 2.0)))
        glat_l3 = np.rad2deg(np.arccos(np.sqrt(radius_ratio / 3.0)))
        self.assertAlmostEqual(float(predict_kutiev_2002(alt, glat_l2, 12)), 5326.0 + 0.24 * alt)
        self.assertTrue(np.isfinite(predict_kutiev_2002(alt, glat_l3, 12)))
        self.assertTrue(np.isnan(predict_kutiev_2002(alt, glat_l3 + 0.01, 12)))

    def test_metrics_ignore_unsupported_samples(self):
        metrics = evaluate_kutiev_metrics(
            np.array([2500.0, 3000.0, 4000.0, 5000.0]),
            np.array([2550.0, 2900.0, np.nan, 4900.0]),
        )
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["acc_10"], 100.0)


if __name__ == "__main__":
    unittest.main()
