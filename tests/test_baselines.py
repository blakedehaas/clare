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
        # Literal hand calculations at h=1000 km and |glat|=10 degrees.
        cases = [
            (10, 12, 3628.0),   # DNE: 2778 + 630 + 220
            (-10, 12, 3654.0),  # DSE: 3114 + 500 + 40
            (10, 23, 1836.0),   # NNE: 1536 + 170 + 130
            (-10, 23, 1886.0),  # NSE: 1676 + 140 + 70
        ]
        for glat, gmlt, expected in cases:
            self.assertEqual(float(predict_kutiev_2002(1000, glat, gmlt)), expected)

    def test_published_midlatitude_equation_and_coefficients(self):
        # At L=2 the (L^5 - 2^5) correction is exactly zero.
        altitude = 1000.0
        radius_ratio = (EARTH_RADIUS_KM + altitude) / EARTH_RADIUS_KM
        glat_l2 = np.rad2deg(np.arccos(np.sqrt(radius_ratio / 2.0)))
        cases = [
            (glat_l2, 12, 5566.0), (-glat_l2, 12, 4928.0),
            (glat_l2, 23, 2261.0), (-glat_l2, 23, 2187.0),
        ]
        for glat, gmlt, expected in cases:
            self.assertAlmostEqual(float(predict_kutiev_2002(altitude, glat, gmlt)), expected, places=3)

        # At L=3, L^5 - 2^5 = 211, giving simple literal endpoint values.
        glat_l3 = np.rad2deg(np.arccos(np.sqrt(radius_ratio / 3.0)))
        cases = [
            (glat_l3, 12, 9553.9), (-glat_l3, 12, 9274.6),
            (glat_l3, 23, 6270.0), (-glat_l3, 23, 6238.2),
        ]
        for glat, gmlt, expected in cases:
            self.assertAlmostEqual(float(predict_kutiev_2002(altitude, glat, gmlt)), expected, places=2)

    def test_rejects_regions_not_defined_by_the_paper(self):
        predictions = predict_kutiev_2002(
            [999, 6371, 3000, 3000, 3000, 3000, 3000],
            [0, 0, 0, 70, 0, 0, 0],
            [12, 12, 18, 12, np.nan, -1, 24],
        )
        self.assertTrue(np.isnan(predictions).all())
        self.assertEqual(MAX_REGRESSION_ALTITUDE_KM, 6370.0)
        self.assertTrue(np.isfinite(predict_kutiev_2002(1000, 0, 12)))
        self.assertTrue(np.isfinite(predict_kutiev_2002(6370, 0, 12)))

    def test_published_local_time_endpoints(self):
        expected_day, expected_night = 3628.0, 1836.0
        for gmlt in (9.0, 16.0):
            self.assertEqual(float(predict_kutiev_2002(1000, 10, gmlt)), expected_day)
        for gmlt in (22.0, 4.0):
            self.assertEqual(float(predict_kutiev_2002(1000, 10, gmlt)), expected_night)
        for gmlt in (4.001, 8.999, 16.001, 21.999):
            self.assertTrue(np.isnan(predict_kutiev_2002(1000, 10, gmlt)))

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
