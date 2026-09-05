"""
tests/test_baselines.py - Unit tests for baseline models (Kutiev et al., 2002).
"""

import unittest
import numpy as np
from baselines.kutiev_2002 import predict_kutiev_2002, evaluate_kutiev_metrics


class TestBaselines(unittest.TestCase):
    """Verifies physical consistency and mathematical mechanics of baseline models."""

    def test_kutiev_altitude_monotonicity(self):
        """Tests that electron temperature increases monotonically with altitude in plasmasphere."""
        alts = np.array([1000.0, 2500.0, 5000.0, 8000.0], dtype=np.float32)
        ilat = np.full_like(alts, 35.0)
        gmlt = np.full_like(alts, 14.0)

        preds = predict_kutiev_2002(alts, ilat, gmlt)

        self.assertEqual(len(preds), 4)
        # Verify strictly increasing with height
        for i in range(len(preds) - 1):
            self.assertGreater(preds[i + 1], preds[i])
        # Base temperature around 2000-3500 K, topside around 5000-9000 K
        self.assertGreaterEqual(preds[0], 1500.0)
        self.assertLessEqual(preds[-1], 12000.0)

    def test_kutiev_diurnal_variation(self):
        """Tests day vs night electron temperature variation (daytime > nighttime)."""
        alt = 3000.0
        ilat = 40.0
        gmlt_day = 14.0
        gmlt_night = 2.0

        t_day = predict_kutiev_2002(alt, ilat, gmlt_day)
        t_night = predict_kutiev_2002(alt, ilat, gmlt_night)

        self.assertGreater(float(t_day), float(t_night))

    def test_kutiev_latitudinal_variation(self):
        """Tests that high latitude has higher base temperature than equatorial."""
        alt = 2000.0
        gmlt = 12.0
        t_high_lat = predict_kutiev_2002(alt, 60.0, gmlt)
        t_equator = predict_kutiev_2002(alt, 5.0, gmlt)

        self.assertGreater(float(t_high_lat), float(t_equator))

    def test_kutiev_kp_modulation(self):
        """Tests that elevated Kp moderately increases electron temperature."""
        alt = 4000.0
        ilat = 45.0
        gmlt = 15.0

        t_quiet = predict_kutiev_2002(alt, ilat, gmlt, kp=1.0)
        t_storm = predict_kutiev_2002(alt, ilat, gmlt, kp=6.0)

        self.assertGreater(float(t_storm), float(t_quiet))

    def test_evaluate_kutiev_metrics(self):
        """Tests evaluation metrics aggregation for Kutiev predictions."""
        y_true = np.array([2500.0, 3000.0, 4000.0, 5000.0], dtype=np.float32)
        y_pred = np.array([2550.0, 2900.0, 4100.0, 4900.0], dtype=np.float32)

        metrics = evaluate_kutiev_metrics(y_true, y_pred)
        self.assertIn("r2", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("acc_10", metrics)
        self.assertGreater(metrics["r2"], 0.95)
        self.assertEqual(metrics["acc_10"], 100.0)

        # Empty array handling
        empty_metrics = evaluate_kutiev_metrics(np.array([]), np.array([]))
        self.assertEqual(empty_metrics["r2"], 0.0)
