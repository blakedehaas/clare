"""Unit test suite verifying performance metrics, R2, RMSE, MAE, and threshold percentages in evaluate.py."""

import unittest
import numpy as np
from evaluate import compute_metrics, input_columns, output_columns, all_columns


class TestEvaluate(unittest.TestCase):
    """Validates evaluation metrics computation and column metadata consistency."""

    def test_column_definitions(self):
        """Tests that input and output column schemas match telemetry specifications."""
        self.assertEqual(len(output_columns), 1)
        self.assertEqual(output_columns[0], "Te1")
        self.assertIn("Altitude", input_columns)
        self.assertIn("SYM_H_0", input_columns)
        self.assertIn("AL_index_0", input_columns)
        self.assertEqual(len(all_columns), len(input_columns) + len(output_columns))

    def test_compute_metrics_exact_match(self):
        """Tests metrics evaluation when predictions match ground truth exactly."""
        true_vals = [2000.0, 3000.0, 4000.0, 5000.0]
        pred_vals = [2000.0, 3000.0, 4000.0, 5000.0]
        entropy_vals = [0.1, 0.2, 0.15, 0.05]

        metrics = compute_metrics(pred_vals, true_vals, entropy_vals)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["mean_entropy"], 0.125)
        for pct in metrics["percentages"]:
            self.assertEqual(pct, 100.0)
        for rel_pct in metrics["relative_percentages"]:
            self.assertEqual(rel_pct, 100.0)

    def test_compute_metrics_known_deviations(self):
        """Tests metrics with known non-zero deviations."""
        true_vals = [1000.0, 2000.0, 3000.0, 4000.0]
        # Deviations: +50, -150, +250, -350
        pred_vals = [1050.0, 1850.0, 3250.0, 3650.0]

        metrics = compute_metrics(pred_vals, true_vals)
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertGreater(metrics["mae"], 0.0)
        self.assertLess(metrics["r2"], 1.0)
        # Check absolute threshold counts:
        # <= 100: only index 0 (50) -> 1/4 = 25%
        self.assertEqual(metrics["percentages"][0], 25.0)
        # <= 200: index 0 (50), index 1 (150) -> 2/4 = 50%
        self.assertEqual(metrics["percentages"][1], 50.0)


    def test_feed_forward_network(self):
        """Tests FeedForwardNetwork initialization and forward pass dimensions."""
        import torch
        from models.feed_forward import FeedForwardNetwork
        model = FeedForwardNetwork(input_size=154, hidden_size=64, output_size=1)
        x = torch.randn(8, 154)
        out = model(x)
        self.assertEqual(out.shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
