"""
Unit tests for visualizations.py.
Verifies TrainingLossVisualizer, SandwichedBlockVisualizer, DataSliceVisualizer, and CLI.
"""
import os
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visualizations import (
    ColorPalette,
    TrainingLossVisualizer,
    SandwichedBlockVisualizer,
    DataSliceVisualizer
)

class TestVisualizations(unittest.TestCase):

    def setUp(self):
        self.output_dir = "tests/test_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def test_color_palette(self):
        palette = ColorPalette.get_red_palette(3)
        self.assertEqual(len(palette), 3)
        for color in palette:
            self.assertTrue(color.startswith("#"))

    def test_training_loss_visualizer(self):
        vis = TrainingLossVisualizer()
        steps_per_epoch = 50
        vis.set_steps_per_epoch(steps_per_epoch)

        for s in range(1, 151):
            if (s - 1) % steps_per_epoch == 0:
                vis.add_epoch_start(s // steps_per_epoch, s)
            vis.add_train_step(s, 1.0 / (1.0 + s * 0.05))
            if s % 25 == 0:
                vis.add_val_loss(s, 1.1 / (1.0 + s * 0.05))
                vis.add_test_loss(s, 1.15 / (1.0 + s * 0.05))

        out_path = os.path.join(self.output_dir, "test_loss_curve.png")
        fig = vis.generate_plot(save_path=out_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_sandwiched_block_visualizer(self):
        n = 300
        dates = pd.date_range("1991-01-31", periods=n, freq="15min")
        df = pd.DataFrame({
            "DateTimeFormatted": dates,
            "Te1": 1500 + 500 * np.sin(np.linspace(0, 2*np.pi, n)),
            "Te1_pred": 1500 + 480 * np.sin(np.linspace(0, 2*np.pi, n)),
            "split": ["train"] * 100 + ["test"] * 100 + ["train"] * 100
        })

        slice_df, meta = SandwichedBlockVisualizer.extract_sandwiched_slice(
            df, split_col="split", min_block_size=50
        )
        self.assertEqual(meta["train1_len"], 100)
        self.assertEqual(meta["test_len"], 100)
        self.assertEqual(meta["train2_len"], 100)

        m1 = np.zeros(len(slice_df), dtype=bool); m1[:100] = True
        m2 = np.zeros(len(slice_df), dtype=bool); m2[100:200] = True
        m3 = np.zeros(len(slice_df), dtype=bool); m3[200:] = True

        vis = SandwichedBlockVisualizer()
        out_path = os.path.join(self.output_dir, "test_sandwiched.png")
        fig = vis.generate_plot(slice_df, m1, m2, m3, save_path=out_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_data_slice_visualizer(self):
        n = 50
        dates = pd.date_range("1991-05-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "DateTimeFormatted": dates,
            "Te1": np.random.uniform(1000, 3000, n),
            "Te1_pred": np.random.uniform(1000, 3000, n)
        })
        vis = DataSliceVisualizer()
        out_path = os.path.join(self.output_dir, "test_slice.png")
        fig = vis.plot_slice(df, save_path=out_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(out_path))

    def test_epoch_test_loss_plot(self):
        vis = TrainingLossVisualizer()
        steps_per_epoch = 50
        vis.set_steps_per_epoch(steps_per_epoch)

        for s in range(1, 151):
            if (s - 1) % steps_per_epoch == 0:
                vis.add_epoch_start(s // steps_per_epoch, s)
            vis.add_train_step(s, 1.0 / (1.0 + s * 0.05))
            if s % 25 == 0:
                vis.add_val_loss(s, 1.1 / (1.0 + s * 0.05))
                vis.add_test_loss(s, 1.15 / (1.0 + s * 0.05))

        out_path = os.path.join(self.output_dir, "test_epoch_test_loss.png")
        fig = vis.generate_epoch_test_loss_plot(save_path=out_path)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_all_michael_cases(self):
        import torch
        import torch.nn as nn

        # Dummy model
        class DummyModel(nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 150))

        model = DummyModel()
        device = torch.device("cpu")

        # Dummy datasets
        n_train = 600
        n_test = 400
        train_ds = {
            "input_ids": torch.randn(n_train, 10),
            "label": torch.randint(0, 150, (n_train,))
        }
        test_ds = {
            "input_ids": torch.randn(n_test, 10),
            "label": torch.randint(0, 150, (n_test,))
        }

        paths = SandwichedBlockVisualizer.generate_all_michael_cases(
            model=model,
            device=device,
            train_ds=train_ds,
            test_ds=test_ds,
            block_size=50,
            num_candidates=5,
            output_dir=self.output_dir,
            model_name="test_model"
        )

        self.assertIn("random", paths)
        self.assertIn("best", paths)
        self.assertIn("worst", paths)
        self.assertIn("median", paths)
        self.assertIn("mean", paths)

        for case, p in paths.items():
            self.assertTrue(os.path.exists(p), f"Path {p} for case {case} does not exist")
            self.assertGreater(os.path.getsize(p), 5000)

if __name__ == "__main__":
    unittest.main()

