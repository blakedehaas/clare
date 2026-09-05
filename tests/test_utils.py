"""Unit test suite verifying dataset sampling, normalization, and statistical transformation utilities in utils.py."""

import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils import (
    SamplingDataset,
    calculate_stats,
    normalize_ds,
    unnormalize_mean,
    unnormalize_var
)


class TestUtils(unittest.TestCase):
    """Validates multi-dataset sampling and normalization operations."""

    def test_sampling_dataset_defaults_and_sampling(self):
        """Tests SamplingDataset with default uniform ratios and index generation."""
        ds1 = TensorDataset(torch.ones(10, 2))
        ds2 = TensorDataset(torch.zeros(20, 2))

        sampler = SamplingDataset([ds1, ds2], batch_size=4)
        self.assertEqual(len(sampler), 30)
        self.assertEqual(len(sampler.sampling_ratios), 2)
        self.assertAlmostEqual(float(sampler.sampling_ratios[0]), 0.5)

        # Sample across multiple batches to exercise index regeneration
        samples = [sampler[i] for i in range(12)]
        self.assertEqual(len(samples), 12)
        for s in samples:
            self.assertEqual(s[0].shape, (2,))

    def test_sampling_dataset_custom_ratios_and_validation(self):
        """Tests custom sampling ratio enforcement and error handling."""
        ds1 = TensorDataset(torch.ones(5, 1))
        ds2 = TensorDataset(torch.zeros(5, 1))

        # Valid custom ratios
        sampler = SamplingDataset([ds1, ds2], sampling_ratios=[0.8, 0.2], batch_size=8)
        self.assertAlmostEqual(float(sampler.sampling_ratios[0]), 0.8)

        # Mismatched length assertion
        with self.assertRaises(AssertionError):
            SamplingDataset([ds1, ds2], sampling_ratios=[0.5, 0.3, 0.2])

        # Non-unit sum assertion
        with self.assertRaises(AssertionError):
            SamplingDataset([ds1, ds2], sampling_ratios=[0.5, 0.2])

    def test_unnormalize_helpers(self):
        """Tests continuous temperature unnormalization for predictions and variance."""
        mean_val = 2500.0
        std_val = 500.0
        normalized_pred = 1.5

        phys_temp = unnormalize_mean(normalized_pred, mean_val, std_val)
        self.assertAlmostEqual(phys_temp, 3250.0)

        normalized_var = 0.04
        phys_var = unnormalize_var(normalized_var, std_val)
        self.assertAlmostEqual(phys_var, 0.04 * (500.0 ** 2))

    def test_calculate_stats_with_mock_dataset(self):
        """Tests calculate_stats extraction across tabular dataset columns."""
        class MockDataset:
            def __init__(self, df):
                self.df = df
            def with_format(self, fmt):
                return self.df

        df = pd.DataFrame({
            "Altitude": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "Te1": [2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
        })
        mock_ds = MockDataset(df)
        means, stds = calculate_stats(mock_ds, ["Altitude", "Te1"])

        self.assertAlmostEqual(means["Altitude"], 3000.0)
        self.assertAlmostEqual(means["Te1"], 4000.0)
        self.assertAlmostEqual(stds["Altitude"], float(df["Altitude"].std()))

    def test_normalize_ds(self):
        """Tests normalize_ds mapping on dataset."""
        class MockHFMapDataset:
            def __init__(self):
                self.features = ["Altitude", "Te1"]
            def map(self, fn, batched=True, batch_size=10000, num_proc=None):
                batch = {"Altitude": [1000.0, 3000.0], "Te1": [1000.0, 2000.0]}
                return fn(batch)

        mock_ds = MockHFMapDataset()
        means = {"Altitude": 2000.0, "Te1": 1500.0}
        stds = {"Altitude": 500.0, "Te1": 500.0}
        res = normalize_ds(mock_ds, means, stds, ["Altitude"], normalize_output=True)
        self.assertIn("Te1", res)
        self.assertIn("Altitude", res)


if __name__ == "__main__":
    unittest.main()
