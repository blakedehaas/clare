"""Unit test suite verifying Bayesian optimization suite, storage backends, and dynamic architectures in sweep.py."""

import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import optuna

from sweep import (
    create_optuna_storage,
    SwiGLUExpert,
    DynamicSpacePhysicsModel,
    SpaceWeatherDataset,
    parse_args
)


class TestSweep(unittest.TestCase):
    """Validates Optuna storage resolvers, MoE neural components, and dataset streaming in sweep.py."""

    def test_create_optuna_storage_backends(self):
        """Tests standard RDBMS vs. Lustre-safe JournalFileStorage storage backend initialization."""
        # 1. Standard SQLite string
        sqlite_spec = "sqlite:///test_sweep.db"
        res_sqlite = create_optuna_storage(sqlite_spec)
        self.assertEqual(res_sqlite, sqlite_spec)

        # 2. Journal file path ending in .log
        log_spec = "tests/test_output/journal_test.log"
        res_journal = create_optuna_storage(log_spec)
        from optuna.storages import JournalStorage
        self.assertIsInstance(res_journal, JournalStorage)

        # 3. Journal URI with journal:// scheme
        uri_spec = "journal://tests/test_output/journal_uri.log"
        res_uri = create_optuna_storage(uri_spec)
        self.assertIsInstance(res_uri, JournalStorage)

    def test_swiglu_expert_forward(self):
        """Tests SwiGLU expert feed-forward transformation with activation clamping."""
        expert = SwiGLUExpert(d_model=32, expert_dim=64, clamp_val=10.0)
        x = torch.randn(4, 32)
        out = expert(x)
        self.assertEqual(out.shape, (4, 32))

    def test_dynamic_space_physics_model_dense(self):
        """Tests DynamicSpacePhysicsModel under standard dense MLP configuration."""
        model = DynamicSpacePhysicsModel(
            input_dim=156,
            output_dim=150,
            d_model=64,
            num_layers=2,
            use_moe=False
        )
        x = torch.randn(3, 156)
        out = model(x)
        self.assertEqual(out.shape, (3, 150))

    def test_dynamic_space_physics_model_moe(self):
        """Tests DynamicSpacePhysicsModel with Mixture-of-Experts routing and top-k selection."""
        model = DynamicSpacePhysicsModel(
            input_dim=156,
            output_dim=150,
            d_model=64,
            num_layers=2,
            num_experts=4,
            top_k=2,
            use_moe=True
        )
        x = torch.randn(3, 156)
        out = model(x)
        self.assertEqual(out.shape, (3, 150))

    def test_space_weather_dataset_with_mock_hf(self):
        """Tests SpaceWeatherDataset feature stacking and target temperature quantization."""
        class MockHFDataset:
            def __init__(self):
                self.column_names = ["Altitude", "GCLAT", "Te1"]
                self.data = {
                    "Altitude": [2000.0, 4000.0],
                    "GCLAT": [30.0, 60.0],
                    "Te1": [2500.0, 4500.0]
                }
            def __getitem__(self, col):
                return self.data[col]

        mock_ds = MockHFDataset()
        dataset = SpaceWeatherDataset(mock_ds)
        self.assertEqual(len(dataset), 2)
        x, y = dataset[0]
        self.assertEqual(x.dim(), 1)
        self.assertEqual(y.item(), 25)  # 2500 // 100 = 25

    def test_space_weather_dataset_with_input_ids(self):
        """Tests SpaceWeatherDataset when pre-extracted input_ids and label columns exist."""
        class MockTokenDataset:
            def __init__(self):
                self.column_names = ["input_ids", "label"]
                self.data = {
                    "input_ids": [[1.0, 2.0], [3.0, 4.0]],
                    "label": [10, 20]
                }
            def __getitem__(self, col):
                return self.data[col]

        mock_ds = MockTokenDataset()
        dataset = SpaceWeatherDataset(mock_ds)
        self.assertEqual(len(dataset), 2)
        x, y = dataset[1]
        self.assertEqual(y.item(), 20)

    def test_sweep_objective(self):
        """Tests Optuna objective evaluation function in sweep.py."""
        from sweep import objective
        import argparse

        args = argparse.Namespace(epochs_per_trial=1)
        study = optuna.create_study()
        trial = study.ask()

        from sweep import INPUT_COLUMNS
        dummy_loader = [(torch.randn(2, len(INPUT_COLUMNS)), torch.randint(0, 150, (2,)))]
        val_loss = objective(
            trial=trial,
            args=args,
            train_loader=dummy_loader,
            val_normal_loader=dummy_loader,
            val_storm_loader=dummy_loader,
            device=torch.device("cpu")
        )
        self.assertIsInstance(val_loss, float)


if __name__ == "__main__":
    unittest.main()
