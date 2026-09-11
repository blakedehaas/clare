import numpy as np
import torch

from models.long_context import LongContextTransformer
from train_transformer import ContextDataset, FixedTimeContextDataset


def test_context_masks_query_target_and_resets_at_gaps():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    targets = np.arange(6, dtype=np.float32) * 100
    timestamps = np.array([
        "2020-01-01T00:00", "2020-01-01T00:01", "2020-01-01T00:02",
        "2020-01-01T01:00", "2020-01-01T01:01", "2020-01-01T01:02",
    ], dtype="datetime64[m]")
    dataset = ContextDataset(features, targets, timestamps, context_length=2, max_gap_minutes=5)

    assert len(dataset) == 2
    tokens, label = dataset[0]
    assert tokens.shape == (3, 5)
    assert tokens[-1, -3].item() == 0
    assert tokens[-1, -2].item() == 0
    assert label == 2
    assert dataset[1][0][0, 0].item() == features[3, 0]


def test_transformer_returns_temperature_logits():
    model = LongContextTransformer(input_size=5, context_length=2, hidden_size=16, num_layers=1, num_heads=4)
    assert model(torch.randn(4, 3, 5)).shape == (4, 150)


def test_fixed_time_context_averages_bins_and_marks_missing_bins():
    features = np.array([[2], [4], [8], [16]], dtype=np.float32)
    targets = np.array([100, 300, 500, 700], dtype=np.float32)
    timestamps = np.array([
        "2020-01-01T00:00", "2020-01-01T00:04", "2020-01-01T00:12", "2020-01-01T00:15",
    ], dtype="datetime64[m]")
    dataset = FixedTimeContextDataset(features, targets, timestamps, horizon_hours=1, bin_minutes=5)

    tokens, label = dataset[3]
    assert tokens.shape == (13, 4)
    assert tokens[-4, 0].item() == 3
    assert tokens[-3, -2].item() == 0
    assert tokens[-2, 0].item() == 8
    assert np.isclose(tokens[-4, -3].item(), 200 / 15_000)
    assert tokens[-1, -2].item() == 0
    assert label == 7
