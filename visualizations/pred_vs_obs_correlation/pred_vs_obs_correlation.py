"""Generate validation and storm observed-vs-predicted correlation figures for canonical CLARE."""

from pathlib import Path
import json
import sys

import numpy as np
import torch
from datasets import load_from_disk

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import evaluate as canonical_eval
import models.feed_forward as model_definition

DATASET_DIR = PROJECT_ROOT / "dataset" / "processed_dataset_blocksplit_212m_s0"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "2_212m_s0_ts0.pth"
STATS_PATH = PROJECT_ROOT / "checkpoints" / "2_212m_s0_norm_stats.json"
BATCH_SIZE_PREDICT = 2048


def load_model_and_stats(device):
    state = canonical_eval.load_state_dict(CHECKPOINT_PATH)
    model = model_definition.FeedForwardNetwork(
        len(canonical_eval.INPUT_COLUMNS),
        2048,
        canonical_eval.TE_NUM_CLASSES,
    )
    model.load_state_dict(state)
    model.to(device).eval()

    with STATS_PATH.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return model, stats["mean"], stats["std"]


def predict_clare(model, dataset, means, stds, device):
    centers = (
        torch.arange(
            canonical_eval.TE_NUM_CLASSES,
            device=device,
            dtype=torch.float32,
        )
        * canonical_eval.TE_BIN_WIDTH_K
        + 50.0
    )

    predictions = []
    observations = []
    with torch.no_grad():
        for batch in dataset.iter(batch_size=BATCH_SIZE_PREDICT):
            x, y = canonical_eval.prepare_features(batch, means, stds)
            logits = model(torch.from_numpy(x).to(device))
            probabilities = torch.softmax(logits, dim=1)
            predictions.append((probabilities @ centers).cpu().numpy())
            observations.append(y)

    return np.concatenate(predictions), np.concatenate(observations)

import matplotlib.pyplot as plt


def plot_correlation(observed, predicted, output_filename, title, bounds):
    valid = np.isfinite(observed) & np.isfinite(predicted) & (observed >= 0) & (predicted >= 0)
    observed = observed[valid]
    predicted = predicted[valid]

    data_min, data_max = bounds
    bins = np.linspace(data_min, data_max, 101)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = plt.get_cmap("viridis")
    ax.set_facecolor(cmap(0))
    hist = ax.hist2d(predicted, observed, bins=bins, cmap=cmap)
    cbar = fig.colorbar(hist[3], ax=ax, pad=0.02)
    cbar.set_label("Number of Observations")

    ax.plot([data_min, data_max], [data_min, data_max], "r--", linewidth=2)
    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)
    ax.set_xlabel("Predicted Electron Temperature [K]")
    ax.set_ylabel("Observed Electron Temperature [K]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    output = SCRIPT_DIR / output_filename
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, means, stds = load_model_and_stats(device)

    configs = [
        (
            "validation",
            DATASET_DIR / "val-blocks",
            "validation_pred_vs_obs_correlation.png",
            "Correlation for Validation Dataset Model Performance",
        ),
        (
            "test-storm",
            DATASET_DIR / "test-storm",
            "test_storm_pred_vs_obs_correlation.png",
            "Correlation for Held-Out Solar Storm (Jan 31 - Feb 7 1991)",
        ),
    ]

    results = {}
    for name, path, _, _ in configs:
        dataset = load_from_disk(str(path))
        predicted, observed = predict_clare(model, dataset, means, stds, device)
        results[name] = (observed, predicted)

    validation_observed, validation_predicted = results["validation"]
    valid = (
        np.isfinite(validation_observed)
        & np.isfinite(validation_predicted)
        & (validation_observed >= 0)
        & (validation_predicted >= 0)
    )
    bounds = (
        float(min(validation_observed[valid].min(), validation_predicted[valid].min())),
        float(max(validation_observed[valid].max(), validation_predicted[valid].max())),
    )

    for name, _, filename, title in configs:
        observed, predicted = results[name]
        plot_correlation(observed, predicted, filename, title, bounds)


if __name__ == "__main__":
    main()
