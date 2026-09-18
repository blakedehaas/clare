"""Generate the canonical CLARE validation residual figure."""

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

PLOT_OUTPUT_FILENAME = "validation_pred_vs_obs_residual.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, means, stds = load_model_and_stats(device)

    validation = load_from_disk(str(DATASET_DIR / "val-blocks"))
    predicted, observed = predict_clare(model, validation, means, stds, device)
    residual = observed - predicted

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        predicted,
        residual,
        marker=".",
        linestyle="none",
        markersize=4,
        alpha=0.5,
    )
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Electron Temperature [K]")
    ax.set_ylabel("Residual (Observed - Predicted) Electron Temperature [K]")
    ax.set_title("Residual Plot for Canonical CLARE Validation Predictions")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    output = SCRIPT_DIR / PLOT_OUTPUT_FILENAME
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
