"""Plot observed and canonical CLARE-predicted Te through the held-out storm."""

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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PLOT_OUTPUT_FILENAME = "pred_vs_obs_line.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, means, stds = load_model_and_stats(device)

    storm = load_from_disk(str(DATASET_DIR / "test-storm"))
    predicted, observed = predict_clare(model, storm, means, stds, device)

    df = storm.select_columns(["DateTimeFormatted"]).to_pandas()
    df["Te1"] = observed
    df["Te1_pred"] = predicted
    df["DateTimeFormatted"] = pd.to_datetime(df["DateTimeFormatted"])
    df = df.sort_values("DateTimeFormatted")

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(df["DateTimeFormatted"], df["Te1"], linewidth=1, label="Obs")
    ax.plot(df["DateTimeFormatted"], df["Te1_pred"], linewidth=2, label="Mod")
    ax.set_xlabel("Time")
    ax.set_ylabel("Electron Temperature (Te) [K]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1000)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.text(
        -0.03,
        -0.1,
        "1991",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 0.95, 1])

    output = SCRIPT_DIR / PLOT_OUTPUT_FILENAME
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
