"""Plot storm observations, canonical CLARE predictions, and spacecraft context."""

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

import argparse
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset

PLOT_OUTPUT_FILENAME = "pred_vs_obs_dotplot.png"
MARKER_SIZE = 2
LOCATION_COLUMNS = ["ILAT", "GLAT", "GMLT"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="1991-01-31")
    parser.add_argument("--end-date", default="1991-02-07")
    return parser.parse_args()


def main():
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d") + timedelta(days=1)

    storm = load_from_disk(str(DATASET_DIR / "test-storm"))
    df = storm.to_pandas()
    df["DateTimeFormatted"] = pd.to_datetime(df["DateTimeFormatted"])
    df = df[
        (df["DateTimeFormatted"] >= start)
        & (df["DateTimeFormatted"] < end)
    ].copy()
    if df.empty:
        raise ValueError("No held-out storm rows fall inside the requested date range.")

    filtered_dataset = Dataset.from_pandas(df, preserve_index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, means, stds = load_model_and_stats(device)
    predicted, observed = predict_clare(
        model,
        filtered_dataset,
        means,
        stds,
        device,
    )
    df["Te1"] = observed
    df["Te1_pred"] = predicted
    df = df.sort_values("DateTimeFormatted")

    fig, axes = plt.subplots(
        nrows=1 + len(LOCATION_COLUMNS),
        ncols=1,
        sharex=True,
        figsize=(18, 12),
        gridspec_kw={"hspace": 0.1},
    )

    axes[0].scatter(
        df["DateTimeFormatted"],
        df["Te1"],
        s=MARKER_SIZE,
        label="Obs",
        marker=".",
    )
    axes[0].scatter(
        df["DateTimeFormatted"],
        df["Te1_pred"],
        s=MARKER_SIZE,
        label="Mod",
        marker=".",
    )
    axes[0].set_ylabel("Te [K]")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1000)
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].legend(loc="upper left", markerscale=4)

    for i, column in enumerate(LOCATION_COLUMNS, start=1):
        axes[i].scatter(
            df["DateTimeFormatted"],
            df[column],
            s=MARKER_SIZE,
            marker=".",
        )
        axes[i].set_ylabel(column)
        axes[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate()

    start_str = df["DateTimeFormatted"].min().strftime("%Y-%m-%d")
    end_str = df["DateTimeFormatted"].max().strftime("%Y-%m-%d")
    fig.suptitle(
        f"Observed vs Predicted Te and Location Data ({start_str} to {end_str})",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output = SCRIPT_DIR / PLOT_OUTPUT_FILENAME
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
