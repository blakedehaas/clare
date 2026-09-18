"""Generate canonical CLARE synthetic heatmaps from the shared synthetic input dataset."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

try:
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
except ImportError as exc:
    raise SystemExit("The spacepy package is required for this visualization.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import evaluate as canonical_eval
import models.feed_forward as model_definition

SYNTHETIC_DATA_PATH = SCRIPT_DIR / "synthetic_output_dataset.parquet"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "2_212m_s0_ts0.pth"
STATS_PATH = PROJECT_ROOT / "checkpoints" / "2_212m_s0_norm_stats.json"

BATCH_SIZE_PREDICT = 8192

FIXED_GCLAT = 43.0
FIXED_GCLON = 289.0
RE_KM = 6371.0

PRED_TEMP_COLUMN = "Te1_pred"
CONFIDENCE_COLUMN = "confidence"
SOLAR_FEATURE_COLUMNS = ["Kp_index", "SYM_H_0", "f107_index_0", "AL_index_0"]
KP_INDEX_SCALE_FACTOR = 10.0
CONFIDENCE_VMAX_PERCENT = 50.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l-shell", type=float, default=3.0)
    return parser.parse_args()


def calculate_millstone_anchored_coords(df, l_shell):
    footprint_alt_km = 1000.0

    unique_times = df["DateTimeFormatted"].drop_duplicates()
    ticks_unique = Ticktock(
        unique_times.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "ISO",
    )
    footprint = coord.Coords(
        [[footprint_alt_km, FIXED_GCLAT, FIXED_GCLON]] * len(unique_times),
        "GDZ",
        "sph",
        ticks=ticks_unique,
    )
    footprint_sm = footprint.convert("SM", "sph")
    mlt_unique = (footprint_sm.long / 15.0 + 12.0) % 24.0

    mlt_map = pd.Series(mlt_unique, index=unique_times.values)
    lon_sm_map = pd.Series(footprint_sm.long, index=unique_times.values)

    mlt_all = df["DateTimeFormatted"].map(mlt_map).to_numpy()
    lon_sm_rad = np.deg2rad(
        df["DateTimeFormatted"].map(lon_sm_map).to_numpy()
    )

    radius_re = (df["Altitude"].to_numpy() + RE_KM) / RE_KM
    ratio = radius_re / l_shell
    if np.any(ratio > 1.0):
        raise ValueError(
            f"L={l_shell:g} cannot geometrically contain every requested "
            "altitude in the synthetic grid."
        )

    magnetic_lat_rad = np.arccos(np.sqrt(np.clip(ratio, 0.0, 1.0)))

    x_sm = radius_re * np.cos(magnetic_lat_rad) * np.cos(lon_sm_rad)
    y_sm = radius_re * np.cos(magnetic_lat_rad) * np.sin(lon_sm_rad)
    z_sm = radius_re * np.sin(magnetic_lat_rad)

    ticks_all = Ticktock(
        df["DateTimeFormatted"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "ISO",
    )
    spacecraft_sm = coord.Coords(
        np.column_stack([x_sm, y_sm, z_sm]),
        "SM",
        "car",
        ticks=ticks_all,
    )
    spacecraft_gdz = spacecraft_sm.convert("GDZ", "sph")
    spacecraft_mag = spacecraft_sm.convert("MAG", "sph")

    return pd.DataFrame(
        {
            "GMLT": mlt_all.astype(float),
            "XXLAT": spacecraft_gdz.lati.astype(float),
            "XXLON": spacecraft_gdz.long.astype(float),
            "GLAT": spacecraft_mag.lati.astype(float),
            "GCLAT": np.full(len(df), FIXED_GCLAT, dtype=float),
            "GCLON": np.full(len(df), FIXED_GCLON, dtype=float),
        },
        index=df.index,
    )


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


def predict_batch(model, batch, means, stds, device):
    values = {
        column: batch[column].to_numpy()
        for column in batch.columns
    }
    values["Te1"] = np.zeros(len(batch), dtype=np.float32)

    x, _ = canonical_eval.prepare_features(
        values,
        means,
        stds,
    )

    centers = (
        torch.arange(
            canonical_eval.TE_NUM_CLASSES,
            device=device,
            dtype=torch.float32,
        )
        * canonical_eval.TE_BIN_WIDTH_K
        + 50.0
    )

    temperatures = []
    confidences = []

    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE_PREDICT):
            tensor = torch.from_numpy(
                x[start : start + BATCH_SIZE_PREDICT]
            ).to(device)
            probabilities = torch.softmax(
                model(tensor),
                dim=1,
            )
            temperatures.append(
                (probabilities @ centers).cpu().numpy()
            )
            confidences.append(
                (100.0 * probabilities.max(dim=1).values).cpu().numpy()
            )

    return np.concatenate(temperatures), np.concatenate(confidences)


def load_and_prepare_synthetic_data(l_shell):
    if not SYNTHETIC_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Required synthetic dataset not found: {SYNTHETIC_DATA_PATH}. "
            "Run generate_synthetic_data.py first."
        )

    table = pq.read_table(SYNTHETIC_DATA_PATH)
    df = table.to_pandas()
    df["DateTimeFormatted"] = pd.to_datetime(df["DateTimeFormatted"])

    coords = calculate_millstone_anchored_coords(df, l_shell)
    df = pd.concat([df, coords], axis=1)
    df["ILAT"] = np.rad2deg(
        np.arccos(1.0 / np.sqrt(l_shell))
    )

    missing = [
        column
        for column in canonical_eval.RAW_INPUT_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Synthetic dataset is missing required model inputs: {missing}"
        )

    if df[canonical_eval.RAW_INPUT_COLUMNS].isna().any().any():
        raise ValueError(
            "Synthetic dataset contains missing canonical model inputs."
        )

    return df


def plot_combined_visualization(df, l_shell, output_path):
    available = [
        column
        for column in SOLAR_FEATURE_COLUMNS
        if column in df.columns
    ]

    temp_heatmap = df.pivot(
        index="Altitude",
        columns="DateTimeFormatted",
        values=PRED_TEMP_COLUMN,
    )
    conf_heatmap = df.pivot(
        index="Altitude",
        columns="DateTimeFormatted",
        values=CONFIDENCE_COLUMN,
    )
    line_data = (
        df[["DateTimeFormatted"] + available]
        .drop_duplicates()
        .set_index("DateTimeFormatted")
    )

    num_subplots = 2 + len(available)
    fig, axes = plt.subplots(
        num_subplots,
        1,
        figsize=(20, 6 + 2 * num_subplots),
        sharex=True,
        gridspec_kw={
            "height_ratios": [4, 4] + [1] * len(available)
        },
    )

    fig.suptitle(
        f"CLARE Predictions: Millstone Hill Anchor (L={l_shell:g})",
        fontsize=20,
    )

    temp_image = axes[0].pcolormesh(
        temp_heatmap.columns,
        temp_heatmap.index,
        temp_heatmap.values,
        shading="auto",
        cmap="turbo",
        vmin=0,
    )
    axes[0].set_ylabel("Altitude [km]", fontsize=14)
    axes[0].set_title(
        "Predicted Electron Temperature",
        fontsize=16,
        pad=10,
    )

    conf_image = axes[1].pcolormesh(
        conf_heatmap.columns,
        conf_heatmap.index,
        conf_heatmap.values,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=CONFIDENCE_VMAX_PERCENT,
    )
    axes[1].set_ylabel("Altitude [km]", fontsize=14)
    axes[1].set_title(
        "Peak Model Prediction Confidence",
        fontsize=16,
        pad=10,
    )

    labels = {
        "Kp_index": "Kp Index",
        "SYM_H_0": "SYM-H (nT)",
        "f107_index_0": "F10.7 Index",
        "AL_index_0": "AL Index (nT)",
    }

    for i, feature in enumerate(available):
        ax = axes[2 + i]
        data = (
            line_data[feature] / KP_INDEX_SCALE_FACTOR
            if feature == "Kp_index"
            else line_data[feature]
        )
        ax.plot(line_data.index, data, linewidth=1.5)
        ax.set_ylabel(labels.get(feature, feature), fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(
            line_data.index.min(),
            line_data.index.max(),
        )

    bottom = axes[-1]
    bottom.xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )
    bottom.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )
    bottom.set_xlabel(
        f"Date in {temp_heatmap.columns.min().year}",
        fontsize=14,
    )
    plt.setp(
        bottom.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    fig.subplots_adjust(right=0.90, top=0.94)

    pos_temp = axes[0].get_position()
    cax_temp = fig.add_axes(
        [
            pos_temp.x1 + 0.01,
            pos_temp.y0,
            0.015,
            pos_temp.height,
        ]
    )
    fig.colorbar(
        temp_image,
        cax=cax_temp,
    ).set_label(
        "Temperature [K]",
        fontsize=12,
    )

    pos_conf = axes[1].get_position()
    cax_conf = fig.add_axes(
        [
            pos_conf.x1 + 0.01,
            pos_conf.y0,
            0.015,
            pos_conf.height,
        ]
    )
    cbar_conf = fig.colorbar(
        conf_image,
        cax=cax_conf,
    )
    cbar_conf.set_label(
        "Confidence (%)",
        fontsize=12,
    )

    ticks = cbar_conf.get_ticks()
    tick_labels = [f"{int(value)}" for value in ticks]
    if tick_labels:
        tick_labels[-1] = f"{tick_labels[-1]}+"
    cbar_conf.set_ticks(ticks)
    cbar_conf.set_ticklabels(tick_labels)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    l_shell = args.l_shell

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, means, stds = load_model_and_stats(device)

    df = load_and_prepare_synthetic_data(l_shell)

    temperatures = []
    confidences = []
    for start in tqdm(
        range(0, len(df), BATCH_SIZE_PREDICT),
        desc=f"Predicting L={l_shell:g}",
    ):
        batch = df.iloc[
            start : start + BATCH_SIZE_PREDICT
        ]
        batch_temp, batch_conf = predict_batch(
            model,
            batch,
            means,
            stds,
            device,
        )
        temperatures.append(batch_temp)
        confidences.append(batch_conf)

    df[PRED_TEMP_COLUMN] = np.concatenate(temperatures)
    df[CONFIDENCE_COLUMN] = np.concatenate(confidences)

    output = (
        SCRIPT_DIR
        / f"L{l_shell:.1f}_MillstoneHill_synthetic_visualization.png"
    )
    plot_combined_visualization(
        df,
        l_shell,
        output,
    )


if __name__ == "__main__":
    main()
