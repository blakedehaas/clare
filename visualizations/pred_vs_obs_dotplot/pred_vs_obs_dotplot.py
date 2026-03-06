import os
import sys
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datasets import load_from_disk

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import INPUT_COLUMNS, OUTPUT_COLUMNS, DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH, DEFAULT_DATASET_DIR
from inference import load_model, load_normalization_stats, preprocess_dataframe, predict

# --- Configuration ---
DATASET_DIR = os.path.join(DEFAULT_DATASET_DIR, "test-storm")
PLOT_OUTPUT_FILENAME = "pred_vs_obs_dotplot.png"
MODEL_OUTPUT_COLUMN = 'Te1_pred'
TIME_COLUMN = 'DateTimeFormatted'
LOCATION_COLUMNS = ['ILAT', 'GLAT', 'GMLT']
MARKER_SIZE = 2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot observed vs. predicted Te1 and location data.")
    parser.add_argument("--start-date", default="1991-01-31", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="1991-02-07", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def plot_results(df, time_col, obs_col, pred_col, loc_cols, output_filename):
    """Plots observed/predicted Te1 and location data using subplots and dots."""
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    df_sorted = df.sort_values(by=time_col)
    time_values = df_sorted[time_col]

    num_subplots = 1 + len(loc_cols)
    fig, axes = plt.subplots(
        nrows=num_subplots, ncols=1, sharex=True,
        figsize=(18, 3 * num_subplots),
        gridspec_kw={'hspace': 0.1}
    )
    if num_subplots == 1:
        axes = [axes]

    ax_te1 = axes[0]
    ax_te1.scatter(time_values, df_sorted[obs_col], s=MARKER_SIZE, color='blue', label='Obs', marker='.')
    ax_te1.scatter(time_values, df_sorted[pred_col], s=MARKER_SIZE, color='red', label='Mod', marker='.')
    ax_te1.set_ylabel(r"$\log_{10}$ Te [K]")
    ax_te1.set_yscale('log')
    ax_te1.set_ylim(bottom=1000)
    ax_te1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax_te1.legend(loc='upper left', markerscale=4)

    for i, loc_col in enumerate(loc_cols):
        ax_loc = axes[i + 1]
        ax_loc.scatter(time_values, df_sorted[loc_col], s=MARKER_SIZE, label=loc_col, marker='.', color='green')
        ax_loc.set_ylabel(loc_col)
        ax_loc.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax_bottom = axes[-1]
    ax_bottom.set_xlabel("Time")
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_bottom.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate()

    date_format = "%Y-%m-%d"
    start_str = time_values.min().strftime(date_format)
    end_str = time_values.max().strftime(date_format)
    fig.suptitle(f"Observed vs Predicted Te and Location Data ({start_str} to {end_str})", y=0.99)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    print(f"Saving plot to: {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import torch

    args = parse_arguments()
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') + timedelta(days=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(DEFAULT_CHECKPOINT_PATH, device)
    means, stds = load_normalization_stats(DEFAULT_STATS_PATH)

    print(f"Loading dataset from: {DATASET_DIR}")
    dataset = load_from_disk(DATASET_DIR)
    print(f"Dataset loaded. Total entries: {len(dataset)}")
    df_full = dataset.to_pandas()

    if not pd.api.types.is_datetime64_any_dtype(df_full[TIME_COLUMN]):
        df_full[TIME_COLUMN] = pd.to_datetime(df_full[TIME_COLUMN])

    df_filtered = df_full[
        (df_full[TIME_COLUMN] >= start_date) & (df_full[TIME_COLUMN] < end_date)
    ].copy()
    print(f"Entries in selected period: {len(df_filtered)}")

    if df_filtered.empty:
        print("No data found for the specified time period. Exiting.")
        sys.exit(0)

    df_processed = preprocess_dataframe(df_filtered, means, stds)
    predictions = predict(model, df_processed, device)
    df_filtered.loc[:, MODEL_OUTPUT_COLUMN] = predictions

    plot_results(
        df=df_filtered, time_col=TIME_COLUMN,
        obs_col=OUTPUT_COLUMNS[0], pred_col=MODEL_OUTPUT_COLUMN,
        loc_cols=LOCATION_COLUMNS, output_filename=PLOT_OUTPUT_FILENAME
    )
