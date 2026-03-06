import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datasets import load_from_disk

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import OUTPUT_COLUMNS, DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH, DEFAULT_DATASET_DIR
from inference import load_model, load_normalization_stats, preprocess_dataframe, predict

# --- Configuration ---
DATASET_DIR = os.path.join(DEFAULT_DATASET_DIR, "test-storm")
PLOT_OUTPUT_FILENAME = "pred_vs_obs_line.png"
MODEL_OUTPUT_COLUMN = 'Te1_pred'
TIME_COLUMN = 'DateTimeFormatted'


def plot_obs_vs_pred(df, output_filename):
    """Plots observed and predicted Te1 over time with line styling."""
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COLUMN]):
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    df.sort_values(by=TIME_COLUMN, inplace=True)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(df[TIME_COLUMN], df[OUTPUT_COLUMNS[0]], linestyle='-', linewidth=1, color='blue', label='Obs')
    ax.plot(df[TIME_COLUMN], df[MODEL_OUTPUT_COLUMN], linestyle='-', linewidth=2, color='red', label='Mod')

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\log_{10}$ Electron Temperature (Te1) [K]")
    ax.set_yscale('log')
    ax.set_ylim(bottom=1000)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.gcf().autofmt_xdate()

    ax.text(-0.03, -0.1, '1991', transform=ax.transAxes,
            ha='right', va='top', fontsize=plt.rcParams['axes.labelsize'])

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    print(f"Saving plot to: {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(DEFAULT_CHECKPOINT_PATH, device)
    means, stds = load_normalization_stats(DEFAULT_STATS_PATH)

    print(f"Loading dataset from: {DATASET_DIR}")
    dataset = load_from_disk(DATASET_DIR)
    print(f"Dataset loaded. Number of entries: {len(dataset)}")
    df_original = dataset.to_pandas()

    df_processed = preprocess_dataframe(df_original, means, stds)
    predictions = predict(model, df_processed, device)
    df_original[MODEL_OUTPUT_COLUMN] = predictions

    plot_obs_vs_pred(df_original, PLOT_OUTPUT_FILENAME)
