import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datasets import load_from_disk

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import INPUT_COLUMNS, OUTPUT_COLUMNS, DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH, DEFAULT_DATASET_DIR
from inference import load_model, load_normalization_stats, preprocess_dataframe, predict

# --- Configuration ---
DATASET_DIR = os.path.join(DEFAULT_DATASET_DIR, "test-storm")
PLOT_OUTPUT_FILENAME = "pred_dotplot.png"
MODEL_OUTPUT_COLUMN = 'Te1_pred'
TIME_COLUMN = 'DateTimeFormatted'


def plot_altitude_time_te1_color(df, output_filename):
    """Plots Altitude vs. Time colored by predicted Te1."""
    fig, ax = plt.subplots(figsize=(18, 8))

    df_plot = df.copy()
    df_plot[TIME_COLUMN] = pd.to_datetime(df_plot[TIME_COLUMN])
    df_plot = df_plot.sort_values(by=TIME_COLUMN)

    sc = ax.scatter(
        df_plot[TIME_COLUMN], df_plot['Altitude'],
        c=df_plot[MODEL_OUTPUT_COLUMN], cmap='turbo',
        s=10, alpha=0.8, edgecolor='none'
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Predicted Electron Temperature [K]', fontsize=12)

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Altitude [km]", fontsize=14)
    ax.set_title("Satellite Altitude Profile with Predicted Electron Temperature", fontsize=16)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
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

    df_processed = preprocess_dataframe(df_original.copy(), means, stds)
    predictions = predict(model, df_processed, device)
    df_original[MODEL_OUTPUT_COLUMN] = predictions

    plot_altitude_time_te1_color(df_original, PLOT_OUTPUT_FILENAME)
