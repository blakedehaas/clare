import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import OUTPUT_COLUMNS, DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH, DEFAULT_DATASET_DIR
from inference import load_model, load_normalization_stats, preprocess_dataframe, predict

# --- Configuration ---
DATASET_DIR = os.path.join(DEFAULT_DATASET_DIR, "test-normal")
PLOT_OUTPUT_FILENAME = "test_normal_pred_vs_obs_residual.png"
MODEL_OUTPUT_COLUMN = 'Te1_pred'


def plot_residual_dot_plot(df, output_filename):
    df['Residual'] = df[OUTPUT_COLUMNS[0]] - df[MODEL_OUTPUT_COLUMN]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(df[MODEL_OUTPUT_COLUMN], df['Residual'],
            marker='.', linestyle='none', color='steelblue',
            markersize=4, alpha=0.5)

    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Predicted Electron Temperature [K]")
    ax.set_ylabel("Residual (Observed - Predicted) Electron Temperature [K]")
    ax.set_title("Residual Plot")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Residual plot saved to: {output_filename}")
    plt.show()


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(DEFAULT_CHECKPOINT_PATH, device)
    means, stds = load_normalization_stats(DEFAULT_STATS_PATH)

    dataset = load_from_disk(DATASET_DIR)
    df_original = dataset.to_pandas()

    df_processed = preprocess_dataframe(df_original, means, stds)
    predictions = predict(model, df_processed, device)
    df_original[MODEL_OUTPUT_COLUMN] = predictions

    plot_residual_dot_plot(df_original, PLOT_OUTPUT_FILENAME)
