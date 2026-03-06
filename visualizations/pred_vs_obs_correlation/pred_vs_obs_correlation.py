import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import OUTPUT_COLUMNS, DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH, DEFAULT_DATASET_DIR
from inference import load_model, load_normalization_stats, preprocess_dataframe, predict

# --- Configuration ---
MODEL_OUTPUT_COLUMN = 'Te1_pred'


def plot_correlation_histogram(df, output_filename, obs_col, pred_col, title, bounds=None):
    """Generates and saves a 2D correlation histogram on a linear scale."""
    y_true = df[obs_col].values
    y_pred = df[pred_col].values

    valid_mask = (y_true >= 0) & (y_pred >= 0)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        print(f"Warning: No valid data points for {output_filename}. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = plt.get_cmap('viridis')
    ax.set_facecolor(cmap(0))

    if bounds:
        data_min, data_max = bounds
    else:
        data_min = min(y_true_valid.min(), y_pred_valid.min())
        data_max = max(y_true_valid.max(), y_pred_valid.max())

    linear_bins = np.linspace(data_min, data_max, 101)
    h = ax.hist2d(y_pred_valid, y_true_valid, bins=linear_bins, cmap=cmap)

    cbar = fig.colorbar(h[3], ax=ax, pad=0.02)
    cbar.set_label('Number of Observations')

    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)
    ax.plot([data_min, data_max], [data_min, data_max], 'r--', linewidth=2)

    ax.set_xlabel("Predicted Electron Temperature [K]")
    ax.set_ylabel("Observed Electron Temperature [K]")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Correlation plot saved to: {output_filename}")
    plt.close(fig)


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(DEFAULT_CHECKPOINT_PATH, device)
    means, stds = load_normalization_stats(DEFAULT_STATS_PATH)

    dataset_configs = [
        {
            "name": "test-normal",
            "dir": os.path.join(DEFAULT_DATASET_DIR, "test-normal"),
            "plot_filename": "test_normal_pred_vs_obs_correlation.png",
            "title": "Correlation for Test Dataset Model Performance"
        },
        {
            "name": "test-storm",
            "dir": os.path.join(DEFAULT_DATASET_DIR, "test-storm"),
            "plot_filename": "test_storm_pred_vs_obs_correlation.png",
            "title": "Correlation for Known Solar Storm (Jan 30 - Feb 7 1991) Model Performance"
        }
    ]

    # Process all datasets
    results_dfs = {}
    for config in dataset_configs:
        print(f"\n--- Processing dataset: {config['name']} ---")
        dataset = load_from_disk(config['dir'])
        df = dataset.to_pandas()

        df_processed = preprocess_dataframe(df, means, stds)
        predictions = predict(model, df_processed, device)
        df[MODEL_OUTPUT_COLUMN] = predictions
        results_dfs[config['name']] = df

    # Determine global plot bounds from test-normal
    df_normal = results_dfs['test-normal']
    y_true = df_normal[OUTPUT_COLUMNS[0]].values
    y_pred = df_normal[MODEL_OUTPUT_COLUMN].values
    valid_mask = (y_true >= 0) & (y_pred >= 0)
    data_min = min(y_true[valid_mask].min(), y_pred[valid_mask].min())
    data_max = max(y_true[valid_mask].max(), y_pred[valid_mask].max())
    plot_bounds = (data_min, data_max)
    print(f"\nGlobal plot bounds: [{data_min:.2f}, {data_max:.2f}]")

    # Generate plots
    for config in dataset_configs:
        print(f"Generating plot for {config['name']}...")
        plot_correlation_histogram(
            df=results_dfs[config['name']],
            output_filename=config['plot_filename'],
            obs_col=OUTPUT_COLUMNS[0],
            pred_col=MODEL_OUTPUT_COLUMN,
            title=config['title'],
            bounds=plot_bounds
        )

    print("\nScript finished successfully.")
