import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add top-level directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import models.feed_forward as model_definition
import constants

# --- Configuration ---
DATASET_DIR = os.path.join("../../", "dataset", "processed_dataset_01_31_storm", "test-storm")
MODEL_NAME = '1_47'
CHECKPOINT_PATH = os.path.join("../../", "checkpoints", "checkpoint.pth")
STATS_PATH = os.path.join("../../", "checkpoints", "norm_stats.json")
PLOT_OUTPUT_FILENAME = "pred_vs_obs_dotplot.png"
BATCH_SIZE_PREDICT = 1024

# Define input/output columns (must match training)
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
output_columns = ['Te1'] # Target variable
model_output_column = 'Te1_pred'
time_column = 'DateTimeFormatted'
location_columns_to_plot = ['ILAT', 'GLAT', 'GMLT'] # Columns for subplotting

# Model hyperparameters (must match trained model)
input_size = len(input_columns)
hidden_size = 2048
output_size = 150 # Number of classes for Te1 bins
MARKER_SIZE = 2

# --- Utility Functions ---

def check_path_exists(path, is_dir=False, description="Path"):
    """Checks if a file or directory exists and exits if not."""
    exists = os.path.isdir(path) if is_dir else os.path.exists(path)
    path_type = "Directory" if is_dir else "File"
    if not exists:
        print(f"Error: {description} ({path_type}) not found at '{path}'")
        print("Please ensure the path is correct.")
        sys.exit(1)
    print(f"Found {description} at: {path}")

def parse_arguments():
    """Parses command-line arguments for start and end dates."""
    parser = argparse.ArgumentParser(description="Plot observed vs. predicted Te1 and location data for a specified time period.")
    parser.add_argument(
        "--start-date",
        default="1991-01-31",
        help="Start date for the plot period (YYYY-MM-DD). Default: 1991-01-31"
    )
    parser.add_argument(
        "--end-date",
        default="1991-02-07",
        help="End date for the plot period (YYYY-MM-DD). Default: 1991-02-07"
    )
    return parser.parse_args()

def validate_date_string(date_str):
    """Validates and converts a date string to a datetime object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD.")
        sys.exit(1)

# --- Core Functions ---

def load_model(model_path, input_size, hidden_size, output_size, device):
    """Loads the pre-trained model."""
    print(f"Loading model checkpoint from: {model_path}")
    check_path_exists(model_path, is_dir=False, description="Model checkpoint")
    model = model_definition.FeedForwardNetwork(input_size, hidden_size, output_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
         print(f"Error: Model checkpoint file not found at {model_path}")
         sys.exit(1)
    except Exception as e:
         print(f"Error loading model state_dict: {e}")
         sys.exit(1)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def load_normalization_stats(stats_path):
    """Loads the normalization means and stds."""
    print(f"Loading normalization stats from: {stats_path}")
    check_path_exists(stats_path, is_dir=False, description="Normalization stats file")
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        if 'mean' not in stats or 'std' not in stats:
             raise KeyError("Stats file must contain 'mean' and 'std' keys.")
        print("Normalization stats loaded successfully.")
        return stats['mean'], stats['std']
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {stats_path}")
         sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in normalization stats file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading normalization stats: {e}")
        sys.exit(1)


def preprocess_data(df: pd.DataFrame, means: dict, stds: dict) -> pd.DataFrame:
    """Applies normalization consistent with training."""
    print("Preprocessing data (applying normalizations)...")
    df_processed = df.copy()

    # Apply custom normalizations from constants file
    if hasattr(constants, 'NORMALIZATIONS') and isinstance(constants.NORMALIZATIONS, dict):
        for col, norm_func in constants.NORMALIZATIONS.items():
             if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = norm_func(df_processed[col])
             else: # Warning can be noisy if constants file has many unused entries
                 print(f"Info: Column '{col}' in constants.NORMALIZATIONS not found in current DataFrame slice.")
    else:
        print("Info: constants.NORMALIZATIONS not defined or not a dict. Skipping custom function normalization.")

    # Apply group normalization for solar indices
    index_groups = {
        'AL_index': [col for col in input_columns if col.startswith('AL_index')],
        'SYM_H': [col for col in input_columns if col.startswith('SYM_H')],
        'f107_index': [col for col in input_columns if col.startswith('f107_index')]
    }
    for group_name, group_cols in index_groups.items():
        relevant_group_cols = [col for col in group_cols if col in df_processed.columns]
        if not relevant_group_cols:
            print(f"Info: No columns for group '{group_name}' found in current DataFrame slice.")
            continue # Skip if none of the group's columns are present

        if group_name in means and group_name in stds:
            group_mean = means[group_name]
            group_std = stds[group_name]
            if group_std is None or group_std == 0:
                 print(f"Warning: Standard deviation for group '{group_name}' is zero or None. Skipping normalization for this group.")
                 continue
            for col in relevant_group_cols:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = (df_processed[col].astype(np.float32) - group_mean) / group_std
        else:
            print(f"Warning: Normalization stats for group '{group_name}' not found. Skipping normalization.")

    # Apply individual normalization for remaining input columns
    individual_norm_cols = [
        col for col in input_columns
        if col in df_processed.columns and
           col not in [item for sublist in index_groups.values() for item in sublist] and # Not in groups
           (not hasattr(constants, 'NORMALIZATIONS') or col not in constants.NORMALIZATIONS) # Not handled by custom func
    ]

    for col in individual_norm_cols:
         if col in means and col in stds:
             col_mean = means[col]
             col_std = stds[col]
             if col_std is None or col_std == 0:
                  print(f"Warning: Standard deviation for column '{col}' is zero or None. Skipping normalization.")
                  continue
             df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
             df_processed[col] = (df_processed[col].astype(np.float32) - col_mean) / col_std
         else:
             # This might be okay if some inputs aren't normalized, but warn just in case.
             print(f"Warning: Normalization stats for individual column '{col}' not found. Skipping normalization.")


    # Handle potential NaNs introduced by normalization or missing data
    cols_to_check_for_nan = [col for col in input_columns if col in df_processed.columns]
    nan_counts = df_processed[cols_to_check_for_nan].isnull().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"Warning: Found {total_nans} NaN values after normalization. Filling with 0.")
        print("NaN counts per column:\n", nan_counts[nan_counts > 0])
        df_processed[cols_to_check_for_nan] = df_processed[cols_to_check_for_nan].fillna(0)

    print("Data preprocessing complete.")
    return df_processed

def predict_te1(model, df_processed: pd.DataFrame, device) -> np.ndarray:
    """Generates Te1 predictions using the model."""
    print("Generating Te1 predictions...")

    # Ensure all required input columns exist AFTER preprocessing
    missing_input_cols = [col for col in input_columns if col not in df_processed.columns]
    if missing_input_cols:
        print(f"Error: Missing required input columns for prediction after preprocessing: {missing_input_cols}")
        print(f"Available columns in processed data: {df_processed.columns.tolist()}")
        sys.exit(1)

    input_data = df_processed[input_columns].values.astype(np.float32)

    input_tensor = torch.tensor(input_data)
    dataset = TensorDataset(input_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False)

    all_predictions = []
    with torch.no_grad(): # Disable gradient calculations for inference
        for batch_tensor, in tqdm(data_loader, desc="Predicting", unit="batch"):
            batch_tensor = batch_tensor.to(device)
            logits = model(batch_tensor)

            # Convert logits to predicted temperature value
            predicted_classes = torch.argmax(logits, dim=1)
            # This formula MUST match the binning strategy used during training
            predicted_temps = predicted_classes.cpu().numpy() * 100 + 50
            all_predictions.extend(predicted_temps)

    print("Prediction generation complete.")
    return np.array(all_predictions)


def plot_results(
    df: pd.DataFrame,
    time_col: str,
    obs_col: str,
    pred_col: str,
    loc_cols: list,
    output_filename: str
):
    """Plots observed/predicted Te1 and location data using subplots and dots."""
    print("Generating plot...")

    # --- Data Validation ---
    required_cols = [time_col, obs_col, pred_col] + loc_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
         print(f"Error: Missing required columns for plotting: {missing_cols}")
         print(f"Available columns in DataFrame: {df.columns.tolist()}")
         sys.exit(1)
    if df.empty:
        print("Error: DataFrame is empty. Cannot generate plot (check date range?).")
        sys.exit(1)

    # --- Data Preparation ---
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        print(f"Converting '{time_col}' column to datetime objects...")
        df[time_col] = pd.to_datetime(df[time_col])

    # Sort by time for a coherent plot
    print("Sorting data by time...")
    df_sorted = df.sort_values(by=time_col)
    time_values = df_sorted[time_col]

    # --- Plotting ---
    num_subplots = 1 + len(loc_cols)
    # Adjust figsize height based on the number of subplots
    fig, axes = plt.subplots(
        nrows=num_subplots,
        ncols=1,
        sharex=True, # Link x-axes
        figsize=(18, 3 * num_subplots), # Height proportional to number of plots
        gridspec_kw={'hspace': 0.1} # Reduce vertical space between plots
    )

    # Ensure axes is always an array, even if num_subplots is 1
    if num_subplots == 1:
        axes = [axes]

    # Subplot 1: Observed vs Predicted Te1 (Using dots, no lines)
    ax_te1 = axes[0]
    ax_te1.scatter(time_values, df_sorted[obs_col], s=MARKER_SIZE, color='blue', label='Obs', marker='.')
    ax_te1.scatter(time_values, df_sorted[pred_col], s=MARKER_SIZE, color='red', label='Mod', marker='.')
    ax_te1.set_ylabel(r"$\log_{10}$ Te [K]")
    ax_te1.set_yscale('log')
    ax_te1.set_ylim(bottom=1000) # Truncate y-axis below 10^3 K
    ax_te1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax_te1.legend(loc='upper left', markerscale=4) # Increase marker size in legend

    # Subplots 2 onwards: Location Data
    for i, loc_col in enumerate(loc_cols):
        ax_loc = axes[i + 1]
        ax_loc.scatter(time_values, df_sorted[loc_col], s=MARKER_SIZE, label=loc_col, marker='.', color='green')
        ax_loc.set_ylabel(loc_col)
        ax_loc.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # ax_loc.legend(loc='upper left', markerscale=4) # Optional: legend per subplot

    # --- X-axis Formatting (applied to the last subplot) ---
    ax_bottom = axes[-1]
    ax_bottom.set_xlabel("Time")
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_bottom.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate() # Auto-rotate date labels

    # Add figure title (optional)
    date_format = "%Y-%m-%d"
    start_str = time_values.min().strftime(date_format)
    end_str = time_values.max().strftime(date_format)
    fig.suptitle(f"Observed vs Predicted Te and Location Data ({start_str} to {end_str})", y=0.99)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly for suptitle

    # --- Save and Show ---
    print(f"Saving plot to: {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print("Displaying plot...")
    plt.show()
    print("Visualization complete.")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()
    start_date = validate_date_string(args.start_date)
    # Add one day to end date to make the range inclusive
    end_date = validate_date_string(args.end_date) + timedelta(days=1)

    print(f"Selected time period: {start_date.strftime('%Y-%m-%d')} to {args.end_date}") # Show inclusive end date user provided

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model and Normalization Stats ---
    model = load_model(CHECKPOINT_PATH, input_size, hidden_size, output_size, device)
    means, stds = load_normalization_stats(STATS_PATH)

    # --- 2. Load Dataset ---
    print(f"Loading dataset from: {DATASET_DIR}")
    check_path_exists(DATASET_DIR, is_dir=True, description="Dataset directory")
    try:
        dataset = load_from_disk(DATASET_DIR)
        print(f"Dataset loaded successfully. Total entries: {len(dataset)}")
        df_full = dataset.to_pandas()

        # Data Validation
        if time_column not in df_full.columns:
             print(f"Error: Time column '{time_column}' not found in the dataset.")
             sys.exit(1)

        if not pd.api.types.is_datetime64_any_dtype(df_full[time_column]):
             print(f"Converting '{time_column}' column to datetime objects...")
             df_full[time_column] = pd.to_datetime(df_full[time_column])

        required_cols_load = set(input_columns + output_columns + location_columns_to_plot)
        missing_cols = required_cols_load - set(df_full.columns)
        if missing_cols:
            print(f"Warning: Missing expected columns in the loaded dataset: {missing_cols}")

    except Exception as e:
        print(f"An error occurred during dataset loading: {e}")
        sys.exit(1)

    # --- 3. Filter Data by Time Period ---
    print(f"Filtering data between {start_date} and {end_date} (exclusive end)...")
    df_filtered = df_full[
        (df_full[time_column] >= start_date) &
        (df_full[time_column] < end_date)
    ].copy()
    print(f"Number of entries in selected period: {len(df_filtered)}")

    if df_filtered.empty:
         print("No data found for the specified time period. Exiting.")
         sys.exit(0)

    # --- 4. Preprocess Filtered Data ---
    # Pass only the filtered data to preprocessing
    df_processed = preprocess_data(df_filtered, means, stds)

    # --- 5. Generate Predictions ---
    # Generate predictions only for the filtered data
    predictions = predict_te1(model, df_processed, device)

    # --- 6. Add Predictions to the Filtered DataFrame ---
    # Add predictions back to the filtered DataFrame for plotting
    if len(predictions) == len(df_filtered):
         df_filtered.loc[:, model_output_column] = predictions
    else:
         print(f"Error: Number of predictions ({len(predictions)}) does not match filtered DataFrame length ({len(df_filtered)}). Cannot add prediction column.")
         sys.exit(1)

    # --- 7. Plot Results ---
    # Pass the filtered DataFrame with predictions to the plotting function
    plot_results(
        df=df_filtered,
        time_col=time_column,
        obs_col=output_columns[0],
        pred_col=model_output_column,
        loc_cols=location_columns_to_plot,
        output_filename=PLOT_OUTPUT_FILENAME
    )

    print("Script finished successfully.")