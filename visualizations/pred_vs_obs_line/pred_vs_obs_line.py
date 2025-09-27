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
PLOT_OUTPUT_FILENAME = "pred_vs_obs_line.png"
BATCH_SIZE_PREDICT = 1024

# Define input/output columns (must match training)
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
output_columns = ['Te1'] # Target variable
model_output_column = 'Te1_pred'
time_column = 'DateTimeFormatted'

# Model hyperparameters (must match trained model)
input_size = len(input_columns)
hidden_size = 2048
output_size = 150 # Number of classes for Te1 bins

def check_file_exists(filepath, description):
    """Checks if a file exists and exits if not."""
    if not os.path.exists(filepath):
        print(f"Error: {description} not found at '{filepath}'")
        print("Please ensure the file exists and the path is correct.")
        sys.exit(1)
    print(f"Found {description} at: {filepath}")

def load_model(model_path, input_size, hidden_size, output_size, device):
    """Loads the pre-trained model."""
    print(f"Loading model checkpoint from: {model_path}")
    check_file_exists(model_path, "Model checkpoint")
    model = model_definition.FeedForwardNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def load_normalization_stats(stats_path):
    """Loads the normalization means and stds."""
    print(f"Loading normalization stats from: {stats_path}")
    check_file_exists(stats_path, "Normalization stats file")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print("Normalization stats loaded successfully.")
    return stats['mean'], stats['std']

def preprocess_data(df, means, stds):
    """Applies normalization consistent with training."""
    print("Preprocessing data (applying normalizations)...")
    df_processed = df.copy()

    if hasattr(constants, 'NORMALIZATIONS') and isinstance(constants.NORMALIZATIONS, dict):
        for col, norm_func in constants.NORMALIZATIONS.items():
             if col in df_processed.columns:
                # Apply normalization function if column exists
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = norm_func(df_processed[col])
             else:
                 print(f"Warning: Column '{col}' specified in constants.NORMALIZATIONS not found in DataFrame.")

    else:
        print("Warning: constants.NORMALIZATIONS not found or not a dictionary. Skipping this normalization step.")


    # Apply group normalization for solar indices
    index_groups = {
        'AL_index': [col for col in input_columns if col.startswith('AL_index')],
        'SYM_H': [col for col in input_columns if col.startswith('SYM_H')],
        'f107_index': [col for col in input_columns if col.startswith('f107_index')]
    }
    for group_name, group_cols in index_groups.items():
        if group_name in means and group_name in stds:
            group_mean = means[group_name]
            group_std = stds[group_name]
            if group_std == 0:
                 print(f"Warning: Standard deviation for group '{group_name}' is zero. Skipping normalization for this group.")
                 continue
            for col in group_cols:
                if col in df_processed.columns:
                    # Ensure column is numeric before normalization
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    # Apply normalization
                    df_processed[col] = (df_processed[col].astype(np.float32) - group_mean) / group_std
                else:
                     print(f"Warning: Column '{col}' for group '{group_name}' not found in DataFrame.")

        else:
            print(f"Warning: Normalization stats for group '{group_name}' not found. Skipping normalization.")

    # Handle potential NaNs introduced by normalization or missing data
    cols_to_check_for_nan = [col for col in input_columns if col in df_processed.columns]
    nans_before = df_processed[cols_to_check_for_nan].isnull().sum().sum()
    if nans_before > 0:
        print(f"Warning: Found {nans_before} NaN values after normalization. Filling with 0.")
        df_processed[cols_to_check_for_nan] = df_processed[cols_to_check_for_nan].fillna(0)

    print("Data preprocessing complete.")
    return df_processed

def predict_te1(model, df_processed, device):
    """Generates Te1 predictions using the model."""
    print("Generating Te1 predictions...")
    
    # Ensure all required input columns exist after preprocessing
    missing_input_cols = [col for col in input_columns if col not in df_processed.columns]
    if missing_input_cols:
        print(f"Error: Missing required input columns for prediction: {missing_input_cols}")
        sys.exit(1)
        
    input_data = df_processed[input_columns].values.astype(np.float32)
    
    input_tensor = torch.tensor(input_data)
    dataset = TensorDataset(input_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False)
    
    all_predictions = []
    with torch.no_grad(): # Disable gradient calculations for inference
        for batch_tensor, in tqdm(data_loader, desc="Predicting"):
            batch_tensor = batch_tensor.to(device)
            logits = model(batch_tensor)
            
            # Convert logits to predicted temperature value
            predicted_classes = torch.argmax(logits, dim=1)
            # This formula MUST match the binning strategy used during training
            predicted_temps = predicted_classes.cpu().numpy() * 100 + 50
            all_predictions.extend(predicted_temps)
            
    print("Prediction generation complete.")
    return all_predictions

def plot_obs_vs_pred(df, output_filename):
    """Plots observed and predicted Te1 over time with specific styling."""
    print("Generating plot with custom styling...")

    required_plot_cols = [time_column, output_columns[0], model_output_column]
    missing_plot_cols = [col for col in required_plot_cols if col not in df.columns]
    if missing_plot_cols:
         print(f"Error: Missing required columns for plotting: {missing_plot_cols}")
         print(f"Available columns: {df.columns.tolist()}")
         sys.exit(1)

    # --- Data Preparation ---
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        print(f"Converting '{time_column}' column to datetime objects...")
        df[time_column] = pd.to_datetime(df[time_column])

    # Sort by time for a coherent plot
    print("Sorting data by time...")
    df.sort_values(by=time_column, inplace=True)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(18, 8))

    # Plot Observed Te1
    ax.plot(df[time_column], df[output_columns[0]], linestyle='-', linewidth=1, color='blue', label='Obs', marker=None)

    # Plot Predicted Te1
    ax.plot(df[time_column], df[model_output_column], linestyle='-', linewidth=2, color='red', label='Mod', marker=None)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\log_{10}$ Electron Temperature (Te1) [K]")

    ax.set_yscale('log')
    ax.set_ylim(bottom=1000) # Truncate below 10^3

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    plt.gcf().autofmt_xdate()

    # Add '1991' year label to the left (change label manually for appropriate plot styling)
    ax.text(-0.03, -0.1, '1991', transform=ax.transAxes,
            ha='right', va='top', fontsize=plt.rcParams['axes.labelsize'])

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    print(f"Saving plot to: {output_filename}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print("Displaying plot...")
    plt.show()

    print("Visualization complete.")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model and Normalization Stats ---
    model = load_model(CHECKPOINT_PATH, input_size, hidden_size, output_size, device)
    means, stds = load_normalization_stats(STATS_PATH)

    # --- 2. Load Dataset ---
    print(f"Loading dataset from: {DATASET_DIR}")
    check_file_exists(DATASET_DIR, "Dataset directory")
    try:
        dataset = load_from_disk(DATASET_DIR)
        print(f"Dataset loaded successfully. Number of entries: {len(dataset)}")
        df_original = dataset.to_pandas()

        # Data Validation
        required_load_cols = input_columns + output_columns
        if time_column not in df_original.columns:
             print(f"Warning: '{time_column}' column not found.")

        missing_cols = [col for col in required_load_cols + [time_column] if col not in df_original.columns]
        if missing_cols:
            print(f"Error: Missing required columns in the loaded dataset: {missing_cols}")
            print(f"Available columns: {df_original.columns.tolist()}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during dataset loading: {e}")
        sys.exit(1)

    # --- 3. Preprocess Data ---
    df_processed = preprocess_data(df_original, means, stds)

    # --- 4. Generate Predictions ---
    predictions = predict_te1(model, df_processed, device)

    # Add predictions to the original DataFrame
    if len(predictions) == len(df_original):
         df_original[model_output_column] = predictions
    else:
         print(f"Error: Number of predictions ({len(predictions)}) does not match DataFrame length ({len(df_original)}). Cannot add prediction column.")
         sys.exit(1)


    # --- 5. Plot Results ---
    plot_obs_vs_pred(df_original, PLOT_OUTPUT_FILENAME)