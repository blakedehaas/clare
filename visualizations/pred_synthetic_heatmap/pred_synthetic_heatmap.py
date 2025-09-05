import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add top-level directory to path for custom imports (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import models.feed_forward as model_definition
import constants

# --- Configuration ---
SYNTHETIC_DATA_PATH = "synthetic_data_for_heatmap.parquet"
CHECKPOINT_PATH = os.path.join("../../", "checkpoints", "checkpoint.pth")
STATS_PATH = os.path.join("../../", "checkpoints", "norm_stats.json")
PLOT_OUTPUT_FILENAME = "pred_sythetic_heatmap.png"
BATCH_SIZE_PREDICT = 4096

# Define input/output columns (must match training)
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
model_output_column = 'Te1_pred'

# Model hyperparameters (must match trained model)
input_size = len(input_columns)
hidden_size = 2048
output_size = 150 # Number of classes for Te1 bins

# --- Reused Functions from your Original Script ---
def check_file_exists(filepath, description):
    if not os.path.exists(filepath):
        print(f"Error: {description} not found at '{filepath}'")
        sys.exit(1)
    print(f"Found {description} at: {filepath}")

def load_model(model_path, device):
    print(f"Loading model checkpoint from: {model_path}")
    check_file_exists(model_path, "Model checkpoint")
    model = model_definition.FeedForwardNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def load_normalization_stats(stats_path):
    check_file_exists(stats_path, "Normalization stats file")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']

def preprocess_data(df, means, stds):
    df_processed = df.copy()
    if hasattr(constants, 'NORMALIZATIONS'):
        for col, norm_func in constants.NORMALIZATIONS.items():
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').apply(norm_func)
    index_groups = {
        'AL_index': [c for c in input_columns if c.startswith('AL_index')],
        'SYM_H': [c for c in input_columns if c.startswith('SYM_H')],
        'f107_index': [c for c in input_columns if c.startswith('f107_index')]
    }
    for group, cols in index_groups.items():
        if group in means and group in stds and stds[group] != 0:
            df_processed[cols] = (df_processed[cols] - means[group]) / stds[group]
    df_processed[input_columns] = df_processed[input_columns].fillna(0)
    return df_processed

def predict_te1(model, df_processed, device):
    input_tensor = torch.tensor(df_processed[input_columns].values.astype(np.float32))
    dataset = TensorDataset(input_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch_tensor, in tqdm(data_loader, desc="Predicting Te1"):
            logits = model(batch_tensor.to(device))
            predicted_classes = torch.argmax(logits, dim=1)
            predicted_temps = predicted_classes.cpu().numpy() * 100 + 50
            predictions.extend(predicted_temps)
    return predictions

# --- New Heatmap Plotting Function ---
def plot_heatmap(df, output_filename):
    """
    Creates and saves a heatmap of predicted Te1 vs. Altitude and Time.
    """
    print("Generating heatmap visualization...")
    
    # Pivot the data to create a 2D grid for the heatmap
    # Index (Y-axis): Altitude, Columns (X-axis): Time, Values: Predicted Te1
    heatmap_data = df.pivot(
        index='Altitude', 
        columns='DateTimeFormatted', 
        values=model_output_column
    )

    # Get the axes labels from the pivoted data
    time_labels = heatmap_data.columns
    altitude_labels = heatmap_data.index
    
    fig, ax = plt.subplots(figsize=(18, 8))

    # Use pcolormesh for accurate grid plotting
    # We use LogNorm for the color scale if Te1 values vary over orders of magnitude
    im = ax.pcolormesh(
        time_labels, 
        altitude_labels, 
        heatmap_data.values, 
        shading='auto',
        cmap='turbo' # A colormap similar to the example image (blue-green-yellow-red)
    )

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Predicted Electron Temperature [K]', fontsize=12)

    # Set labels and title
    ax.set_title('Predicted Electron Temperature Heatmap', fontsize=16)
    ax.set_xlabel('Date in 1991', fontsize=14)
    ax.set_ylabel('Altitude [km]', fontsize=14)
    
    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Set y-axis limits
    ax.set_ylim(min(altitude_labels), max(altitude_labels))

    plt.tight_layout()
    
    # Save and show the plot
    print(f"Saving heatmap to: {output_filename}")
    plt.savefig(output_filename, dpi=300)
    plt.show()

# --- Main Execution Logic ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model and Stats
    model = load_model(CHECKPOINT_PATH, device)
    means, stds = load_normalization_stats(STATS_PATH)

    # 2. Load the SYNTHETIC dataset
    print(f"Loading synthetic dataset from: {SYNTHETIC_DATA_PATH}")
    check_file_exists(SYNTHETIC_DATA_PATH, "Synthetic dataset")
    df_synthetic = pd.read_parquet(SYNTHETIC_DATA_PATH)
    df_synthetic['DateTimeFormatted'] = pd.to_datetime(df_synthetic['DateTimeFormatted'])

    # 3. Preprocess the synthetic data
    print("Preprocessing synthetic data...")
    df_processed = preprocess_data(df_synthetic, means, stds)
    
    # 4. Generate predictions
    print("Generating predictions for synthetic data...")
    predictions = predict_te1(model, df_processed, device)
    df_synthetic[model_output_column] = predictions

    # 5. Plot the heatmap
    plot_heatmap(df_synthetic, PLOT_OUTPUT_FILENAME)