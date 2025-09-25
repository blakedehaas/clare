import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import models.feed_forward as model_definition
import constants

# --- Configuration ---
DATASET_DIR = os.path.join("../../", "dataset", "processed_dataset_01_31_storm", "test-normal")
MODEL_NAME = '1_47'
CHECKPOINT_PATH = os.path.join("../../", "checkpoints", "checkpoint.pth")
STATS_PATH = os.path.join("../../", "checkpoints", "norm_stats.json")
PLOT_OUTPUT_FILENAME = f"test_normal_pred_vs_obs_residual.png"
BATCH_SIZE_PREDICT = 1024

input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
output_columns = ['Te1']
model_output_column = 'Te1_pred'

input_size = len(input_columns)
hidden_size = 2048
output_size = 150

def check_file_exists(filepath, description):
    if not os.path.exists(filepath):
        print(f"Error: {description} not found at '{filepath}'")
        sys.exit(1)

def load_model(model_path, input_size, hidden_size, output_size, device):
    check_file_exists(model_path, "Model checkpoint")
    model = model_definition.FeedForwardNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    if hasattr(constants, 'NORMALIZATIONS') and isinstance(constants.NORMALIZATIONS, dict):
        for col, norm_func in constants.NORMALIZATIONS.items():
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = norm_func(df_processed[col])

    index_groups = {
        'AL_index': [col for col in input_columns if col.startswith('AL_index')],
        'SYM_H': [col for col in input_columns if col.startswith('SYM_H')],
        'f107_index': [col for col in input_columns if col.startswith('f107_index')]
    }
    for group_name, group_cols in index_groups.items():
        if group_name in means and group_name in stds and stds[group_name] != 0:
            group_mean, group_std = means[group_name], stds[group_name]
            for col in group_cols:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = (df_processed[col].astype(np.float32) - group_mean) / group_std

    cols_to_check_for_nan = [col for col in input_columns if col in df_processed.columns]
    if df_processed[cols_to_check_for_nan].isnull().sum().sum() > 0:
        df_processed[cols_to_check_for_nan] = df_processed[cols_to_check_for_nan].fillna(0)
    return df_processed

def predict_te1(model, df_processed, device):
    input_data = df_processed[input_columns].values.astype(np.float32)
    dataset = TensorDataset(torch.tensor(input_data))
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False)
    
    all_predictions = []
    with torch.no_grad():
        for batch_tensor, in tqdm(data_loader, desc="Predicting"):
            logits = model(batch_tensor.to(device))
            predicted_classes = torch.argmax(logits, dim=1)
            predicted_temps = predicted_classes.cpu().numpy() * 100 + 50
            all_predictions.extend(predicted_temps)
    return all_predictions

def plot_residual_dot_plot(df, output_filename):
    df['Residual'] = df[output_columns[0]] - df[model_output_column]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(df[model_output_column], df['Residual'],
            marker='.',
            linestyle='none',
            color='steelblue',
            markersize=4,
            alpha=0.5)

    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)

    ax.set_xlabel("Predicted Electron Temperature [K]")
    ax.set_ylabel("Residual (Observed - Predicted) Electron Temperature [K]")
    ax.set_title(f"Residual Plot for Model {MODEL_NAME}")

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Residual plot saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(CHECKPOINT_PATH, input_size, hidden_size, output_size, device)
    means, stds = load_normalization_stats(STATS_PATH)

    check_file_exists(DATASET_DIR, "Dataset directory")
    dataset = load_from_disk(DATASET_DIR)
    df_original = dataset.to_pandas()

    df_processed = preprocess_data(df_original, means, stds)
    predictions = predict_te1(model, df_processed, device)

    if len(predictions) == len(df_original):
        df_original[model_output_column] = predictions
    else:
        print(f"Error: Prediction length mismatch. Predictions: {len(predictions)}, Data: {len(df_original)}.")
        sys.exit(1)

    plot_residual_dot_plot(df_original, PLOT_OUTPUT_FILENAME)