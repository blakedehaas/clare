import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    INPUT_COLUMNS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    NORMALIZATIONS, INDEX_GROUPS,
    DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH,
)
from models.feed_forward import FeedForwardNetwork


def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None):
    """Loads a trained FeedForwardNetwork from a checkpoint file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_normalization_stats(stats_path=DEFAULT_STATS_PATH):
    """Loads group normalization means and stds from a JSON file."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']


def normalize_batch_dataset(batch):
    """Applies feature-specific normalizations to a HuggingFace dataset batch."""
    for col, norm_func in NORMALIZATIONS.items():
        if col in batch:
            batch[col] = norm_func(batch[col])
    return batch


def normalize_group_dataset(batch, means, stds):
    """Applies group z-score normalization to a HuggingFace dataset batch."""
    for group_name, group_cols in INDEX_GROUPS.items():
        if group_name not in means or group_name not in stds:
            continue
        group_mean = means[group_name]
        group_std = stds[group_name]
        for col in group_cols:
            if col in batch:
                values = np.array(batch[col], dtype=np.float32)
                batch[col] = (values - group_mean) / group_std
    return batch


def preprocess_dataframe(df, means, stds):
    """Applies all normalizations to a pandas DataFrame (for visualization scripts)."""
    df_processed = df.copy()

    for col, norm_func in NORMALIZATIONS.items():
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = norm_func(df_processed[col])

    for group_name, group_cols in INDEX_GROUPS.items():
        if group_name not in means or group_name not in stds:
            continue
        group_mean = means[group_name]
        group_std = stds[group_name]
        if group_std == 0:
            continue
        for col in group_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = (df_processed[col].astype(np.float32) - group_mean) / group_std

    cols_to_check = [col for col in INPUT_COLUMNS if col in df_processed.columns]
    nans = df_processed[cols_to_check].isnull().sum().sum()
    if nans > 0:
        print(f"Warning: Found {nans} NaN values after normalization. Filling with 0.")
        df_processed[cols_to_check] = df_processed[cols_to_check].fillna(0)

    return df_processed


def predict(model, df_processed, device=None, batch_size=1024):
    """Runs inference on a preprocessed DataFrame; returns predicted temperatures."""
    if device is None:
        device = next(model.parameters()).device

    input_data = df_processed[INPUT_COLUMNS].values.astype(np.float32)
    input_tensor = torch.tensor(input_data)
    dataset = TensorDataset(input_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for (batch_tensor,) in tqdm(data_loader, desc="Predicting"):
            batch_tensor = batch_tensor.to(device)
            logits = model(batch_tensor)
            predicted_classes = torch.argmax(logits, dim=1)
            predicted_temps = predicted_classes.cpu().numpy() * 100 + 50
            all_predictions.extend(predicted_temps)

    return np.array(all_predictions)
