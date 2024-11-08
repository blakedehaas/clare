# time_model.py

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  # For experiment tracking
import argparse
from datetime import datetime

# Import custom modules
from DataProcessor import DataProcessor
from FeatureEngineer import FeatureEngineer
from CustomDataset import CustomDataset
from ModelBuilder import ModelBuilder
from Evaluator import Evaluator
from Trainer import Trainer
from Visualization import Visualization

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Time-Dependent ML Model Training with Time Series Cross-Validation")
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for DataLoader')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for LSTM layers')
parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate for scheduler')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--seq_length', type=int, default=24, help='Sequence length for time series data')
parser.add_argument('--data_path', type=str, default='/glade/campaign/univ/ucul0008/auroral_precipitation_ml/Data/combined_time_dataset.tsv', help='Path to the dataset')
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create an output directory based on current date & time
current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"output_TimeDependentModel_{current_timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

def save_model_and_pipeline(model, data_processor, path):
    """
    Saves the model's state dictionary and normalization statistics.
    
    Args:
        model (torch.nn.Module): Trained model.
        data_processor (DataProcessor): DataProcessor instance containing normalization stats.
        path (str): Directory path to save the files.
    """
    # Save the model
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    # Save normalization stats
    data_processor.save_transformations(os.path.join(path, 'norm_stats.json'))
    print(f"Model and preprocessing pipeline saved to {path}.")

def main():
    # Initialize WandB
    wandb.init(
        project="electron-temp-density-prediction",
        config=vars(args)
    )

    # Data Processing
    data_processor = DataProcessor(data_path=args.data_path)
    data_processor.preprocess_data()
    df = data_processor.df

    # Time series cross-validation: each year is a fold
    unique_years = sorted(df['Year'].unique())
    print(f"Unique years in dataset: {unique_years}")

    for fold_idx, val_year in enumerate(unique_years):
        print(f'\nFold {fold_idx + 1}/{len(unique_years)} - Validation Year: {val_year}')

        # Split data
        train_df = df[df['Year'] < val_year].reset_index(drop=True)
        val_df = df[df['Year'] == val_year].reset_index(drop=True)

        if train_df.empty:
            print(f"No training data available for validation year {val_year}. Skipping this fold.")
            continue

        # Calculate normalization statistics on training data only
        columns_to_normalize = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT',
                                'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H', 'f107_index']
        output_columns = ['Te1', 'Ne1']

        medians, iqr = data_processor.calculate_stats(train_df, columns_to_normalize + output_columns)
        norm_stats = {
            'medians': medians.to_dict(),
            'iqr': iqr.to_dict()
        }

        # Normalize training and validation data
        train_df_norm = data_processor.normalize_df(train_df.copy(), medians, iqr, columns_to_normalize + output_columns)
        val_df_norm = data_processor.normalize_df(val_df.copy(), medians, iqr, columns_to_normalize + output_columns)

        # Drop unnecessary columns
        drop_columns = ['Year', 'DateTimeFormatted']
        train_df_norm = train_df_norm.drop(columns=drop_columns, errors='ignore')
        val_df_norm = val_df_norm.drop(columns=drop_columns, errors='ignore')

        # Define input and output columns
        input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT',
                         'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H',
                         'f107_index', 'Time_Minutes_sin', 'Time_Minutes_cos',
                         'Day_of_Year_sin', 'Day_of_Year_cos', 'Solar_Cycle_sin',
                         'Solar_Cycle_cos', 'GMLT_sin', 'GMLT_cos']
        # Ensure these columns exist in the dataset
        missing_columns = set(input_columns) - set(train_df_norm.columns)
        if missing_columns:
            print(f"Missing columns in the dataset: {missing_columns}")
            continue

        # Create datasets
        train_dataset = CustomDataset(train_df_norm, input_columns, output_columns, seq_length=args.seq_length)
        val_dataset = CustomDataset(val_df_norm, input_columns, output_columns, seq_length=args.seq_length)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Print shapes of the first batch to verify data integrity
        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Train Batch {batch_idx + 1}:")
            print(f"Input shape: {x.shape}")  # Expected: (batch_size, seq_length, input_size)
            print(f"Target shape: {y.shape}")  # Expected: (batch_size, output_size)
            break  # Only print the first batch

        for batch_idx, (x, y) in enumerate(val_loader):
            print(f"Validation Batch {batch_idx + 1}:")
            print(f"Input shape: {x.shape}")  # Expected: (batch_size, seq_length, input_size)
            print(f"Target shape: {y.shape}")  # Expected: (batch_size, output_size)
            break  # Only print the first batch

        # Initialize Model
        input_size = len(input_columns)
        output_size = len(output_columns)
        model = ModelBuilder(input_size, args.hidden_size, output_size, dropout=args.dropout).to(device)

        # If multiple GPUs are available, use DataParallel
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Define loss function, optimizer, and scheduler
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

        total_steps = args.num_epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1e4
        )

        # Initialize Evaluator and Trainer
        evaluator = Evaluator(model, criterion, device)
        trainer = Trainer(model, optimizer, scheduler, criterion, device, args.patience, evaluator, output_dir)

        # Train the model
        trainer.train(args.num_epochs, train_loader, val_loader)

        # Evaluate on validation set
        val_loss = evaluator.evaluate_metrics(val_loader)
        print(f'Fold {fold_idx + 1} - {val_year}: Validation Loss: {val_loss:.4f}')

        # Residual Analysis
        residuals = evaluator.residual_analysis(val_loader)
        # Print basic statistics of residuals
        print(f'Residuals - Mean: {np.mean(residuals, axis=0)}, Std: {np.std(residuals, axis=0)}')

        # Feature Importance using SHAP
        # For feature importance, we can use a sample from the validation set
        # Note: SHAP can be computationally expensive, so use a small sample
        sample_size = min(100, len(val_dataset))
        if sample_size == 0:
            print("No samples available for SHAP analysis.")
            shap_values = []
        else:
            sample_indices = np.random.choice(len(val_dataset), sample_size, replace=False)
            sample_data = torch.stack([val_dataset[i][0] for i in sample_indices]).to(device)
            shap_values = evaluator.compute_feature_importance(sample_data)
            # SHAP values can be a list (for multi-output models)
            # Print shape for each output
            for idx, sv in enumerate(shap_values):
                print(f'SHAP values for output {idx + 1}: {sv.shape}')

        # Initialize Visualization
        fold_dir = os.path.join(output_dir, f'Fold_{fold_idx + 1}_{val_year}')
        os.makedirs(fold_dir, exist_ok=True)
        visualization = Visualization(fold_dir)
        
        # Plot training and validation loss over epochs
        visualization.plot_training_validation_loss(trainer.train_losses, trainer.val_losses, output_name='loss_plot.png')

        # Plot residuals histogram
        visualization.plot_residuals_histogram(residuals, output_name='residuals_histogram.png')

        # Plot SHAP summary plots
        if shap_values:
            visualization.plot_shap_summary(shap_values, feature_names=input_columns, output_name='shap_summary.png')
        else:
            print("Skipping SHAP summary plot due to lack of SHAP values.")

        # Save the best model and normalization stats
        save_model_and_pipeline(model, data_processor, fold_dir)

    if __name__ == '__main__':
        main()
