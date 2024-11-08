# TestSuite.py

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
import sys
import shap


# Import custom modules
from DataProcessor import DataProcessor
from FeatureEngineer import FeatureEngineer
from CustomDataset import CustomDataset
from ModelBuilder import ModelBuilder
from Evaluator import Evaluator
from Trainer import Trainer
from Visualization import Visualization

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Test Suite for Time-Dependent ML Model with Time Series Cross-Validation")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model')
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for LSTM layers')
parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate for scheduler')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--seq_length', type=int, default=24, help='Sequence length for time series data')
parser.add_argument('--data_path', type=str, default='/glade/campaign/univ/ucul0008/auroral_precipitation_ml/Data/Dataset v3.0/combined_time_dataset.tsv', help='Path to the dataset')
parser.add_argument('--subset_fraction', type=float, default=0.1, help='Fraction of data to take from each year for testing (e.g., 0.1 for 10%)')
parser.add_argument('--output_dir', type=str, default='output_TestSuite', help='Directory to save all output files')
args = parser.parse_args()

# Validate subset_fraction
if not 0 < args.subset_fraction <= 1:
    print("[ERROR] Argument --subset_fraction must be between 0 and 1.")
    sys.exit(1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Create the main output directory for this run
try:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory created: {args.output_dir}")
except Exception as e:
    print(f"[ERROR] Failed to create output directory: {e}")
    sys.exit(1)

def save_model_and_pipeline(model, data_processor, path):
    """
    Saves the model's state dictionary and normalization statistics.
    
    Args:
        model (torch.nn.Module): Trained model.
        data_processor (DataProcessor): DataProcessor instance containing normalization stats.
        path (str): Directory path to save the files.
    """
    try:
        # Save the model
        torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
        print(f"[INFO] Model state_dict saved to {path}/model.pth")
        
        # Save normalization stats
        data_processor.save_transformations(os.path.join(path, 'norm_stats.json'))
        print(f"[INFO] Normalization statistics saved to {path}/norm_stats.json")
    except Exception as e:
        print(f"[ERROR] Failed to save model and pipeline: {e}")

def main():
    print("[DEBUG] Starting main() function.")
    
    # Initialize WandB for experiment tracking
    print("[DEBUG] Initializing WandB.")
    wandb.init(
        project="electron-temp-density-prediction-test",
        name=f"TestSuite_Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=vars(args)
    )
    print("[DEBUG] WandB initialized.")

    # Data Processing
    print("[DEBUG] Initializing DataProcessor.")
    data_processor = DataProcessor(data_path=args.data_path)
    print("[DEBUG] Preprocessing data.")
    data_processor.preprocess_data()
    df = data_processor.df
    print(f"[INFO] Data loaded with shape: {df.shape}")

    # Ensure 'Year' column exists; if not, extract it from 'DateTimeFormatted'
    if 'Year' not in df.columns:
        if 'DateTimeFormatted' in df.columns:
            print("[DEBUG] Extracting 'Year' from 'DateTimeFormatted'.")
            df['Year'] = pd.to_datetime(df['DateTimeFormatted']).dt.year
            print("[INFO] 'Year' column extracted.")
        else:
            print("[ERROR] Dataset must contain either 'Year' or 'DateTimeFormatted' column.")
            sys.exit(1)

    # Time series cross-validation: each year is a fold
    unique_years = sorted(df['Year'].unique())
    print(f"[INFO] Unique years in dataset: {unique_years}")

    for fold_idx, val_year in enumerate(unique_years):
        print(f'\n[INFO] Fold {fold_idx + 1}/{len(unique_years)} - Validation Year: {val_year}')

        # Split data
        train_df = df[df['Year'] < val_year].reset_index(drop=True)
        val_df = df[df['Year'] == val_year].reset_index(drop=True)
        print(f"[DEBUG] Training data shape: {train_df.shape}")
        print(f"[DEBUG] Validation data shape: {val_df.shape}")

        if train_df.empty:
            print(f"[WARNING] No training data available for validation year {val_year}. Skipping this fold.")
            continue

        # Determine samples_per_year based on subset_fraction
        samples_per_year_dict = {}
        for year in unique_years:
            if year >= val_year:
                continue  # Only consider years in training data
            year_train_df = train_df[train_df['Year'] == year]
            n_samples = max(1, int(args.subset_fraction * len(year_train_df)))
            samples_per_year_dict[year] = n_samples
            print(f"[DEBUG] Year {year}: Selecting {n_samples} samples for training.")

        # Sample equally from each year in training data
        sampled_train_df = pd.DataFrame()
        for year, n_samples in samples_per_year_dict.items():
            year_df = train_df[train_df['Year'] == year]
            if len(year_df) >= n_samples:
                sampled_year_df = year_df.sample(n=n_samples, random_state=42)
                print(f"[DEBUG] Year {year}: Sampled {n_samples} records.")
            else:
                sampled_year_df = year_df
                print(f"[DEBUG] Year {year}: Not enough records. Taking all {len(year_df)} records.")
            sampled_train_df = pd.concat([sampled_train_df, sampled_year_df], ignore_index=True)
        print(f"[INFO] Fold {fold_idx + 1}: Total sampled training records: {len(sampled_train_df)}")

        # Sample equally from each year in validation data
        sampled_val_df = pd.DataFrame()
        val_year_val_df = val_df[val_df['Year'] == val_year]
        n_val_samples = max(1, int(args.subset_fraction * len(val_year_val_df)))
        samples_per_val_year = {val_year: n_val_samples}
        print(f"[DEBUG] Year {val_year}: Selecting {n_val_samples} samples for validation.")

        for year, n_samples in samples_per_val_year.items():
            year_df = val_df[val_df['Year'] == year]
            if len(year_df) >= n_samples:
                sampled_year_df = year_df.sample(n=n_samples, random_state=42)
                print(f"[DEBUG] Year {year}: Sampled {n_samples} records.")
            else:
                sampled_year_df = year_df
                print(f"[DEBUG] Year {year}: Not enough records. Taking all {len(year_df)} records.")
            sampled_val_df = pd.concat([sampled_val_df, sampled_year_df], ignore_index=True)
        print(f"[INFO] Fold {fold_idx + 1}: Total sampled validation records: {len(sampled_val_df)}")

        # Calculate normalization statistics on training data only
        columns_to_normalize = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT',
                                'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H', 'f107_index']
        output_columns = ['Te1', 'Ne1']

        print("[DEBUG] Calculating normalization statistics.")
        medians, iqr = data_processor.calculate_stats(sampled_train_df, columns_to_normalize + output_columns)
        norm_stats = {
            'medians': medians.to_dict(),
            'iqr': iqr.to_dict()
        }
        print("[INFO] Normalization statistics calculated.")

        # Normalize training and validation data
        print("[DEBUG] Normalizing training data.")
        train_df_norm = data_processor.normalize_df(sampled_train_df.copy(), medians, iqr, columns_to_normalize + output_columns)
        print("[DEBUG] Normalizing validation data.")
        val_df_norm = data_processor.normalize_df(sampled_val_df.copy(), medians, iqr, columns_to_normalize + output_columns)
        print("[INFO] Data normalization complete.")

        # Drop unnecessary columns
        drop_columns = ['Year', 'DateTimeFormatted']
        print("[DEBUG] Dropping unnecessary columns.")
        train_df_norm = train_df_norm.drop(columns=drop_columns, errors='ignore')
        val_df_norm = val_df_norm.drop(columns=drop_columns, errors='ignore')
        print("[INFO] Unnecessary columns dropped.")

        # Define input and output columns
        input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT',
                         'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H',
                         'f107_index', 'Time_Minutes_sin', 'Time_Minutes_cos',
                         'Day_of_Year_sin', 'Day_of_Year_cos', 'Solar_Cycle_sin',
                         'Solar_Cycle_cos', 'GMLT_sin', 'GMLT_cos']
        # Ensure these columns exist in the dataset
        missing_columns = set(input_columns) - set(train_df_norm.columns)
        if missing_columns:
            print(f"[ERROR] Missing columns in the dataset: {missing_columns}. Skipping this fold.")
            continue
        print("[INFO] All required input columns are present.")

        # Create datasets
        print("[DEBUG] Creating CustomDataset instances.")
        train_dataset = CustomDataset(train_df_norm, input_columns, output_columns, seq_length=args.seq_length)
        val_dataset = CustomDataset(val_df_norm, input_columns, output_columns, seq_length=args.seq_length)
        print("[INFO] Datasets created.")

        # Create DataLoaders
        print("[DEBUG] Initializing DataLoaders.")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print("[INFO] DataLoaders initialized.")

        # Print shapes of the first batch to verify data integrity
        print("[DEBUG] Verifying first training batch.")
        try:
            x, y = next(iter(train_loader))
            print(f"[INFO] Train Batch 1 - Input shape: {x.shape}, Target shape: {y.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve first training batch: {e}")
            continue

        print("[DEBUG] Verifying first validation batch.")
        try:
            x, y = next(iter(val_loader))
            print(f"[INFO] Validation Batch 1 - Input shape: {x.shape}, Target shape: {y.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve first validation batch: {e}")
            continue

        # Initialize Model
        print("[DEBUG] Initializing the model.")
        input_size = len(input_columns)
        output_size = len(output_columns)
        model = ModelBuilder(input_size, args.hidden_size, output_size, dropout=args.dropout).to(device)
        print(f"[INFO] Model initialized with input_size={input_size}, hidden_size={args.hidden_size}, output_size={output_size}")

        # If multiple GPUs are available, use DataParallel
        if torch.cuda.device_count() > 1:
            print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel.")
            model = nn.DataParallel(model)
        else:
            print("[INFO] Single GPU or CPU detected. Using single device.")

        # Define loss function, optimizer, and scheduler
        print("[DEBUG] Setting up loss function, optimizer, and scheduler.")
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
        print("[INFO] Loss function, optimizer, and scheduler set up.")

        # Initialize Evaluator and Trainer
        print("[DEBUG] Initializing Evaluator and Trainer.")
        evaluator = Evaluator(model, criterion, device)
        trainer = Trainer(model, optimizer, scheduler, criterion, device, args.patience, evaluator, args.output_dir)
        print("[INFO] Evaluator and Trainer initialized.")

        # Train the model
        print("[DEBUG] Starting training process.")
        trainer.train(args.num_epochs, train_loader, val_loader)
        print("[INFO] Training completed.")

        # Evaluate on validation set
        print("[DEBUG] Evaluating model on validation set.")
        val_loss = evaluator.evaluate_metrics(val_loader)
        print(f'[INFO] Fold {fold_idx + 1} - {val_year}: Validation Loss: {val_loss:.4f}')

        # Residual Analysis
        print("[DEBUG] Performing residual analysis.")
        residuals = evaluator.residual_analysis(val_loader)
        # Print basic statistics of residuals
        residual_mean = np.mean(residuals, axis=0)
        residual_std = np.std(residuals, axis=0)
        print(f'Residuals - Mean: {residual_mean}, Std: {residual_std}')

        # Feature Importance using SHAP
        print("[DEBUG] Computing feature importance using SHAP.")
        # For feature importance, we can use a sample from the validation set
        # Note: SHAP can be computationally expensive, so use a small sample
        sample_size = min(100, len(val_dataset))
        if sample_size == 0:
            print("[WARNING] No samples available for SHAP analysis.")
            shap_values = []
        else:
            try:
                sample_indices = np.random.choice(len(val_dataset), sample_size, replace=False)
                sample_data = torch.stack([val_dataset[i][0] for i in sample_indices]).to(device)
                shap_values = evaluator.compute_feature_importance(sample_data)
                # SHAP values can be a list (for multi-output models)
                # Print shape for each output
                for idx, sv in enumerate(shap_values):
                    print(f'[INFO] SHAP values for output {idx + 1}: {sv.shape}')
            except Exception as e:
                print(f"[ERROR] Failed to compute SHAP values: {e}")
                shap_values = []

        # Initialize Visualization
        print("[DEBUG] Initializing Visualization module.")
        fold_dir = os.path.join(args.output_dir, f'Fold_{fold_idx + 1}_{val_year}')
        try:
            os.makedirs(fold_dir, exist_ok=True)
            print(f"[INFO] Fold directory created: {fold_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create fold directory: {e}")
            continue

        try:
            visualization = Visualization(fold_dir)
            print("[INFO] Visualization module initialized.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Visualization module: {e}")
            continue
        
        # Plot training and validation loss over epochs
        print("[DEBUG] Plotting training and validation loss.")
        try:
            visualization.plot_training_validation_loss(trainer.train_losses, trainer.val_losses, output_name='loss_plot.png')
            print("[INFO] Training and validation loss plot saved.")
        except Exception as e:
            print(f"[ERROR] Failed to plot training and validation loss: {e}")

        # Plot residuals histogram
        print("[DEBUG] Plotting residuals histogram.")
        try:
            visualization.plot_residuals_histogram(residuals, output_name='residuals_histogram.png')
            print("[INFO] Residuals histogram plot saved.")
        except Exception as e:
            print(f"[ERROR] Failed to plot residuals histogram: {e}")

        # Plot SHAP summary plots
        if isinstance(shap_values, shap.Explanation):
            print("[DEBUG] Plotting SHAP summary plot.")
            try:
                visualization.plot_shap_summary(shap_values, feature_names=input_columns, output_name='shap_summary.png')
                print("[INFO] SHAP summary plot saved.")
            except Exception as e:
                print(f"[ERROR] Failed to plot SHAP summary: {e}")
        else:
            print("[WARNING] Skipping SHAP summary plot due to lack of SHAP values.")

        # Save the best model and normalization stats
        print("[DEBUG] Saving the model and normalization statistics.")
        try:
            save_model_and_pipeline(model, data_processor, fold_dir)
            print("[INFO] Model and normalization statistics saved.")
        except Exception as e:
            print(f"[ERROR] Failed to save model and pipeline: {e}")

        print(f"[INFO] Fold {fold_idx + 1} completed.\n")

    print("[INFO] All folds completed successfully.")

if __name__ == '__main__':
    main()
