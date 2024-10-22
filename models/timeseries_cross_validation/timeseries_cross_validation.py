# Import necessary libraries
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  # If you're using Weights & Biases
import utils  # Ensure utils.py is accessible
from models.feed_forward import FF_2Network  # Import your model
from evaluate import evaluate_model  # Import the updated evaluation function
import argparse

# Parsing command line arguments from sweep
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--dropout_rate', type=float)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--max_lr', type=float)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--weight_decay', type=float)
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to save model checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

# Function to load model checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print(f"=> Loading checkpoint from {checkpoint}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return epoch

# Read the dataset and parse dates
data_path = 'Data/akebono_solar_combined.tsv'
df = pd.read_csv(data_path, sep='\t', parse_dates=['DateTimeFormatted'])

# Define input columns by excluding unnecessary ones
columns_to_exclude = ['Te2', 'Te3', 'Ne2', 'Ne3', 'Te1', 'Ne1']
input_columns = [col for col in df.columns if col not in columns_to_exclude + ['DateTimeFormatted']]  # Exclude unnecessary columns
output_columns = ['Te1', 'Ne1']  # Predicting Te1 and Ne1

# Combine input and output columns for normalization
columns_to_normalize = input_columns + output_columns

# Add a 'Year' column and sort by date
df['Year'] = df['DateTimeFormatted'].dt.year
df = df.sort_values('DateTimeFormatted').reset_index(drop=True)
years = sorted(df['Year'].unique())

# Prepare results storage
fold_results = []

# Loop over each year for cross-validation
for fold, val_year in enumerate(years):
    print(f'Year-Based Fold {fold + 1}/{len(years)} - Validation Year: {val_year}')

    # Training and validation data split
    train_df = df[df['Year'] < val_year]
    val_df = df[df['Year'] == val_year]

    if train_df.empty:
        print('No training data for this fold. Skipping.')
        continue

    train_df = train_df.drop(columns=['Year', 'DateTimeFormatted'])
    val_df = val_df.drop(columns=['Year', 'DateTimeFormatted'])

    # Normalize data
    means, stds = utils.calculate_stats(train_df, columns_to_normalize)
    train_df_norm = utils.normalize_df(train_df.copy(), means, stds, columns_to_normalize)
    val_df_norm = utils.normalize_df(val_df.copy(), means, stds, columns_to_normalize)

    # Data loaders
    train_ds = utils.DataFrameDataset(train_df_norm, input_columns, output_columns)
    val_ds = utils.DataFrameDataset(val_df_norm, input_columns, output_columns)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Initialize the model
    model = FF_2Network(len(input_columns), args.hidden_size, len(output_columns), dropout_rate=args.dropout_rate).to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    # WandB initialization for each fold
    wandb.init(
        project="auroral-precipitation-ml",
        name=f"Year_{val_year}",
        config={
            "fold": fold + 1,
            "validation_year": val_year,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "max_lr": args.max_lr,
            "hidden_size": args.hidden_size,
            "dropout_rate": args.dropout_rate,
            "weight_decay": args.weight_decay,
        }
    )

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate and log validation loss every epoch
        val_loss = evaluate_model(model, val_loader, criterion, device, means[output_columns], stds[output_columns], fold+1)
        wandb.log({"validation_loss": val_loss, "epoch": epoch + 1})

        print(f'Epoch [{epoch+1}/{args.num_epochs}] completed. Validation Loss: {val_loss:.4f}')

    # Save the final model
    save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1})

    # Finish WandB run
    wandb.finish()
