import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wandb
from models.feed_forward import FF_2Network
import utils

# Hyperparameters
batch_size = 4096
num_epochs = 1
max_lr = 5e-5
model_name = 'ff2_2'
eval_every_step = 250
log_every_step = 10

input_columns = ['ILAT', 'GLAT', 'GMLT', 'AL_index', 'SYM_H', 'f107_index']
output_column = 'Te1'
train_df = pd.read_csv('data/train_v3.tsv', sep='\t')
eval_df = pd.read_csv('data/validation_v3.tsv', sep='\t')

# Keep only input and output columns
columns_to_keep = input_columns + [output_column]
train_df = train_df[columns_to_keep]
eval_df = eval_df[columns_to_keep]

# Normalize all location and atmospheric parameters (mean = 0 and std dev = 1)
columns_to_normalize = ['ILAT', 'GLAT', 'GMLT', 'AL_index', 'SYM_H', 'f107_index', 'Te1']

# Combine train and validation datasets
combined_df = pd.concat([train_df, eval_df], axis=0)

# Calculate mean and std for the specified columns across combined dataset
means, stds = utils.calculate_stats(combined_df, columns_to_normalize)

# Save stats to a JSON file
with open('data/v3_norm_stats.json', 'w') as f:
    json.dump({'mean': means.to_dict(), 'std': stds.to_dict()}, f)

# Normalize train and eval datasets using the combined stats
train_df_norm = utils.normalize_df(train_df, means, stds, columns_to_normalize)
eval_df_norm = utils.normalize_df(eval_df, means, stds, columns_to_normalize)

# Verify there are no NaNs in the normalized DataFrame
assert train_df_norm[columns_to_normalize].isna().sum().sum() == 0, "NaN values found in normalized data"
assert eval_df_norm[columns_to_normalize].isna().sum().sum() == 0, "NaN values found in normalized data"

# Set up data loader
train_ds = utils.DataFrameDataset(train_df_norm, input_columns, output_column)
eval_ds = utils.DataFrameDataset(eval_df_norm, input_columns, output_column)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=8)

# Initialize the model
input_size = len(input_columns)
hidden_size = 2048
output_size = 1
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=max_lr)

# Implement One Cycle LR
steps_per_epoch = len(train_loader)
total_train_steps = num_epochs * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, 
                                                       eta_min=max_lr / 100)

# Initialize wandb run
wandb.init(
    project="auroral-precipitation-ml",
    config={
        "dataset_size": len(train_df),
        "validation_size": len(eval_df),
        "columns": columns_to_keep
    }
)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.to("cuda")
            y = y.to("cuda")
            if model.__class__.__name__ == 'GaussianNetwork':
                y_pred, var = model(x)
                loss = criterion(y_pred, y, var)
            else:
                y_pred = model(x)
                loss = criterion(y_pred, y)

            total_loss += loss.item()
    return total_loss / len(data_loader)

# Training loop
total_steps = 0
for epoch in range(num_epochs):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to("cuda")
        y = y.to("cuda")

        # Forward pass
        if model.__class__.__name__ == 'GaussianNetwork':
            y_pred, var = model(x)
            loss = criterion(y_pred, y, var)
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step()
        total_steps += 1

        # Log train loss and learning rate every log_every_step iterations
        if total_steps % log_every_step == 0:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "total_steps": total_steps
            })
        
        if total_steps % eval_every_step == 0:
            test_loss = evaluate_model(model, eval_loader, criterion)
            wandb.log({
                "test_loss": test_loss,
                "total_steps": total_steps
            })
    print(f'Epoch [{epoch+1}/{num_epochs}]')

    # Log epoch-level metrics
    wandb.log({
        "epoch": epoch + 1
    })

# Evaluate the model at the end of training
test_loss = evaluate_model(model, eval_loader, criterion)
print(f'Final Test Loss: {test_loss:.4f}')

# Log final test loss
wandb.log({
    "test_loss": test_loss,
    "total_steps": total_steps
})

print('Training finished!')

# Save the model
torch.save(model.state_dict(), f'{model_name}.pth')