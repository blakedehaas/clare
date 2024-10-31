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
from models.feed_forward import FF_2Network, FF_4Network, FF_5Network
import utils
import time
import datasets
# Hyperparameters
batch_size = 512
num_epochs = 1
max_lr = 1e-4
model_name = '1_23'
log_every_step = 10

with open("dataset/columns.txt", "r") as f:
    input_columns = eval(f.read().strip())
output_columns = ['Te1']
train_ds = datasets.Dataset.load_from_disk("data/akebono_solar_combined_v5_experimental_train_norm_preprocessed")
eval_ds = datasets.Dataset.load_from_disk("data/akebono_solar_combined_v5_experimental_eval_norm_preprocessed")
# train_ds = train_ds.remove_columns(['DateTimeFormatted', 'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])
# val_ds = val_ds.remove_columns(['DateTimeFormatted', 'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])

# Double-check that only the specified columns are kept
# all_columns = input_columns + output_columns
# assert set(train_ds.column_names) == set(all_columns), "Mismatch in columns after selection"
# assert set(eval_ds.column_names) == set(all_columns), "Mismatch in columns after selection"
# print(f"Verified: train_ds now contains only the {len(all_columns)} specified columns")

# # Normalize all location and atmospheric parameters (mean = 0 and std dev = 1)
# columns_to_normalize = input_columns

# Load or calculate mean and std for the specified columns
# stats_file = f'data/{model_name}_norm_stats.json'
# if os.path.exists(stats_file):
#     print(f"Loading existing normalization stats from {stats_file}")
#     with open(stats_file, 'r') as f:
#         stats = json.load(f)
#         means = stats['mean']
#         stds = stats['std']
# else:
#     pass
    # means, stds = utils.calculate_stats(train_ds, columns_to_normalize)
    # # Save stats to a JSON file
    # with open(stats_file, 'w') as f:
    #     json.dump({'mean': means, 'std': stds}, f)

# # Normalize train and eval datasets using the combined stats
# train_ds = utils.normalize_df(train_ds, means, stds, columns_to_normalize)
# val_ds = utils.normalize_df(val_ds, means, stds, columns_to_normalize)

# # Verify there are no NaNs in the normalized DataFrame
# assert train_df_norm[columns_to_normalize].isna().sum().sum() == 0, "NaN values found in normalized data"
# assert eval_df_norm[columns_to_normalize].isna().sum().sum() == 0, "NaN values found in normalized data"
# Set up data loader

# Convert to tensor
# def convert_to_tensor(row):
#     input_ids = torch.tensor([v for k,v in row.items() if k not in output_columns])
#     label = torch.tensor([v for k,v in row.items() if k in output_columns])
#     label = ((label - 6) / 0.05).clamp(0, 79).long().squeeze()
#     return {"input_ids": input_ids, "label": label}
# eval_ds = eval_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
# eval_ds.save_to_disk("data/akebono_solar_combined_v5_experimental_eval_norm_preprocessed")
# train_ds = train_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
# train_ds.save_to_disk("data/akebono_solar_combined_v5_experimental_train_norm_preprocessed")
eval_ds.set_format(type="torch")
train_ds.set_format(type="torch")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

# Initialize the model
input_size = len(input_columns)
hidden_size = 2048
output_size = 80
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
# Calculate and print total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=max_lr)

# Implement One Cycle LR
steps_per_epoch = len(train_loader)
total_train_steps = num_epochs * steps_per_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, 
                                                       eta_min=max_lr / 1000)

# Initialize wandb run
wandb.init(
    project="auroral-precipitation-ml",
    config={
        "dataset_size": len(train_ds),
        "validation_size": len(eval_ds),
    }
)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch["input_ids"].to("cuda")
            y = batch["label"].to("cuda")
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
    for i, batch in enumerate(tqdm(train_loader)):
        x = batch["input_ids"].to("cuda")
        y = batch["label"].to("cuda")

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
        
        # Evaluate the model 3 times per epoch
        if total_steps % ((len(train_loader) + 2) // 3) == 0:
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
torch.save(model.state_dict(), f'./checkpoints/{model_name}.pth')