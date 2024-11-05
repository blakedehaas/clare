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
model_name = '1_24'
log_every_step = 10

input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3']
output_columns = ['Te1']
train_path = "data/akebono_solar_combined_v6_chu_train"
train_subdirs = [f.path for f in os.scandir(train_path) if f.is_dir()]
train_datasets = []
for subdir in tqdm(train_subdirs, desc="Loading train datasets"):
    ds = datasets.Dataset.load_from_disk(subdir)
    train_datasets.append(ds)
train_ds = datasets.concatenate_datasets(train_datasets)
val_ds = datasets.Dataset.load_from_disk("data/akebono_solar_combined_v6_chu_val")

train_ds = train_ds.remove_columns(['DateTimeFormatted', 'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])
val_ds = val_ds.remove_columns(['DateTimeFormatted', 'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])

# Double-check that only the specified columns are kept
all_columns = input_columns + output_columns
assert set(train_ds.column_names) == set(all_columns), "Mismatch in columns after selection"
assert set(val_ds.column_names) == set(all_columns), "Mismatch in columns after selection"
columns_to_normalize = input_columns

# Load or calculate mean and std for the specified columns
stats_file = f'data/{model_name}_norm_stats.json'
if os.path.exists(stats_file):
    print(f"Loading existing normalization stats from {stats_file}")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
        means = stats['mean']
        stds = stats['std']
else:
    means, stds = utils.calculate_stats(val_ds, columns_to_normalize)
    with open(stats_file, 'w') as f:
        json.dump({'mean': means, 'std': stds}, f)

# Normalize train and eval datasets using the combined stats
train_ds = utils.normalize_ds(train_ds, means, stds, columns_to_normalize)
val_ds = utils.normalize_ds(val_ds, means, stds, columns_to_normalize, normalize_output=True)

# Convert to tensor
def convert_to_tensor(row):
    input_ids = torch.tensor([v for k,v in row.items() if k not in output_columns])
    label = torch.tensor([v for k,v in row.items() if k in output_columns])
    label = ((label - 6) / 0.05).clamp(0, 79).long().squeeze()
    return {"input_ids": input_ids, "label": label}
val_ds = val_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
train_ds = train_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
val_ds.set_format(type="torch")
train_ds.set_format(type="torch")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

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
        "validation_size": len(val_ds),
    }
)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch["input_ids"].to("cuda")
            y = batch["label"].to("cuda")
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
            test_loss = evaluate_model(model, val_loader, criterion)
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
test_loss = evaluate_model(model, val_loader, criterion)
print(f'Final Test Loss: {test_loss:.4f}')

# Log final test loss
wandb.log({
    "test_loss": test_loss,
    "total_steps": total_steps
})

print('Training finished!')

# Save the model
torch.save(model.state_dict(), f'./checkpoints/{model_name}.pth')