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

# Hyperparameters
batch_size = 512
num_epochs = 1
max_lr = 1e-4
model_name = '1_10'
log_every_step = 10

input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H', 'Mag_Scalar_B', 'exists_Mag_Scalar_B', 'Mag_Vector_B', 'exists_Mag_Vector_B', 'Mag_B_Lat_GSE', 'exists_Mag_B_Lat_GSE', 'Mag_B_Long_GSE', 'exists_Mag_B_Long_GSE', 'Mag_BX_GSE', 'exists_Mag_BX_GSE', 'Mag_BY_GSE', 'exists_Mag_BY_GSE', 'Mag_BZ_GSE', 'exists_Mag_BZ_GSE', 'Mag_BY_GSM', 'exists_Mag_BY_GSM', 'Mag_BZ_GSM', 'exists_Mag_BZ_GSM', 'Mag_RMS_Mag', 'exists_Mag_RMS_Mag', 'Mag_RMS_Vector', 'exists_Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'exists_Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'exists_Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE', 'exists_Mag_RMS_BZ_GSE', 'Plasma_SW_Temp', 'exists_Plasma_SW_Temp', 'Plasma_SW_Density', 'exists_Plasma_SW_Density', 'Plasma_SW_Speed', 'exists_Plasma_SW_Speed', 'Plasma_SW_Flow_Long', 'exists_Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'exists_Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 'exists_Plasma_Alpha_Prot_Ratio', 'Plasma_Sigma_T', 'exists_Plasma_Sigma_T', 'Plasma_Sigma_N', 'exists_Plasma_Sigma_N', 'Plasma_Sigma_V', 'exists_Plasma_Sigma_V', 'Plasma_Sigma_Phi_V', 'exists_Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'exists_Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio', 'exists_Plasma_Sigma_Ratio', 'Solar_Kp', 'exists_Solar_Kp', 'Solar_R_Sunspot', 'exists_Solar_R_Sunspot', 'Solar_Dst', 'exists_Solar_Dst', 'Solar_Ap', 'exists_Solar_Ap', 'Solar_AE', 'exists_Solar_AE', 'Solar_AL', 'exists_Solar_AL', 'Solar_AU', 'exists_Solar_AU', 'Solar_PC', 'exists_Solar_PC', 'Solar_Lyman_Alpha', 'exists_Solar_Lyman_Alpha', 'Particle_Proton_Flux_1MeV', 'exists_Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 'exists_Particle_Proton_Flux_2MeV', 'Particle_Proton_Flux_4MeV', 'exists_Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 'exists_Particle_Proton_Flux_10MeV', 'Particle_Proton_Flux_30MeV', 'exists_Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV', 'exists_Particle_Proton_Flux_60MeV', 'Particle_Flux_Flag', 'BSN_X_GSE', 'exists_BSN_X_GSE', 'BSN_Y_GSE', 'exists_BSN_Y_GSE', 'BSN_Z_GSE', 'exists_BSN_Z_GSE', 'AE_index', 'exists_AE_index', 'AU_index', 'exists_AU_index', 'SYM_D', 'exists_SYM_D', 'ASY_D', 'exists_ASY_D', 'ASY_H', 'exists_ASY_H', 'PCN_index', 'exists_PCN_index', 'f107_index']
output_column = ['Te1']
train_df = pd.read_csv('data/akebono_solar_combined_v3_experiment_more_indices_train.tsv', sep='\t')
eval_df = pd.read_csv('data/akebono_solar_combined_v3_experiment_more_indices_val.tsv', sep='\t')

# Keep only input and output columns
columns_to_keep = input_columns + output_column
train_df = train_df[columns_to_keep]
eval_df = eval_df[columns_to_keep]

# Normalize all location and atmospheric parameters (mean = 0 and std dev = 1)
columns_to_normalize = input_columns + output_column
columns_to_normalize = [col for col in columns_to_normalize if 'exists_' not in col]

# Calculate mean and std for the specified columns across combined dataset
means, stds = utils.calculate_stats(train_df, columns_to_normalize)

# Save stats to a JSON file
with open(f'data/{model_name}_norm_stats.json', 'w') as f:
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
# train_ds = utils.SamplingDataset(train_df_norm, input_columns, output_column, sampling_ratios=[0.2, 0.2, 0.2, 0.4])
# eval_ds = utils.DataFrameDataset(eval_df_norm, input_columns, output_column)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

# Initialize the model
input_size = len(input_columns)
hidden_size = 2048
output_size = 1
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
# Calculate and print total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

# Define loss function and optimizer
criterion = nn.MSELoss()
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