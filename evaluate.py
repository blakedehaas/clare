from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import Dataset, DataLoader
from models.gaussian import GaussianNetwork
from models.feed_forward import FF_1Network, FF_2Network, FF_3Network, FF_4Network, FF_5Network
from models.transformer import TransformerRegressor
import utils
import json
import matplotlib.pyplot as plt
import os


model_name = '1_10'
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H', 'Mag_Scalar_B', 'exists_Mag_Scalar_B', 'Mag_Vector_B', 'exists_Mag_Vector_B', 'Mag_B_Lat_GSE', 'exists_Mag_B_Lat_GSE', 'Mag_B_Long_GSE', 'exists_Mag_B_Long_GSE', 'Mag_BX_GSE', 'exists_Mag_BX_GSE', 'Mag_BY_GSE', 'exists_Mag_BY_GSE', 'Mag_BZ_GSE', 'exists_Mag_BZ_GSE', 'Mag_BY_GSM', 'exists_Mag_BY_GSM', 'Mag_BZ_GSM', 'exists_Mag_BZ_GSM', 'Mag_RMS_Mag', 'exists_Mag_RMS_Mag', 'Mag_RMS_Vector', 'exists_Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'exists_Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'exists_Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE', 'exists_Mag_RMS_BZ_GSE', 'Plasma_SW_Temp', 'exists_Plasma_SW_Temp', 'Plasma_SW_Density', 'exists_Plasma_SW_Density', 'Plasma_SW_Speed', 'exists_Plasma_SW_Speed', 'Plasma_SW_Flow_Long', 'exists_Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'exists_Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 'exists_Plasma_Alpha_Prot_Ratio', 'Plasma_Sigma_T', 'exists_Plasma_Sigma_T', 'Plasma_Sigma_N', 'exists_Plasma_Sigma_N', 'Plasma_Sigma_V', 'exists_Plasma_Sigma_V', 'Plasma_Sigma_Phi_V', 'exists_Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'exists_Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio', 'exists_Plasma_Sigma_Ratio', 'Solar_Kp', 'exists_Solar_Kp', 'Solar_R_Sunspot', 'exists_Solar_R_Sunspot', 'Solar_Dst', 'exists_Solar_Dst', 'Solar_Ap', 'exists_Solar_Ap', 'Solar_AE', 'exists_Solar_AE', 'Solar_AL', 'exists_Solar_AL', 'Solar_AU', 'exists_Solar_AU', 'Solar_PC', 'exists_Solar_PC', 'Solar_Lyman_Alpha', 'exists_Solar_Lyman_Alpha', 'Particle_Proton_Flux_1MeV', 'exists_Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 'exists_Particle_Proton_Flux_2MeV', 'Particle_Proton_Flux_4MeV', 'exists_Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 'exists_Particle_Proton_Flux_10MeV', 'Particle_Proton_Flux_30MeV', 'exists_Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV', 'exists_Particle_Proton_Flux_60MeV', 'Particle_Flux_Flag', 'BSN_X_GSE', 'exists_BSN_X_GSE', 'BSN_Y_GSE', 'exists_BSN_Y_GSE', 'BSN_Z_GSE', 'exists_BSN_Z_GSE', 'AE_index', 'exists_AE_index', 'AU_index', 'exists_AU_index', 'SYM_D', 'exists_SYM_D', 'ASY_D', 'exists_ASY_D', 'ASY_H', 'exists_ASY_H', 'PCN_index', 'exists_PCN_index', 'f107_index']
output_column = ['Te1']
columns_to_keep = input_columns + output_column + ['DateTimeFormatted']
columns_to_normalize = input_columns
columns_to_normalize = [col for col in columns_to_normalize if 'exists_' not in col]

# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 1
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))
model.eval()  # Set the model to evaluation mode

test_df = pd.read_csv('data/akebono_solar_combined_v3_experiment_more_indices_test.tsv', sep='\t')
test_df = test_df[columns_to_keep]

# Load means and std from json file
with open(f'data/{model_name}_norm_stats.json', 'r') as f:
    norm_stats = json.load(f)

means = norm_stats['mean']
stds = norm_stats['std']

target_mean = means[output_column[0]]
target_std = stds[output_column[0]]

del means['Te1']
del stds['Te1']

test_df_norm = utils.normalize_df(test_df, means, stds, columns_to_normalize)
test_ds = utils.DataFrameDataset(test_df_norm, input_columns, output_column)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=os.cpu_count())

predictions, true_values = [], []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x = x.to("cuda")
        y = y.to("cuda")
        
        # Forward pass
        if model.__class__.__name__ == 'GaussianNetwork':
            y_pred, var = model(x)
        else:
            y_pred = model(x)

        # Unnormalize predictions and true values
        # y_true = utils.unnormalize_mean(y.cpu(), target_mean, target_std)
        y_pred = utils.unnormalize_mean(y_pred.cpu(), target_mean, target_std)
        # Convert predictions from log scale back to original scale
        # y_pred = torch.exp(y_pred)
        # y_pred = torch.argmax(y_pred, dim=1) * 100 + 50
        # y_pred = y_pred * 1000
        y_true = y

        predictions.extend(y_pred.flatten().tolist())
        true_values.extend(y_true.flatten().tolist())


deviations = [pred - true for pred, true in zip(predictions, true_values)]
# Extract dateTimeFormatted from the test_df and convert to datetime objects
date_times = pd.to_datetime(test_df['DateTimeFormatted'])

# Extract years for x-axis
years = date_times.dt.year

# Create a scatter plot of deviations over time
plt.figure(figsize=(12, 8))
plt.scatter(date_times, deviations, alpha=0.5, s=1)
plt.xlabel('Year')
plt.ylabel('Deviation from Ground Truth')
plt.title('Deviations Over Time')

# Set x-axis to show only years
years_range = range(min(years), max(years) + 1)
plt.xticks(pd.to_datetime([f"{year}-01-01" for year in years_range]), years_range)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='r', linestyle='--')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save the plot
plt.savefig(f'./checkpoints/{model_name}_deviation_over_time.png')
plt.close()  # Close the figure to free up memory

# Calculate percentages within specified absolute deviations
thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
percentages = [
    sum(abs(dev) <= threshold for dev in deviations) / len(deviations) * 100
    for threshold in thresholds
]

# Calculate percentages within specified relative deviations
relative_thresholds = [5, 10, 15, 20]
relative_percentages = [
    sum(abs(dev) / true * 100 <= threshold for dev, true in zip(deviations, true_values)) / len(deviations) * 100
    for threshold in relative_thresholds
]

# Plot histogram
plt.figure(figsize=(12, 8))
plt.hist(deviations, bins=50, edgecolor='black')
plt.xlabel('Deviation from Ground Truth')
plt.ylabel('Frequency')
plt.title('Distribution of Model Predictions Deviation')

# Add text box with percentages
text = "\n".join([
    f"Within {threshold}: {percentage:.2f}%"
    for threshold, percentage in zip(thresholds, percentages)
] + ["\n"] + [  # Add an empty line between absolute and relative thresholds
    f"Within {threshold}%: {percentage:.2f}%"
    for threshold, percentage in zip(relative_thresholds, relative_percentages)
])
plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot
plt.savefig(f'./checkpoints/{model_name}_deviation.png')
plt.close()  # Close the figure to free up memory

# Print some predictions and targets
print("\nSample predictions and targets:")
num_samples = 10
for i in range(num_samples):
    percent_deviation = (deviations[i] / true_values[i]) * 100
    print(f"Prediction: {predictions[i]:.2f}, Target: {true_values[i]:.2f}, Deviation: {deviations[i]:.2f}, Percent Deviation: {percent_deviation:.2f}%")


# Chu plot (logarithmic axes)
plt.figure(figsize=(10, 8))
h = plt.hist2d(predictions, true_values, bins=100, norm=LogNorm(), cmap='viridis', 
               range=[[0, 15000], [0, 15000]])
plt.colorbar(h[3], label='Obs#')

plt.xlabel('Te$_{model}$ [K]')
plt.ylabel('Te$_{obs}$ [K]')
plt.title('Test (Logarithmic Scale)')

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, 15000)
plt.ylim(1, 15000)
exp_ticks = [10**i for i in range(5)]
plt.xticks(exp_ticks)
plt.yticks(exp_ticks)

plt.plot([1, 15000], [1, 15000], 'r--', alpha=0.75, zorder=10)
predictions_norm = [(pred - target_mean) / target_std for pred in predictions]
true_values_norm = [(true - target_mean) / target_std for true in true_values]
rmse = np.sqrt(np.mean([(pred - true)**2 for pred, true in zip(predictions_norm, true_values_norm)]))
r = np.corrcoef(predictions_norm, true_values_norm)[0, 1]

plt.text(0.05, 0.95, f'RMSE={rmse:.3f}\nr={r:.4f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'./checkpoints/{model_name}_chu.png', dpi=300)
plt.close()

# Correlation plot (regular constant axes)
plt.figure(figsize=(10, 8))
h = plt.hist2d(predictions, true_values, bins=100, norm=LogNorm(), cmap='viridis', 
               range=[[0, 15000], [0, 15000]])
plt.colorbar(h[3], label='Obs#')

plt.xlabel('Te$_{model}$ [K]')
plt.ylabel('Te$_{obs}$ [K]')
plt.title('Test (Linear Scale)')

plt.xlim(0, 15000)
plt.ylim(0, 15000)

plt.plot([0, 15000], [0, 15000], 'r--', alpha=0.75, zorder=10)

plt.text(0.05, 0.95, f'RMSE={rmse:.3f}\nr={r:.4f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'./checkpoints/{model_name}_correlation.png', dpi=300)
plt.close()

