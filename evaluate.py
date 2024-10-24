from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import DataLoader
from models.gaussian import GaussianNetwork
from models.feed_forward import FF_1Network, FF_2Network, FF_3Network, FF_4Network, FF_5Network
from models.transformer import TransformerRegressor
import utils
import json
import matplotlib.pyplot as plt
import os
from datasets import Dataset


model_name = '1_19'
# input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index', 'SYM_H', 'Mag_Scalar_B', 'exists_Mag_Scalar_B', 'Mag_Vector_B', 'exists_Mag_Vector_B', 'Mag_B_Lat_GSE', 'exists_Mag_B_Lat_GSE', 'Mag_B_Long_GSE', 'exists_Mag_B_Long_GSE', 'Mag_BX_GSE', 'exists_Mag_BX_GSE', 'Mag_BY_GSE', 'exists_Mag_BY_GSE', 'Mag_BZ_GSE', 'exists_Mag_BZ_GSE', 'Mag_BY_GSM', 'exists_Mag_BY_GSM', 'Mag_BZ_GSM', 'exists_Mag_BZ_GSM', 'Mag_RMS_Mag', 'exists_Mag_RMS_Mag', 'Mag_RMS_Vector', 'exists_Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'exists_Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'exists_Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE', 'exists_Mag_RMS_BZ_GSE', 'Plasma_SW_Temp', 'exists_Plasma_SW_Temp', 'Plasma_SW_Density', 'exists_Plasma_SW_Density', 'Plasma_SW_Speed', 'exists_Plasma_SW_Speed', 'Plasma_SW_Flow_Long', 'exists_Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'exists_Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 'exists_Plasma_Alpha_Prot_Ratio', 'Plasma_Sigma_T', 'exists_Plasma_Sigma_T', 'Plasma_Sigma_N', 'exists_Plasma_Sigma_N', 'Plasma_Sigma_V', 'exists_Plasma_Sigma_V', 'Plasma_Sigma_Phi_V', 'exists_Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'exists_Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio', 'exists_Plasma_Sigma_Ratio', 'Solar_Kp', 'exists_Solar_Kp', 'Solar_R_Sunspot', 'exists_Solar_R_Sunspot', 'Solar_Dst', 'exists_Solar_Dst', 'Solar_Ap', 'exists_Solar_Ap', 'Solar_AE', 'exists_Solar_AE', 'Solar_AL', 'exists_Solar_AL', 'Solar_AU', 'exists_Solar_AU', 'Solar_PC', 'exists_Solar_PC', 'Solar_Lyman_Alpha', 'exists_Solar_Lyman_Alpha', 'Particle_Proton_Flux_1MeV', 'exists_Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 'exists_Particle_Proton_Flux_2MeV', 'Particle_Proton_Flux_4MeV', 'exists_Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 'exists_Particle_Proton_Flux_10MeV', 'Particle_Proton_Flux_30MeV', 'exists_Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV', 'exists_Particle_Proton_Flux_60MeV', 'Particle_Flux_Flag', 'BSN_X_GSE', 'exists_BSN_X_GSE', 'BSN_Y_GSE', 'exists_BSN_Y_GSE', 'BSN_Z_GSE', 'exists_BSN_Z_GSE', 'AE_index', 'exists_AE_index', 'AU_index', 'exists_AU_index', 'SYM_D', 'exists_SYM_D', 'ASY_D', 'exists_ASY_D', 'ASY_H', 'exists_ASY_H', 'PCN_index', 'exists_PCN_index', 'f107_index']
input_columns =['Altitude',
 'GCLAT',
 'GCLON',
 'ILAT',
 'GLAT',
 'GMLT',
 'XXLAT',
 'XXLON',
 'Mag_Scalar_B',
 'Mag_Vector_B',
 'Mag_B_Lat_GSE',
 'Mag_B_Long_GSE',
 'Mag_BX_GSE',
 'Mag_BY_GSE',
 'Mag_BZ_GSE',
 'Mag_BY_GSM',
 'Mag_BZ_GSM',
 'Mag_RMS_Mag',
 'Mag_RMS_Vector',
 'Mag_RMS_BX_GSE',
 'Mag_RMS_BY_GSE',
 'Mag_RMS_BZ_GSE',
 'Plasma_SW_Temp',
 'Plasma_SW_Density',
 'Plasma_SW_Speed',
 'Plasma_SW_Flow_Long',
 'Plasma_SW_Flow_Lat',
 'Plasma_Alpha_Prot_Ratio',
 'Plasma_Sigma_T',
 'Plasma_Sigma_N',
 'Plasma_Sigma_V',
 'Plasma_Sigma_Phi_V',
 'Plasma_Sigma_Theta_V',
 'Plasma_Sigma_Ratio',
 'Solar_Kp',
 'Solar_R_Sunspot',
 'Solar_Dst',
 'Solar_Ap',
 'Solar_AE',
 'Solar_AL',
 'Solar_AU',
 'Solar_PC',
 'Solar_Lyman_Alpha',
 'Particle_Proton_Flux_1MeV',
 'Particle_Proton_Flux_2MeV',
 'Particle_Proton_Flux_4MeV',
 'Particle_Proton_Flux_10MeV',
 'Particle_Proton_Flux_30MeV',
 'Particle_Proton_Flux_60MeV',
 'Particle_Flux_Flag',
 'exists_Mag_Scalar_B',
 'exists_Mag_Vector_B',
 'exists_Mag_B_Lat_GSE',
 'exists_Mag_B_Long_GSE',
 'exists_Mag_BX_GSE',
 'exists_Mag_BY_GSE',
 'exists_Mag_BZ_GSE',
 'exists_Mag_BY_GSM',
 'exists_Mag_BZ_GSM',
 'exists_Mag_RMS_Mag',
 'exists_Mag_RMS_Vector',
 'exists_Mag_RMS_BX_GSE',
 'exists_Mag_RMS_BY_GSE',
 'exists_Mag_RMS_BZ_GSE',
 'exists_Plasma_SW_Temp',
 'exists_Plasma_SW_Density',
 'exists_Plasma_SW_Speed',
 'exists_Plasma_SW_Flow_Long',
 'exists_Plasma_SW_Flow_Lat',
 'exists_Plasma_Alpha_Prot_Ratio',
 'exists_Plasma_Sigma_T',
 'exists_Plasma_Sigma_N',
 'exists_Plasma_Sigma_V',
 'exists_Plasma_Sigma_Phi_V',
 'exists_Plasma_Sigma_Theta_V',
 'exists_Plasma_Sigma_Ratio',
 'exists_Solar_Kp',
 'exists_Solar_R_Sunspot',
 'exists_Solar_Dst',
 'exists_Solar_Ap',
 'exists_Solar_AE',
 'exists_Solar_AL',
 'exists_Solar_AU',
 'exists_Solar_PC',
 'exists_Solar_Lyman_Alpha',
 'exists_Particle_Proton_Flux_1MeV',
 'exists_Particle_Proton_Flux_2MeV',
 'exists_Particle_Proton_Flux_4MeV',
 'exists_Particle_Proton_Flux_10MeV',
 'exists_Particle_Proton_Flux_30MeV',
 'exists_Particle_Proton_Flux_60MeV',
 'BSN_X_GSE',
 'BSN_Y_GSE',
 'BSN_Z_GSE',
 'AE_index',
 'AU_index',
 'SYM_D',
 'ASY_D',
 'ASY_H',
 'PCN_index',
 'exists_BSN_X_GSE',
 'exists_BSN_Y_GSE',
 'exists_BSN_Z_GSE',
 'exists_AE_index',
 'exists_AU_index',
 'exists_SYM_D',
 'exists_ASY_D',
 'exists_ASY_H',
 'exists_PCN_index',
 'AL_index_0',
 'AL_index_1',
 'AL_index_2',
 'AL_index_3',
 'AL_index_4',
 'AL_index_5',
 'AL_index_6',
 'AL_index_7',
 'AL_index_8',
 'AL_index_9',
 'AL_index_10',
 'AL_index_11',
 'AL_index_12',
 'AL_index_13',
 'AL_index_14',
 'AL_index_15',
 'AL_index_16',
 'AL_index_17',
 'AL_index_18',
 'AL_index_19',
 'AL_index_20',
 'AL_index_21',
 'AL_index_22',
 'AL_index_23',
 'AL_index_24',
 'AL_index_25',
 'AL_index_26',
 'AL_index_27',
 'AL_index_28',
 'AL_index_29',
 'AL_index_30',
 'SYM_H_0',
 'SYM_H_1',
 'SYM_H_2',
 'SYM_H_3',
 'SYM_H_4',
 'SYM_H_5',
 'SYM_H_6',
 'SYM_H_7',
 'SYM_H_8',
 'SYM_H_9',
 'SYM_H_10',
 'SYM_H_11',
 'SYM_H_12',
 'SYM_H_13',
 'SYM_H_14',
 'SYM_H_15',
 'SYM_H_16',
 'SYM_H_17',
 'SYM_H_18',
 'SYM_H_19',
 'SYM_H_20',
 'SYM_H_21',
 'SYM_H_22',
 'SYM_H_23',
 'SYM_H_24',
 'SYM_H_25',
 'SYM_H_26',
 'SYM_H_27',
 'SYM_H_28',
 'SYM_H_29',
 'SYM_H_30',
 'SYM_H_31',
 'SYM_H_32',
 'SYM_H_33',
 'SYM_H_34',
 'SYM_H_35',
 'SYM_H_36',
 'SYM_H_37',
 'SYM_H_38',
 'SYM_H_39',
 'SYM_H_40',
 'SYM_H_41',
 'SYM_H_42',
 'SYM_H_43',
 'SYM_H_44',
 'SYM_H_45',
 'SYM_H_46',
 'SYM_H_47',
 'SYM_H_48',
 'SYM_H_49',
 'SYM_H_50',
 'SYM_H_51',
 'SYM_H_52',
 'SYM_H_53',
 'SYM_H_54',
 'SYM_H_55',
 'SYM_H_56',
 'SYM_H_57',
 'SYM_H_58',
 'SYM_H_59',
 'SYM_H_60',
 'SYM_H_61',
 'SYM_H_62',
 'SYM_H_63',
 'SYM_H_64',
 'SYM_H_65',
 'SYM_H_66',
 'SYM_H_67',
 'SYM_H_68',
 'SYM_H_69',
 'SYM_H_70',
 'SYM_H_71',
 'SYM_H_72',
 'SYM_H_73',
 'SYM_H_74',
 'SYM_H_75',
 'SYM_H_76',
 'SYM_H_77',
 'SYM_H_78',
 'SYM_H_79',
 'SYM_H_80',
 'SYM_H_81',
 'SYM_H_82',
 'SYM_H_83',
 'SYM_H_84',
 'SYM_H_85',
 'SYM_H_86',
 'SYM_H_87',
 'SYM_H_88',
 'SYM_H_89',
 'SYM_H_90',
 'SYM_H_91',
 'SYM_H_92',
 'SYM_H_93',
 'SYM_H_94',
 'SYM_H_95',
 'SYM_H_96',
 'SYM_H_97',
 'SYM_H_98',
 'SYM_H_99',
 'SYM_H_100',
 'SYM_H_101',
 'SYM_H_102',
 'SYM_H_103',
 'SYM_H_104',
 'SYM_H_105',
 'SYM_H_106',
 'SYM_H_107',
 'SYM_H_108',
 'SYM_H_109',
 'SYM_H_110',
 'SYM_H_111',
 'SYM_H_112',
 'SYM_H_113',
 'SYM_H_114',
 'SYM_H_115',
 'SYM_H_116',
 'SYM_H_117',
 'SYM_H_118',
 'SYM_H_119',
 'SYM_H_120',
 'SYM_H_121',
 'SYM_H_122',
 'SYM_H_123',
 'SYM_H_124',
 'SYM_H_125',
 'SYM_H_126',
 'SYM_H_127',
 'SYM_H_128',
 'SYM_H_129',
 'SYM_H_130',
 'SYM_H_131',
 'SYM_H_132',
 'SYM_H_133',
 'SYM_H_134',
 'SYM_H_135',
 'SYM_H_136',
 'SYM_H_137',
 'SYM_H_138',
 'SYM_H_139',
 'SYM_H_140',
 'SYM_H_141',
 'SYM_H_142',
 'SYM_H_143',
 'SYM_H_144',
 'f107_index_0',
 'f107_index_1',
 'f107_index_2',
 'f107_index_3']
output_column = ['Te1']
columns_to_keep = input_columns + output_column + ['DateTimeFormatted']
columns_to_normalize = input_columns
columns_to_normalize = [col for col in columns_to_normalize if 'exists_' not in col]

# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 80
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))
model.eval()  # Set the model to evaluation mode

# test_df = pd.read_csv('data/akebono_solar_combined_v3_experiment_more_indices_test.tsv', sep='\t')
loaded_test_dataset = Dataset.load_from_disk("data/akebono_solar_combined_v4_experimental_test")
test_df = loaded_test_dataset.to_pandas()
test_df = test_df[columns_to_keep]

# Load means and std from json file
with open(f'data/{model_name}_norm_stats.json', 'r') as f:
    norm_stats = json.load(f)

means = norm_stats['mean']
stds = norm_stats['std']

# target_mean = means[output_column[0]]
# target_std = stds[output_column[0]]

# del means[output_column[0]]
# del stds[output_column[0]]

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
        # y_pred = utils.unnormalize_mean(y_pred.cpu(), target_mean, target_std)
        # Convert predictions from log scale back to original scale
        # y_pred_probs = torch.nn.functional.softmax(y_pred, dim=1)

        y_pred = torch.exp(torch.argmax(y_pred, dim=1) * 0.05 + 6 + 0.025)
        # y_pred = torch.argmax(y_pred, dim=1) * 0.05 - 3 + 0.025
        # y_pred = utils.unnormalize_mean(y_pred.cpu(), target_mean, target_std)

        # y_pred = torch.exp(y_pred)
        # y_pred = torch.argmax(y_pred, dim=1) * 100 + 50
        # y_pred = torch.argmax(y_pred, dim=1) * 50 + 25

        # y_pred = y_pred * 1000
        y_true = y

        predictions.extend(y_pred.flatten().tolist())
        true_values.extend(y_true.flatten().tolist())


deviations = [pred - true for pred, true in zip(predictions, true_values)]

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
rmse = np.sqrt(np.mean([(pred - true)**2 for pred, true in zip(predictions, true_values)]))
r = np.corrcoef(predictions, true_values)[0, 1]
plt.text(0.05, 0.95, f'RMSE={rmse:.3f}\nr={r:.4f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'./checkpoints/{model_name}_correlation.png', dpi=300)
plt.close()



# # Extract dateTimeFormatted from the test_df and convert to datetime objects
# date_times = pd.to_datetime(test_df['DateTimeFormatted'])

# # Extract years for x-axis
# years = date_times.dt.year

# # Create a scatter plot of deviations over time
# plt.figure(figsize=(12, 8))
# plt.scatter(date_times, deviations, alpha=0.5, s=1)
# plt.xlabel('Year')
# plt.ylabel('Deviation from Ground Truth')
# plt.title('Deviations Over Time')

# # Set x-axis to show only years
# years_range = range(min(years), max(years) + 1)
# plt.xticks(pd.to_datetime([f"{year}-01-01" for year in years_range]), years_range)

# # Add a horizontal line at y=0 for reference
# plt.axhline(y=0, color='r', linestyle='--')

# # Adjust layout to prevent cutting off labels
# plt.tight_layout()

# # Save the plot
# plt.savefig(f'./checkpoints/{model_name}_deviation_over_time.png')
# plt.close()  # Close the figure to free up memory


# # Chu plot (logarithmic axes)
# plt.figure(figsize=(10, 8))
# h = plt.hist2d(predictions, true_values, bins=100, norm=LogNorm(), cmap='viridis', 
#                range=[[0, 15000], [0, 15000]])
# plt.colorbar(h[3], label='Obs#')

# plt.xlabel('Te$_{model}$ [K]')
# plt.ylabel('Te$_{obs}$ [K]')
# plt.title('Test (Logarithmic Scale)')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1, 15000)
# plt.ylim(1, 15000)
# exp_ticks = [10**i for i in range(5)]
# plt.xticks(exp_ticks)
# plt.yticks(exp_ticks)

# plt.plot([1, 15000], [1, 15000], 'r--', alpha=0.75, zorder=10)
# # predictions_norm = [(pred - target_mean) / target_std for pred in predictions]
# # true_values_norm = [(true - target_mean) / target_std for true in true_values]
# predictions_norm = predictions
# true_values_norm = true_values
# rmse = np.sqrt(np.mean([(pred - true)**2 for pred, true in zip(predictions_norm, true_values_norm)]))
# r = np.corrcoef(predictions_norm, true_values_norm)[0, 1]

# plt.text(0.05, 0.95, f'RMSE={rmse:.3f}\nr={r:.4f}', transform=plt.gca().transAxes,
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# plt.tight_layout()
# plt.savefig(f'./checkpoints/{model_name}_chu.png', dpi=300)
# plt.close()
