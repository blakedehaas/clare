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
import utils


model_name = '1_23'
with open("dataset/columns.txt", "r") as f:
    input_columns = eval(f.read().strip())
output_columns = ['Te1']
all_columns = input_columns + output_columns

# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 80
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))
model.eval()  # Set the model to evaluation mode

test_ds = Dataset.load_from_disk("data/akebono_solar_combined_v5_experimental_test_norm_preprocessed")
# test_ds = test_ds.remove_columns(['DateTimeFormatted', 'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])
# columns_to_normalize = input_columns

# # Load means and std from json file
# with open(f'data/{model_name}_norm_stats.json', 'r') as f:
#     stats = json.load(f)
#     means = stats['mean']
#     stds = stats['std']

# # Normalize dataset
# test_ds = utils.normalize_df(test_ds, means, stds, columns_to_normalize)

# # convert to tensor
# try:
#     def convert_to_tensor(row):
#         input_ids = torch.tensor([v for k,v in row.items() if k not in output_columns])
#         label = torch.tensor([v for k,v in row.items() if k in output_columns])
#         # label = ((label - 6) / 0.05).clamp(0, 79).long().squeeze()
#         return {"input_ids": input_ids, "label": label}
#     test_ds = test_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
# except Exception as e:
#     print(e)
#     import IPython; IPython.embed()

# test_ds.save_to_disk("data/akebono_solar_combined_v5_experimental_test_norm_preprocessed")


# target_mean = means[output_columns[0]]
# target_std = stds[output_columns[0]]

# del means[output_columns[0]]
# del stds[output_columns[0]]

# Define invalid values

# test_df_norm = utils.normalize_df(test_df, means, stds, columns_to_normalize)
# test_ds = utils.DataFrameDataset(test_df_norm, input_columns, output_columns)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count())

predictions, true_values = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        x = torch.stack(batch["input_ids"]).to(torch.float32).permute(1,0).to("cuda")
        y = torch.stack(batch["label"]).permute(1,0).to("cuda")

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
print(text)
plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot
plt.savefig(f'./checkpoints/{model_name}_deviation.png')
plt.close()  # Close the figure to free up memory

# Print some predictions and targets
# print("\nSample predictions and targets:")
# num_samples = 10
# for i in range(num_samples):
#     percent_deviation = (deviations[i] / true_values[i]) * 100
#     print(f"Prediction: {predictions[i]:.2f}, Target: {true_values[i]:.2f}, Deviation: {deviations[i]:.2f}, Percent Deviation: {percent_deviation:.2f}%")


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
