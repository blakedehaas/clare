from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.gaussian import GaussianNetwork
from models.feed_forward import FF_1Network, FF_2Network, FF_3Network
from models.transformer import TransformerRegressor
import utils
import json
import matplotlib.pyplot as plt

model_name = 'ff3_1_0'
input_columns = ['ILAT', 'GLAT', 'GMLT', 'AL_index', 'SYM_H', 'f107_index']
output_column = 'Te1'
columns_to_keep = input_columns + [output_column]
columns_to_normalize = ['ILAT', 'GLAT', 'GMLT', 'AL_index', 'SYM_H', 'f107_index', 'Te1']

# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 1
model = FF_3Network(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load('ff3_1_0.pth'))
model.eval()  # Set the model to evaluation mode

test_df = pd.read_csv('data/test_v4.tsv', sep='\t')
test_df = test_df[columns_to_keep]

# Load means and std from json file
with open(f'data/{model_name}_norm_stats.json', 'w') as f:
    norm_stats = json.load(f)

means = norm_stats['mean']
stds = norm_stats['std']
test_df_norm = utils.normalize_df(test_df, means, stds, columns_to_normalize)
test_ds = utils.DataFrameDataset(test_df_norm, input_columns, output_column)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8)

target_mean = means[output_column]
target_std = stds[output_column]

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
        y_true = utils.unnormalize_mean(y.cpu(), target_mean, target_std)
        y_pred = utils.unnormalize_mean(y_pred.cpu(), target_mean, target_std)
        
        predictions.extend(y_pred.flatten().tolist())
        true_values.extend(y_true.flatten().tolist())

# Calculate deviations
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
plt.savefig('deviation.png')
plt.close()  # Close the figure to free up memory