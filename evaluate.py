from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import DataLoader
import models.feed_forward as models
import utils
import json
import matplotlib.pyplot as plt
import os
import datasets
import utils
import scipy
import constants

model_name = '1_39'
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
output_columns = ['Te1']
all_columns = input_columns + output_columns
dataset = "test-storm" # test-storm or test-normal


# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 150
model = models.FeedForwardNetwork(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))
model.eval()  # Set the model to evaluation mode

# Load dataset
test_ds = datasets.Dataset.load_from_disk(f"dataset/output_dataset/{dataset}")
test_ds = test_ds.remove_columns(['Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3'])


def normalize_batch(batch):
    for col, norm_func in constants.NORMALIZATIONS.items():
        batch[col] = norm_func(batch[col])
    return batch

test_ds = test_ds.map(normalize_batch, batched=True, batch_size=10000, num_proc=os.cpu_count())

# Solar indices
columns_to_normalize = [col for col in input_columns if col.startswith('AL_index') or col.startswith('SYM_H') or col.startswith('f107_index')]
index_groups = {
    'AL_index': [col for col in columns_to_normalize if col.startswith('AL_index')],
    'SYM_H': [col for col in columns_to_normalize if col.startswith('SYM_H')],
    'f107_index': [col for col in columns_to_normalize if col.startswith('f107_index')]
}
# Calculate mean and std for each group
means, stds = {}, {}
stats_file = f'data/{model_name}_norm_stats.json'

if os.path.exists(stats_file):
    print(f"Loading existing normalization stats from {stats_file}")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
        means = stats['mean']
        stds = stats['std']

# Apply normalization to all columns in group at once
group_cols = [col for cols in index_groups.values() for col in cols]
def normalize_group(batch):
    for col in group_cols:
        # Get the group name from the column name (e.g. 'AL_index_1' -> 'AL_index')
        group_name = '_'.join(col.split('_')[:-1]) if col.split('_')[-1].isdigit() else col
        # Convert batch[col] to numpy array before arithmetic operations
        values = np.array(batch[col], dtype=np.float32)
        batch[col] = (values - means[group_name]) / stds[group_name]
    return batch

test_ds = test_ds.map(normalize_group, batched=True, batch_size=10000, num_proc=os.cpu_count())

# Convert to tensor
def convert_to_tensor(row):
    input_ids = torch.tensor([v for k,v in row.items() if k not in output_columns + ['DateTimeFormatted']])
    label = torch.tensor([v for k,v in row.items() if k in output_columns])
    return {
        "input_ids": input_ids, 
        "label": label,
        "DateTimeFormatted": row['DateTimeFormatted']
    }
test_ds = test_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)

def custom_collate(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    labels = torch.stack([torch.tensor(item['label']) for item in batch])
    datetimes = [item['DateTimeFormatted'] for item in batch]
    return {
        'input_ids': input_ids,
        'label': labels,
        'DateTimeFormatted': datetimes
    }

test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate)

predictions, true_values, times = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        x = batch["input_ids"].to("cuda")
        y = batch["label"].to("cuda")

        # Forward pass
        logits = model(x)

        y_pred = torch.argmax(logits, dim=1) * 100 + 50
        y_true = y

        predictions.extend(y_pred.flatten().tolist())
        true_values.extend(y_true.flatten().tolist())
        times.extend(batch['DateTimeFormatted'])

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

# Plot absolute deviation vs ground truth
plt.figure(figsize=(10, 8))

h = plt.hist2d(true_values, deviations, bins=100, norm=LogNorm(), cmap='viridis')
plt.colorbar(h[3], label='Obs#')

plt.xlabel('Te$_{obs}$ [K]')
plt.ylabel('Te$_{model}$ - Te$_{obs}$ [K]')
plt.title('Model Deviation vs Ground Truth')

# Add mean deviation line and print the mean deviation value
bin_means, bin_edges, _ = scipy.stats.binned_statistic(true_values, deviations,
                                                statistic='mean', bins=50)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Mean Deviation')
plt.legend()

# Calculate and print the mean deviation value
mean_deviation = np.mean(deviations)
print(f"Mean Deviation: {mean_deviation:.3f}")

plt.tight_layout()
plt.savefig(f'./checkpoints/{model_name}_deviation_vs_truth.png', dpi=300)
plt.close()
