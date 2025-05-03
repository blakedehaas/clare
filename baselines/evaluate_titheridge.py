"""
Evaluate the performance of Titheridge against the test set targets, calculate the same metrics as `evaluate.py`
"""
import datasets
from tqdm import tqdm
import os
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy

ds = datasets.load_from_disk('/home/michael/auroral-precipitation-ml/dataset/output_dataset/test-normal-baseline-ready')

# Remove rows where z_titheridge_Te is nan
print(f"Dataset size before filtering: {len(ds)}")
ds = ds.filter(lambda x: not np.isnan(x['z_titheridge_Te']))
print(f"Dataset size after filtering: {len(ds)}")

def extract_predictions_and_truth(batch):
    return {
        'predictions': batch['z_titheridge_Te'],
        'true_values': batch['Te1']
    }

results = ds.map(extract_predictions_and_truth, num_proc=os.cpu_count())
predictions = results['predictions']
true_values = results['true_values']

# Calculate metrics and plots
deviations = [pred - true for pred, true in zip(predictions, true_values)]

# Calculate R^2 score
r2 = r2_score(true_values, predictions)
print(f"R^2 Score: {r2:.4f}")

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(true_values, predictions))
print(f"RMSE: {rmse:.4f}")

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

# Add text box with percentages and metrics
text = "\n".join([
    f"R² Score: {r2:.4f}",
    f"RMSE: {rmse:.4f}",
    "\n"  # Add empty line
] + [
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
plt.savefig(f'../checkpoints/titheridge_plot.png')
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
plt.savefig(f'../checkpoints/titheridge_deviation_plot.png', dpi=300)
plt.close()