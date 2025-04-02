"""
Baseline performance with random guessing
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import datasets
import random
from sklearn.metrics import r2_score, mean_squared_error

random.seed(42)

# Load dataset
dataset = "test-normal" # test-storm or test-normal
test_ds = datasets.Dataset.load_from_disk(f"dataset/output_dataset/{dataset}")
test_ds = test_ds.remove_columns([col for col in test_ds.column_names if col != 'Te1'])

predictions, true_values = [], []

for item in tqdm(test_ds, desc="Evaluating"):
    y_true = item["Te1"]
    
    # Generate random prediction between 0 and 15000
    y_pred = random.uniform(0, 15000)
    predictions.append(int(y_pred))
    true_values.append(y_true)

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
plt.savefig(f'./checkpoints/{dataset}_baseline_deviation.png')
plt.close()  # Close the figure to free up memory
