import argparse
import os

from tqdm import tqdm
import numpy as np
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datasets
import scipy
from sklearn.metrics import r2_score, mean_squared_error

from config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, COLUMNS_TO_REMOVE,
    DEFAULT_DATASET_DIR, DEFAULT_CHECKPOINT_PATH, DEFAULT_STATS_PATH,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
)
from inference import (
    load_model, load_normalization_stats,
    normalize_batch_dataset, normalize_group_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the CLARE model.")
    parser.add_argument("--dataset", default="test-normal", choices=["test-normal", "test-storm"],
                        help="Which test split to evaluate on")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--stats", default=None, help="Path to normalization stats JSON")
    parser.add_argument("--dataset-dir", default=None, help="Path to processed dataset directory")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR
    checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINT_PATH
    stats_path = args.stats or DEFAULT_STATS_PATH

    # Load model
    model = load_model(checkpoint_path, device=torch.device("cuda"))
    means, stds = load_normalization_stats(stats_path)

    # Load dataset
    test_ds = datasets.Dataset.load_from_disk(os.path.join(dataset_dir, args.dataset))

    columns_to_remove = [c for c in COLUMNS_TO_REMOVE if c != 'DateTimeFormatted']
    test_ds = test_ds.remove_columns(columns_to_remove)

    # Normalize
    test_ds = test_ds.map(normalize_batch_dataset, batched=True, batch_size=10000, num_proc=os.cpu_count())
    normalize_fn = lambda batch: normalize_group_dataset(batch, means, stds)
    test_ds = test_ds.map(normalize_fn, batched=True, batch_size=10000, num_proc=os.cpu_count())

    # Convert to tensor
    def convert_to_tensor(row):
        input_ids = torch.tensor([v for k, v in row.items() if k in INPUT_COLUMNS])
        label = torch.tensor([v for k, v in row.items() if k in OUTPUT_COLUMNS])
        return {
            "input_ids": input_ids,
            "label": label,
            "DateTimeFormatted": row['DateTimeFormatted']
        }

    all_columns = INPUT_COLUMNS + OUTPUT_COLUMNS
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

    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=os.cpu_count(), collate_fn=custom_collate)

    predictions, true_values, entropy_list, times = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch["input_ids"].to("cuda")
            y = batch["label"].to("cuda")

            logits = model(x)

            softmaxed = torch.softmax(logits, dim=1)
            entropy = -torch.sum(softmaxed * torch.log(softmaxed + 1e-10), dim=1).cpu().numpy()
            entropy_list.extend(entropy)

            y_pred = torch.argmax(logits, dim=1) * 100 + 50
            y_true = y

            predictions.extend(y_pred.flatten().tolist())
            true_values.extend(y_true.flatten().tolist())
            times.extend(batch['DateTimeFormatted'])

    deviations = [pred - true for pred, true in zip(predictions, true_values)]

    r2 = r2_score(true_values, predictions)
    print(f"R^2 Score: {r2:.4f}")

    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    print(f"RMSE: {rmse:.4f}")

    mean_entropy = np.mean(entropy_list)
    print(f"\nEntropy Metrics:")
    print(f"Mean entropy across test set: {mean_entropy:.4f}")

    thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
    percentages = [
        sum(abs(dev) <= threshold for dev in deviations) / len(deviations) * 100
        for threshold in thresholds
    ]

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

    text = "\n".join([
        f"R\u00b2 Score: {r2:.4f}",
        f"RMSE: {rmse:.4f}",
        f"Mean entropy: {mean_entropy:.4f}",
        "\n"
    ] + [
        f"Within {threshold}: {percentage:.2f}%"
        for threshold, percentage in zip(thresholds, percentages)
    ] + ["\n"] + [
        f"Within {threshold}%: {percentage:.2f}%"
        for threshold, percentage in zip(relative_thresholds, relative_percentages)
    ])
    print(text)
    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    os.makedirs('./checkpoints', exist_ok=True)
    plt.savefig(f'./checkpoints/{args.dataset}_plot.png')
    plt.close()

    # Plot absolute deviation vs ground truth
    plt.figure(figsize=(10, 8))

    h = plt.hist2d(true_values, deviations, bins=100, norm=LogNorm(), cmap='viridis')
    plt.colorbar(h[3], label='Obs#')

    plt.xlabel('Te$_{obs}$ [K]')
    plt.ylabel('Te$_{model}$ - Te$_{obs}$ [K]')
    plt.title('Model Deviation vs Ground Truth')

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(true_values, deviations,
                                                           statistic='mean', bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Mean Deviation')
    plt.legend()

    mean_deviation = np.mean(deviations)
    print(f"Mean Deviation: {mean_deviation:.3f}")

    plt.tight_layout()
    plt.savefig(f'./checkpoints/{args.dataset}_deviation_plot.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
