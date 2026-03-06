import argparse
import json
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
import datasets

from config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, COLUMNS_TO_REMOVE,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    INDEX_GROUPS, DEFAULT_DATASET_DIR,
)
from inference import normalize_batch_dataset, normalize_group_dataset
from models.feed_forward import FeedForwardNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CLARE model.")
    parser.add_argument("--model-name", default="clare", help="Name for this training run (used for checkpoint filenames)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Maximum learning rate")
    parser.add_argument("--dataset-dir", default=None, help="Path to processed dataset directory")
    parser.add_argument("--log-every-step", type=int, default=10, help="Log training loss every N steps")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR
    min_lr = args.lr / 1000

    # Load and concatenate training chunks
    train_path = os.path.join(dataset_dir, "train_chunks")
    train_datasets = []
    for folder in sorted(os.listdir(train_path)):
        chunk = datasets.Dataset.load_from_disk(os.path.join(train_path, folder))
        train_datasets.append(chunk)

    train_ds = datasets.concatenate_datasets(train_datasets)
    print("Length train ds", len(train_ds))

    val_ds = datasets.Dataset.load_from_disk(os.path.join(dataset_dir, "test-normal"))
    train_ds = train_ds.remove_columns(COLUMNS_TO_REMOVE)
    val_ds = val_ds.remove_columns(COLUMNS_TO_REMOVE)

    all_columns = INPUT_COLUMNS + OUTPUT_COLUMNS
    assert set(train_ds.column_names) == set(all_columns), "Mismatch in columns after selection"
    assert set(val_ds.column_names) == set(all_columns), "Mismatch in columns after selection"

    # Feature-specific normalization
    train_ds = train_ds.map(normalize_batch_dataset, batched=True, batch_size=10000, num_proc=os.cpu_count())
    val_ds = val_ds.map(normalize_batch_dataset, batched=True, batch_size=10000, num_proc=os.cpu_count())

    # Compute or load group normalization stats
    os.makedirs('./checkpoints', exist_ok=True)
    stats_file = f'checkpoints/{args.model_name}_norm_stats.json'

    if os.path.exists(stats_file):
        print(f"Loading existing normalization stats from {stats_file}")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            means = stats['mean']
            stds = stats['std']
    else:
        means, stds = {}, {}
        for group_name, group_cols in tqdm(INDEX_GROUPS.items(), desc="Calculating group stats"):
            group_values = np.concatenate([train_ds.with_format("pandas")[col].values for col in group_cols])
            means[group_name] = float(np.mean(group_values))
            stds[group_name] = float(np.std(group_values))
        with open(stats_file, 'w') as f:
            json.dump({'mean': means, 'std': stds}, f)

    # Group z-score normalization
    normalize_fn = lambda batch: normalize_group_dataset(batch, means, stds)
    train_ds = train_ds.map(normalize_fn, batched=True, batch_size=10000, num_proc=os.cpu_count())
    val_ds = val_ds.map(normalize_fn, batched=True, batch_size=10000, num_proc=os.cpu_count())

    # Convert to tensors
    def convert_to_tensor(row):
        input_ids = torch.tensor([v for k, v in row.items() if k not in OUTPUT_COLUMNS])
        label = torch.tensor([v for k, v in row.items() if k in OUTPUT_COLUMNS])
        label = (label // 100).clamp(0, 149).long().squeeze()
        return {"input_ids": input_ids, "label": label}

    val_ds = val_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
    train_ds = train_ds.map(convert_to_tensor, num_proc=os.cpu_count(), remove_columns=all_columns)
    val_ds.set_format(type="torch")
    train_ds.set_format(type="torch")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())

    # Initialize model
    model = FeedForwardNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to("cuda")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_loader)
    total_train_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=min_lr)

    wandb.init(
        project="clare",
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
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            x = batch["input_ids"].to("cuda")
            y = batch["label"].to("cuda")

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_steps += 1

            if total_steps % args.log_every_step == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "total_steps": total_steps
                })

            if total_steps % ((len(train_loader) + 2) // 3) == 0:
                test_loss = evaluate_model(model, val_loader, criterion)
                wandb.log({
                    "test_loss": test_loss,
                    "total_steps": total_steps
                })

        print(f'Epoch [{epoch+1}/{args.epochs}]')
        wandb.log({"epoch": epoch + 1})

    test_loss = evaluate_model(model, val_loader, criterion)
    print(f'Final Test Loss: {test_loss:.4f}')
    wandb.log({"test_loss": test_loss, "total_steps": total_steps})

    print('Training finished!')
    torch.save(model.state_dict(), f'./checkpoints/{args.model_name}.pth')


if __name__ == "__main__":
    main()
