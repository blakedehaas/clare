import argparse
import json
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

import constants
from models.long_context import LongContextTransformer


TARGET = "Te1"
TIME = "DateTimeFormatted"
UNUSED = {"Ne1", "Pv1", "Te2", "Ne2", "Pv2", "Te3", "Ne3", "Pv3", "I1", "I2", "I3"}


class ContextDataset(Dataset):
    """Previous N observations plus a target-masked current observation."""

    def __init__(self, features, targets, timestamps, context_length, max_gap_minutes):
        order = np.argsort(timestamps)
        self.features = np.asarray(features, dtype=np.float32)[order]
        self.targets = np.asarray(targets, dtype=np.float32)[order]
        self.timestamps = np.asarray(timestamps, dtype="datetime64[ns]")[order]
        self.context_length = context_length

        gaps = np.diff(self.timestamps).astype("timedelta64[s]").astype(np.int64)
        breaks = np.flatnonzero((gaps < 0) | (gaps > max_gap_minutes * 60)) + 1
        segment_start = np.maximum.accumulate(
            np.where(np.isin(np.arange(len(self.timestamps)), breaks), np.arange(len(self.timestamps)), 0)
        )
        rows = np.arange(len(self.timestamps))
        self.ends = rows[rows - segment_start >= context_length]

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        end = self.ends[index]
        start = end - self.context_length
        x = self.features[start:end + 1]
        history_target = (self.targets[start:end + 1] / 15_000).reshape(-1, 1)
        observed = np.ones((len(x), 1), dtype=np.float32)
        history_target[-1] = 0
        observed[-1] = 0
        age = ((self.timestamps[start:end + 1] - self.timestamps[end]).astype("timedelta64[s]").astype(np.float32) / 3600).reshape(-1, 1)
        tokens = np.concatenate((x, history_target, observed, age), axis=1)
        label = int(np.clip(self.targets[end] // 100, 0, 149))
        return torch.from_numpy(tokens), label


class FixedTimeContextDataset(Dataset):
    """A fixed time horizon aggregated into equal-width historical bins."""

    def __init__(self, features, targets, timestamps, horizon_hours, bin_minutes):
        order = np.argsort(timestamps)
        self.features = np.asarray(features, dtype=np.float32)[order]
        self.targets = np.asarray(targets, dtype=np.float32)[order]
        timestamps = np.asarray(timestamps, dtype="datetime64[ns]")[order]
        self.bin_minutes = bin_minutes
        self.context_length = horizon_hours * 60 // bin_minutes
        row_bins = timestamps.astype("datetime64[m]").astype(np.int64) // bin_minutes
        self.bin_ids, starts, counts = np.unique(row_bins, return_index=True, return_counts=True)
        self.bin_features = np.add.reduceat(self.features, starts, axis=0) / counts[:, None]
        self.bin_targets = np.add.reduceat(self.targets, starts) / counts
        self.row_bins = row_bins

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        wanted = self.row_bins[index] - np.arange(self.context_length, 0, -1)
        positions = np.searchsorted(self.bin_ids, wanted)
        present = positions < len(self.bin_ids)
        present[present] &= self.bin_ids[positions[present]] == wanted[present]
        tokens = np.zeros((self.context_length + 1, self.features.shape[1] + 3), dtype=np.float32)
        tokens[:-1, -1] = -np.arange(self.context_length, 0, -1) * self.bin_minutes / 60
        tokens[:-1, -2] = present
        tokens[:-1, :-3][present] = self.bin_features[positions[present]]
        tokens[:-1, -3][present] = self.bin_targets[positions[present]] / 15_000
        tokens[-1, :-3] = self.features[index]
        return torch.from_numpy(tokens), int(np.clip(self.targets[index] // 100, 0, 149))


def load_dataset(path):
    path = Path(path)
    if (path / "dataset_info.json").exists():
        return datasets.Dataset.load_from_disk(str(path))
    chunks = [datasets.Dataset.load_from_disk(str(chunk)) for chunk in sorted(path.iterdir())]
    return datasets.concatenate_datasets(chunks)


def prepare(dataset, input_columns, stats=None):
    data = dataset.select_columns(input_columns + [TARGET, TIME]).to_pandas()
    features = []
    stats = {} if stats is None else stats
    for column in input_columns:
        values = np.asarray(data[column], dtype=np.float32)
        if column in constants.NORMALIZATIONS:
            values = constants.NORMALIZATIONS[column](values)
        else:
            group = column.rsplit("_", 1)[0]
            if group in {"AL_index", "SYM_H", "f107_index"}:
                if group not in stats:
                    group_columns = [name for name in input_columns if name.startswith(group + "_")]
                    joined = data[group_columns].to_numpy(dtype=np.float32).ravel()
                    stats[group] = {"mean": float(joined.mean()), "std": float(joined.std())}
                values = (values - stats[group]["mean"]) / stats[group]["std"]
        features.append(values)
    return np.column_stack(features), np.asarray(data[TARGET]), np.asarray(data[TIME]), stats


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    with torch.inference_mode():
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            total += criterion(model(tokens), labels).item() * len(labels)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True, help="training split or its train_chunks directory")
    parser.add_argument("--validation-path", required=True, help="continuous validation split")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-gap-minutes", type=int, default=10)
    parser.add_argument("--time-bins", action="store_true", help="use a fixed-duration context instead of previous rows")
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--bin-minutes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--model-name", default="long_context_transformer")
    args = parser.parse_args()

    train = load_dataset(args.train_path)
    validation = load_dataset(args.validation_path)
    input_columns = [name for name in train.column_names if name not in UNUSED | {TARGET, TIME}]
    train_x, train_y, train_time, stats = prepare(train, input_columns)
    val_x, val_y, val_time, _ = prepare(validation, input_columns, stats)
    if args.time_bins:
        if args.horizon_hours * 60 % args.bin_minutes:
            raise ValueError("--bin-minutes must divide --horizon-hours exactly")
        train_data = FixedTimeContextDataset(train_x, train_y, train_time, args.horizon_hours, args.bin_minutes)
        val_data = FixedTimeContextDataset(val_x, val_y, val_time, args.horizon_hours, args.bin_minutes)
        args.context_length = train_data.context_length
    else:
        train_data = ContextDataset(train_x, train_y, train_time, args.context_length, args.max_gap_minutes)
        val_data = ContextDataset(val_x, val_y, val_time, args.context_length, args.max_gap_minutes)
    if not train_data or not val_data:
        raise ValueError("no continuous sequences found; increase --max-gap-minutes or reduce --context-length")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongContextTransformer(train_x.shape[1] + 3, args.context_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    Path("checkpoints").mkdir(exist_ok=True)
    with open(f"checkpoints/{args.model_name}_config.json", "w") as file:
        json.dump({"input_columns": input_columns, "normalization": stats, **vars(args)}, file)

    wandb.init(project="clare", config={**vars(args), "train_sequences": len(train_data), "validation_sequences": len(val_data)})
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for tokens, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            tokens, labels = tokens.to(device), labels.to(device)
            loss = criterion(model(tokens), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                wandb.log({"train_loss": loss.item(), "total_steps": step})
            if step % 1000 == 0:
                torch.save(model.state_dict(), f"checkpoints/{args.model_name}_step_{step}.pth")
            if args.max_steps and step >= args.max_steps:
                break
        val_loss = evaluate(model, val_loader, criterion, device)
        model.train()
        wandb.log({"validation_loss": val_loss, "epoch": epoch + 1, "total_steps": step})
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
