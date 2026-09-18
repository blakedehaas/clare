"""Train CLARE or the matched continuous model on the canonical block-split dataset."""

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
import models.feed_forward as models

parser = argparse.ArgumentParser(description="Train model on seed-split dataset.")
parser.add_argument(
    "--seed",
    type=int,
    default=int(os.environ.get("SPLIT_SEED", 0)),
    help="Dataset split seed.",
)
parser.add_argument(
    "--block-minutes",
    type=int,
    default=int(os.environ.get("BLOCK_MINUTES", 212)),
    help="Temporal block size in minutes.",
)
parser.add_argument(
    "--train-seed",
    type=int,
    default=int(os.environ.get("TRAIN_SEED", 0)),
    help="Independent seed controlling model initialization, dropout, and DataLoader shuffle.",
)
parser.add_argument(
    "--evals-per-epoch",
    type=int,
    default=int(os.environ.get("EVALS_PER_EPOCH", 10)),
    help=(
        "Target number of full validation passes per epoch. "
        "The exact optimizer-step interval is derived from the current "
        "training DataLoader size. Default: 10."
    ),
)
parser.add_argument(
    "--continuous",
    action="store_true",
    help="Train continuous regression variant (MSE loss, one output) instead of classification.",
)
args = parser.parse_args()

split_seed = args.seed
block_minutes = args.block_minutes
train_seed = args.train_seed
evals_per_epoch = args.evals_per_epoch

if evals_per_epoch < 1:
    raise ValueError("--evals-per-epoch must be >= 1")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)
torch.cuda.manual_seed_all(train_seed)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data_loader_generator = torch.Generator()
data_loader_generator.manual_seed(train_seed)

base_model_name = "2"

model_name = (
    f"{base_model_name}_{block_minutes}m_s{split_seed}_ts{train_seed}_continuous"
    if args.continuous
    else f"{base_model_name}_{block_minutes}m_s{split_seed}_ts{train_seed}"
)

dataset_dir = (
    f"dataset/processed_dataset_blocksplit_{block_minutes}m_s{split_seed}"
)

print(f"[INFO] Dataset split seed: {split_seed}")
print(f"[INFO] Training seed: {train_seed}")
print(f"[INFO] Block minutes: {block_minutes}")
print(
    f"[INFO] Mode: "
    f"{'Continuous Regression' if args.continuous else 'Classification-Based Regression'}"
)
print(f"[INFO] Model checkpoint tag: {model_name}")
print(f"[INFO] Reading dataset from: {dataset_dir}")

num_workers = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else 8
)

batch_size = 512
num_epochs = 100
max_lr = 8e-4
min_lr = max_lr / 1000
log_every_step = 1

TE_MIN_K = 0.0
TE_BIN_WIDTH_K = 100.0
TE_NUM_CLASSES = 150
TE_MAX_K_EXCLUSIVE = TE_MIN_K + TE_BIN_WIDTH_K * TE_NUM_CLASSES
SOFT_TARGET_SIGMA_K = 100.0

raw_input_columns = [
    "Altitude",
    "GCLAT",
    "GCLON",
    "ILAT",
    "GLAT",
    "GMLT",
    "XXLAT",
    "XXLON",
    *[f"AL_index_{i}" for i in range(31)],
    *[f"SYM_H_{i}" for i in range(145)],
    *[f"f107_index_{i}" for i in range(4)],
    "Kp_index",
]

input_columns = [
    "Altitude",
    "GCLAT",
    "GCLON_sin",
    "GCLON_cos",
    "ILAT",
    "GLAT",
    "GMLT_sin",
    "GMLT_cos",
    "XXLAT",
    "XXLON_sin",
    "XXLON_cos",
    *[f"AL_index_{i}" for i in range(31)],
    *[f"SYM_H_{i}" for i in range(145)],
    *[f"f107_index_{i}" for i in range(4)],
    "Kp_index",
]

output_columns = ["Te1"]

print(f"[INFO] Model input size: {len(input_columns)}")

def classification_loss_and_temperature(logits, true_te):
    """Gaussian soft-target cross-entropy + expected-temperature decoding."""
    centers = (
        TE_MIN_K
        + (torch.arange(
            TE_NUM_CLASSES,
            device=logits.device,
            dtype=logits.dtype,
        ) + 0.5) * TE_BIN_WIDTH_K
    )
    targets = torch.softmax(
        -0.5 * ((centers[None, :] - true_te[:, None]) / SOFT_TARGET_SIGMA_K) ** 2,
        dim=1,
    )
    loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    pred_te = torch.softmax(logits, dim=1) @ centers
    return loss, pred_te

train_path = os.path.join(dataset_dir, "train_chunks")

if not os.path.isdir(train_path):
    raise FileNotFoundError(f"Training chunk directory not found: {train_path}")

train_chunk_names = [
    name
    for name in os.listdir(train_path)
    if (
        name.startswith("train_chunk_")
        and os.path.isdir(os.path.join(train_path, name))
    )
]

try:
    train_chunk_names = sorted(
        train_chunk_names,
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
except ValueError as exc:
    raise ValueError(
        "Every training chunk directory must end in an integer, "
        "for example train_chunk_1."
    ) from exc

if not train_chunk_names:
    raise RuntimeError(f"No train_chunk_* directories found in {train_path}")

print(f"[INFO] Training chunks: {train_chunk_names}")

train_datasets = [
    datasets.Dataset.load_from_disk(os.path.join(train_path, name))
    for name in train_chunk_names
]
train_ds = datasets.concatenate_datasets(train_datasets)

val_ds = datasets.Dataset.load_from_disk(
    os.path.join(dataset_dir, "val-blocks")
)

print(f"[INFO] Training samples: {len(train_ds):,}")
print(f"[INFO] Validation samples: {len(val_ds):,}")

cols_to_remove = [
    "DateTimeFormatted",
    "Ne1",
    "Pv1",
    "Te2",
    "Ne2",
    "Pv2",
    "Te3",
    "Ne3",
    "Pv3",
    "I1",
    "I2",
    "I3",
]

train_ds = train_ds.remove_columns(
    [c for c in cols_to_remove if c in train_ds.column_names]
)
val_ds = val_ds.remove_columns(
    [c for c in cols_to_remove if c in val_ds.column_names]
)

expected_raw_columns = set(raw_input_columns + output_columns)

assert set(train_ds.column_names) == expected_raw_columns, (
    "Mismatch in training columns.\n"
    f"Missing: {sorted(expected_raw_columns - set(train_ds.column_names))}\n"
    f"Unexpected: {sorted(set(train_ds.column_names) - expected_raw_columns)}"
)
assert set(val_ds.column_names) == expected_raw_columns, (
    "Mismatch in validation columns.\n"
    f"Missing: {sorted(expected_raw_columns - set(val_ds.column_names))}\n"
    f"Unexpected: {sorted(set(val_ds.column_names) - expected_raw_columns)}"
)

def scan_column_range(
    ds,
    column,
    split_name,
    batch_rows=100_000,
):
    """
    Scan one numeric column without materializing unrelated dataset columns.

    Hugging Face Dataset slicing can otherwise decode the full table before
    selecting the requested column, which is extremely expensive for this
    ~200-column dataset.
    """
    minimum = math.inf
    maximum = -math.inf
    count_below_zero = 0
    count_at_or_above_te_max = 0

    column_ds = ds.with_format(
        "numpy",
        columns=[column],
        output_all_columns=False,
    )

    total_batches = math.ceil(
        len(ds) / batch_rows
    )

    print(
        f"[INFO] Scanning {split_name} {column} range: "
        f"{len(ds):,} rows in {total_batches} batch(es)",
        flush=True,
    )

    for batch_idx, start in enumerate(
        range(0, len(ds), batch_rows),
        start=1,
    ):
        stop = min(
            start + batch_rows,
            len(ds),
        )

        values = np.asarray(
            column_ds[start:stop][column],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains NaN or infinite values in rows "
                f"{start}:{stop}."
            )

        minimum = min(
            minimum,
            float(values.min()),
        )
        maximum = max(
            maximum,
            float(values.max()),
        )

        if column == "Te1":
            count_below_zero += int(
                (values < TE_MIN_K).sum()
            )
            count_at_or_above_te_max += int(
                (values >= TE_MAX_K_EXCLUSIVE).sum()
            )

        if (
            batch_idx == 1
            or batch_idx == total_batches
            or batch_idx % 10 == 0
        ):
            print(
                f"[INFO] {split_name} {column} scan: "
                f"batch {batch_idx}/{total_batches} "
                f"({stop:,}/{len(ds):,} rows)",
                flush=True,
            )

    return {
        "min": minimum,
        "max": maximum,
        "below_te_min": count_below_zero,
        "at_or_above_te_max": count_at_or_above_te_max,
    }

def assert_kp_range(ds, split_name):
    stats = scan_column_range(ds, "Kp_index", split_name)

    print(
        f"[INFO] {split_name} Kp range: "
        f"{stats['min']} to {stats['max']}"
    )

    assert stats["min"] >= 0.0, (
        f"{split_name} Kp_index minimum {stats['min']} is below 0."
    )
    assert stats["max"] <= 90.0, (
        f"{split_name} Kp_index maximum {stats['max']} exceeds 90."
    )

    return stats

def assert_te_classification_range(ds, split_name):
    stats = scan_column_range(ds, "Te1", split_name)

    print(
        f"[INFO] {split_name} Te1 range: "
        f"{stats['min']} to {stats['max']} K"
    )

    if not args.continuous:
        assert stats["below_te_min"] == 0, (
            f"{split_name} contains {stats['below_te_min']} Te1 values "
            f"below {TE_MIN_K} K. The fixed classification bins cannot "
            "represent these targets without clipping."
        )
        assert stats["at_or_above_te_max"] == 0, (
            f"{split_name} contains {stats['at_or_above_te_max']} Te1 values "
            f">= {TE_MAX_K_EXCLUSIVE} K. The fixed {TE_NUM_CLASSES}-class "
            "design cannot represent these targets without clipping. "
            "Do not silently clamp them; revise the bin range or filtering "
            "policy explicitly."
        )

    return stats

train_kp_stats = assert_kp_range(train_ds, "train")
val_kp_stats = assert_kp_range(val_ds, "validation")

train_te_stats = assert_te_classification_range(train_ds, "train")
val_te_stats = assert_te_classification_range(val_ds, "validation")

def normalize_batch(batch):
    """
    Apply all deterministic feature transforms from constants.py.

    Keeping these transforms centralized ensures that training, evaluation,
    and inference use identical scalar and circular encodings.
    """
    for col, norm_func in constants.NORMALIZATIONS.items():
        if col in batch:
            batch[col] = norm_func(batch[col])

    for source_col, encode_func in constants.CIRCULAR_ENCODINGS.items():
        if source_col in batch:
            batch.update(
                encode_func(batch[source_col])
            )

    return batch

train_ds = train_ds.map(
    normalize_batch,
    batched=True,
    batch_size=10_000,
    num_proc=num_workers,
)
val_ds = val_ds.map(
    normalize_batch,
    batched=True,
    batch_size=10_000,
    num_proc=num_workers,
)

columns_to_standardize = [
    col
    for col in input_columns
    if (
        col.startswith("AL_index")
        or col.startswith("SYM_H")
        or col.startswith("f107_index")
    )
]

index_groups = {
    "AL_index": [
        col for col in columns_to_standardize
        if col.startswith("AL_index")
    ],
    "SYM_H": [
        col for col in columns_to_standardize
        if col.startswith("SYM_H")
    ],
    "f107_index": [
        col for col in columns_to_standardize
        if col.startswith("f107_index")
    ],
}

def calculate_group_stats(
    ds,
    group_cols,
    group_name,
    batch_rows=50_000,
):
    """
    Calculate one population mean/std across all history columns in a feature
    family using only the training split.

    Only the requested family columns are decoded from Arrow, avoiding repeated
    materialization of the full ~200-column dataset.
    """
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    stats_ds = ds.with_format(
        "numpy",
        columns=group_cols,
        output_all_columns=False,
    )

    total_batches = math.ceil(
        len(ds) / batch_rows
    )

    print(
        f"[INFO] Calculating {group_name} normalization statistics: "
        f"{len(ds):,} rows, {len(group_cols)} columns, "
        f"{total_batches} batch(es)",
        flush=True,
    )

    for batch_idx, start in enumerate(
        range(0, len(ds), batch_rows),
        start=1,
    ):
        stop = min(
            start + batch_rows,
            len(ds),
        )

        batch = stats_ds[start:stop]

        for col in group_cols:
            values = np.asarray(
                batch[col],
                dtype=np.float64,
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite values found while computing normalization "
                    f"statistics for {col}."
                )

            total_sum += float(
                values.sum(dtype=np.float64)
            )
            total_sum_sq += float(
                np.square(
                    values,
                    dtype=np.float64,
                ).sum(dtype=np.float64)
            )
            total_count += int(
                values.size
            )

        if (
            batch_idx == 1
            or batch_idx == total_batches
            or batch_idx % 10 == 0
        ):
            print(
                f"[INFO] {group_name} stats scan: "
                f"batch {batch_idx}/{total_batches} "
                f"({stop:,}/{len(ds):,} rows)",
                flush=True,
            )

    if total_count == 0:
        raise ValueError(
            "Cannot compute normalization statistics from zero values."
        )

    mean = total_sum / total_count

    variance = max(
        total_sum_sq / total_count
        - mean * mean,
        0.0,
    )

    std = math.sqrt(
        variance
    )

    assert np.isfinite(mean), (
        "Normalization mean is not finite."
    )
    assert np.isfinite(std), (
        "Normalization std is not finite."
    )
    assert std > 0.0, (
        "Normalization std must be > 0."
    )

    return float(mean), float(std)

os.makedirs("checkpoints", exist_ok=True)

means = {}
stds = {}

for group_name, group_cols_for_stats in tqdm(
    index_groups.items(),
    desc="Calculating fresh group normalization stats",
):
    mean, std = calculate_group_stats(
        train_ds,
        group_cols_for_stats,
        group_name,
    )
    means[group_name] = mean
    stds[group_name] = std

    print(
        f"[INFO] {group_name}: mean={mean:.8f}, std={std:.8f}"
    )

    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert std > 0.0

stats_file = (
    f"checkpoints/"
    f"{base_model_name}_{block_minutes}m_s{split_seed}_norm_stats.json"
)

normalization_metadata = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_dir": dataset_dir,
    "train_size": len(train_ds),
    "validation_size": len(val_ds),
    "block_minutes": block_minutes,
    "split_seed": split_seed,
    "source": "freshly recomputed from current training split",
    "mean": means,
    "std": stds,
}

with open(stats_file, "w") as f:
    json.dump(normalization_metadata, f, indent=2)

print(f"[INFO] Wrote fresh normalization statistics to {stats_file}")

group_cols = [
    col
    for cols in index_groups.values()
    for col in cols
]

def normalize_group(batch):
    for col in group_cols:
        group_name = (
            "_".join(col.split("_")[:-1])
            if col.split("_")[-1].isdigit()
            else col
        )

        values = np.asarray(
            batch[col],
            dtype=np.float32,
        )

        batch[col] = (
            (values - means[group_name])
            / stds[group_name]
        ).astype(np.float32)

    return batch

train_ds = train_ds.map(
    normalize_group,
    batched=True,
    batch_size=10_000,
    num_proc=num_workers,
)
val_ds = val_ds.map(
    normalize_group,
    batched=True,
    batch_size=10_000,
    num_proc=num_workers,
)

def convert_to_tensor(row):
    input_ids = torch.tensor(
        [row[col] for col in input_columns],
        dtype=torch.float32,
    )

    target_temperature = torch.tensor(
        row["Te1"],
        dtype=torch.float32,
    )

    if args.continuous:
        label = target_temperature.clone()
    else:
        label = torch.floor(
            (target_temperature - TE_MIN_K)
            / TE_BIN_WIDTH_K
        ).long()

    return {
        "input_ids": input_ids,
        "label": label,
        "target_temperature": target_temperature,
    }

val_columns_before_tensor = list(val_ds.column_names)
train_columns_before_tensor = list(train_ds.column_names)

val_ds = val_ds.map(
    convert_to_tensor,
    num_proc=num_workers,
    remove_columns=val_columns_before_tensor,
)
train_ds = train_ds.map(
    convert_to_tensor,
    num_proc=num_workers,
    remove_columns=train_columns_before_tensor,
)

val_ds.set_format(type="torch")
train_ds.set_format(type="torch")

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    generator=data_loader_generator,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

steps_per_epoch = len(train_loader)

eval_every_steps = max(
    1,
    math.ceil(
        steps_per_epoch / evals_per_epoch
    ),
)

actual_evaluations_per_epoch = math.ceil(
    steps_per_epoch / eval_every_steps
)

print(
    f"[INFO] Training steps per epoch: {steps_per_epoch:,}"
)
print(
    f"[INFO] Target validation passes per epoch: {evals_per_epoch}"
)
print(
    f"[INFO] Validation interval: every {eval_every_steps} optimizer step(s)"
)
print(
    f"[INFO] Actual validation passes per epoch: "
    f"{actual_evaluations_per_epoch}"
)

output_dim = 1 if args.continuous else TE_NUM_CLASSES

model = models.FeedForwardNetwork(
    len(input_columns),
    2048,
    output_dim,
).to("cuda")

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"[INFO] Trainable parameters: {trainable_parameters:,}")

assert model.layer1.in_features == len(input_columns)
assert model.layer6.out_features == output_dim

with torch.no_grad():
    smoke_input = torch.zeros(
        2,
        len(input_columns),
        dtype=torch.float32,
        device="cuda",
    )
    smoke_output = model(smoke_input)

expected_smoke_shape = (
    (2, 1)
    if args.continuous
    else (2, TE_NUM_CLASSES)
)
assert tuple(smoke_output.shape) == expected_smoke_shape
assert torch.isfinite(smoke_output).all()

criterion = (
    nn.MSELoss()
    if args.continuous
    else None
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=max_lr,
)

scheduler_patience_evals = max(
    1,
    actual_evaluations_per_epoch,
)

early_stop_patience_evals = max(
    1,
    4 * actual_evaluations_per_epoch,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=scheduler_patience_evals,
    threshold=1e-4,
    threshold_mode="rel",
    min_lr=min_lr,
)

print(
    "[INFO] Scheduler/checkpoint/early-stop metric: validation RMSE"
)
print(
    f"[INFO] Scheduler patience: "
    f"{scheduler_patience_evals} validation evaluations "
    f"(~1 epoch)"
)
print(
    f"[INFO] Early-stop patience: "
    f"{early_stop_patience_evals} validation evaluations "
    f"(~4 epochs after required LR reductions)"
)

wandb.init(
    project="clare",
    name=model_name,
    config={
        "base_model_name": base_model_name,
        "model_name": model_name,
        "block_minutes": block_minutes,
        "split_seed": split_seed,
        "train_seed": train_seed,
        "mode": (
            "continuous"
            if args.continuous
            else "classification_based_regression"
        ),
        "dataset_dir": dataset_dir,
        "dataset_size": len(train_ds),
        "validation_size": len(val_ds),
        "input_size": len(input_columns),
        "input_columns": input_columns,
        "hidden_size": 2048,
        "output_dim": output_dim,
        "trainable_parameters": trainable_parameters,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "evals_per_epoch_target": evals_per_epoch,
        "eval_every_steps": eval_every_steps,
        "evaluations_per_epoch_actual": actual_evaluations_per_epoch,
        "scheduler_patience_evals": scheduler_patience_evals,
        "early_stop_patience_evals": early_stop_patience_evals,
        "scheduler_metric": "val_rmse",
        "checkpoint_selection_metric": "val_rmse",
        "early_stopping_metric": "val_rmse",
        "te_min_k": TE_MIN_K,
        "te_max_k_exclusive": TE_MAX_K_EXCLUSIVE,
        "te_bin_width_k": TE_BIN_WIDTH_K,
        "te_num_classes": TE_NUM_CLASSES,
        "classification_loss": "gaussian_soft_target_cross_entropy",
        "soft_target_sigma_k": SOFT_TARGET_SIGMA_K,
        "classification_decode": "softmax_weighted_bin_center_expectation",
        "normalization_stats_file": stats_file,
        "normalization_means": means,
        "normalization_stds": stds,
        "circular_encoding": {
            "GCLON": ["sin", "cos"],
            "GMLT": ["sin", "cos"],
            "XXLON": ["sin", "cos"],
        },
        "transform_source": "constants.py",
        "altitude_normalization": {
            "min_km": constants.ALTITUDE_MIN_KM,
            "max_km": constants.ALTITUDE_MAX_KM,
            "output_range": [-1.0, 1.0],
        },
        "kp_normalization": {
            "min": constants.KP_MIN,
            "max": constants.KP_MAX,
            "output_range": [-1.0, 1.0],
        },
        "deterministic_algorithms": True,
        "train_kp_min": train_kp_stats["min"],
        "train_kp_max": train_kp_stats["max"],
        "val_kp_min": val_kp_stats["min"],
        "val_kp_max": val_kp_stats["max"],
        "train_te_min": train_te_stats["min"],
        "train_te_max": train_te_stats["max"],
        "val_te_min": val_te_stats["min"],
        "val_te_max": val_te_stats["max"],
    },
)

wandb.define_metric("total_steps")
wandb.define_metric("epoch")
wandb.define_metric("epoch_progress")

wandb.define_metric("*", step_metric="total_steps")

wandb.summary["normalization_stats_generated_utc"] = (
    normalization_metadata["generated_utc"]
)

def physical_regression_metrics(pred_temps, true_temps):
    pred_temps = np.asarray(pred_temps, dtype=np.float64)
    true_temps = np.asarray(true_temps, dtype=np.float64)

    errors = pred_temps - true_temps

    rmse = float(
        np.sqrt(np.mean(np.square(errors)))
    )
    mae = float(
        np.mean(np.abs(errors))
    )
    bias = float(
        np.mean(errors)
    )
    acc_10pct = float(
        np.mean(
            np.abs(errors)
            <= 0.10 * np.abs(true_temps)
        )
    )

    if len(true_temps) > 1 and np.std(true_temps) > 0:
        r2 = float(
            r2_score(true_temps, pred_temps)
        )
    else:
        r2 = float("nan")

    if (
        len(true_temps) > 1
        and np.std(true_temps) > 0
        and np.std(pred_temps) > 0
    ):
        pearson_r = float(
            np.corrcoef(
                true_temps,
                pred_temps,
            )[0, 1]
        )
    else:
        pearson_r = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "pearson_r": pearson_r,
        "acc_10pct": acc_10pct,
    }

def evaluate_model(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_pred_classes = []
    all_labels = []
    all_pred_temps = []
    all_true_temps = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["input_ids"].to(
                "cuda",
                non_blocking=True,
            )
            y = batch["label"].to(
                "cuda",
                non_blocking=True,
            )

            raw_te = batch["target_temperature"].to(
                "cuda",
                non_blocking=True,
            )

            y_pred = model(x)

            if args.continuous:
                preds = y_pred.squeeze(-1)
                loss = criterion(preds, y)
                pred_temps = preds
            else:
                loss, pred_temps = classification_loss_and_temperature(
                    y_pred,
                    raw_te,
                )

                pred_classes = y_pred.argmax(dim=1)

                all_pred_classes.append(
                    pred_classes.cpu()
                )
                all_labels.append(
                    y.cpu()
                )

            batch_size_actual = int(x.shape[0])
            total_loss += (
                float(loss.item())
                * batch_size_actual
            )
            total_examples += batch_size_actual

            all_pred_temps.append(
                pred_temps.cpu()
            )
            all_true_temps.append(
                raw_te.cpu()
            )

    if total_examples == 0:
        raise RuntimeError("Cannot evaluate an empty DataLoader.")

    pred_temps = torch.cat(
        all_pred_temps
    ).numpy()
    true_temps = torch.cat(
        all_true_temps
    ).numpy()

    metrics = {
        "loss": total_loss / total_examples,
        **physical_regression_metrics(
            pred_temps,
            true_temps,
        ),
    }

    if not args.continuous:
        pred_classes = torch.cat(
            all_pred_classes
        ).numpy()
        labels = torch.cat(
            all_labels
        ).numpy()

        metrics["accuracy"] = float(
            accuracy_score(
                labels,
                pred_classes,
            )
        )
        metrics["macro_f1"] = float(
            f1_score(
                labels,
                pred_classes,
                average="macro",
                zero_division=0,
            )
        )

    return metrics

def wandb_validation_payload(
    metrics,
    *,
    epoch_number,
    epoch_progress,
    batch_in_epoch,
    total_steps,
    lr,
    lr_reduction_count,
):
    payload = {
        "val_loss": metrics["loss"],
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_bias": metrics["bias"],
        "val_r2": metrics["r2"],
        "val_pearson_r": metrics["pearson_r"],
        "val_acc_10pct": metrics["acc_10pct"],
        "learning_rate": lr,
        "lr_reduction_count": lr_reduction_count,
        "epoch": epoch_number,
        "epoch_progress": epoch_progress,
        "batch_in_epoch": batch_in_epoch,
        "total_steps": total_steps,
    }

    if not args.continuous:
        payload["val_accuracy"] = metrics["accuracy"]
        payload["val_macro_f1"] = metrics["macro_f1"]

    return payload

def make_checkpoint_metadata(
    stage,
    metrics,
    *,
    epoch_number,
    total_steps,
    lr_reduction_count,
    early_stopped,
):
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "selection_criterion": (
            "minimum validation RMSE"
            if stage in {"best", "selected_best"}
            else "last training state"
        ),
        "model_name": model_name,
        "base_model_name": base_model_name,
        "mode": (
            "continuous"
            if args.continuous
            else "classification_based_regression"
        ),
        "dataset_dir": dataset_dir,
        "block_minutes": block_minutes,
        "split_seed": split_seed,
        "train_seed": train_seed,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "input_size": len(input_columns),
        "input_columns": input_columns,
        "hidden_size": 2048,
        "output_dim": output_dim,
        "trainable_parameters": trainable_parameters,
        "batch_size": batch_size,
        "epoch": epoch_number,
        "total_steps": total_steps,
        "evals_per_epoch_target": evals_per_epoch,
        "eval_every_steps": eval_every_steps,
        "evaluations_per_epoch_actual": actual_evaluations_per_epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "lr_reduction_count": lr_reduction_count,
        "early_stopped": early_stopped,
        "scheduler_metric": "val_rmse",
        "checkpoint_selection_metric": "val_rmse",
        "early_stopping_metric": "val_rmse",
        "min_delta_rmse_k": min_delta_rmse_k,
        "target_binning": {
            "min_k": TE_MIN_K,
            "max_k_exclusive": TE_MAX_K_EXCLUSIVE,
            "bin_width_k": TE_BIN_WIDTH_K,
            "num_classes": TE_NUM_CLASSES,
            "loss": "gaussian_soft_target_cross_entropy",
            "soft_target_sigma_k": SOFT_TARGET_SIGMA_K,
            "decode": "softmax_weighted_bin_center_expectation",
        },
        "circular_encoding": {
            "GCLON": ["sin", "cos"],
            "GMLT": ["sin", "cos"],
            "XXLON": ["sin", "cos"],
        },
        "transform_source": "constants.py",
        "altitude_normalization": {
            "min_km": constants.ALTITUDE_MIN_KM,
            "max_km": constants.ALTITUDE_MAX_KM,
            "output_range": [-1.0, 1.0],
        },
        "kp_normalization": {
            "min": constants.KP_MIN,
            "max": constants.KP_MAX,
            "output_range": [-1.0, 1.0],
        },
        "normalization": normalization_metadata,
        "metrics": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (float, int, np.floating, np.integer),
                )
                else value
            )
            for key, value in metrics.items()
        },
    }

def save_checkpoint_metadata(path, metadata):
    with open(path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            allow_nan=True,
        )

total_steps = 0
best_val_rmse = float("inf")

min_delta_rmse_k = 1.0

best_step = 0
best_epoch = 0.0
best_metrics = None

evals_since_improvement = 0
lr_reduction_count = 0
max_lr_reductions = 4

stop_training = False
early_stopped = False

best_checkpoint_path = (
    f"./checkpoints/{model_name}_best.pth"
)
best_metadata_path = (
    f"./checkpoints/{model_name}_best_metadata.json"
)
last_checkpoint_path = (
    f"./checkpoints/{model_name}_last.pth"
)
last_metadata_path = (
    f"./checkpoints/{model_name}_last_metadata.json"
)
canonical_checkpoint_path = (
    f"./checkpoints/{model_name}.pth"
)
canonical_metadata_path = (
    f"./checkpoints/{model_name}_metadata.json"
)

for epoch_idx in range(num_epochs):
    model.train()

    progress = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Epoch {epoch_idx + 1}/{num_epochs}",
    )

    for batch_in_epoch, batch in progress:
        x = batch["input_ids"].to(
            "cuda",
            non_blocking=True,
        )
        y = batch["label"].to(
            "cuda",
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        y_pred = model(x)

        if args.continuous:
            loss = criterion(
                y_pred.squeeze(-1),
                y,
            )
        else:
            raw_te = batch["target_temperature"].to(
                "cuda",
                non_blocking=True,
            )
            loss, _ = classification_loss_and_temperature(
                y_pred,
                raw_te,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {total_steps + 1}: "
                f"{loss.item()}"
            )

        loss.backward()

        for name, parameter in model.named_parameters():
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ):
                raise FloatingPointError(
                    f"Non-finite gradient in parameter {name} "
                    f"at step {total_steps + 1}."
                )

        optimizer.step()

        total_steps += 1

        epoch_progress = (
            epoch_idx
            + batch_in_epoch / len(train_loader)
        )
        epoch_number = epoch_idx + 1

        if total_steps % log_every_step == 0:
            wandb.log(
                {
                    "train_loss": float(loss.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch_number,
                    "epoch_progress": epoch_progress,
                    "batch_in_epoch": batch_in_epoch,
                    "total_steps": total_steps,
                },
            )

        should_validate = (
            total_steps % eval_every_steps == 0
            or batch_in_epoch == len(train_loader)
        )

        if should_validate:
            val_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
            )
            val_loss = val_metrics["loss"]
            val_rmse = val_metrics["rmse"]

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_rmse)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr < old_lr:
                lr_reduction_count += 1

            wandb.log(
                wandb_validation_payload(
                    val_metrics,
                    epoch_number=epoch_number,
                    epoch_progress=epoch_progress,
                    batch_in_epoch=batch_in_epoch,
                    total_steps=total_steps,
                    lr=new_lr,
                    lr_reduction_count=lr_reduction_count,
                ),
            )

            if (
                best_val_rmse == float("inf")
                or val_rmse < best_val_rmse - min_delta_rmse_k
            ):
                best_val_rmse = val_rmse
                best_step = total_steps
                best_epoch = epoch_progress
                best_metrics = dict(val_metrics)
                evals_since_improvement = 0

                torch.save(
                    model.state_dict(),
                    best_checkpoint_path,
                )

                save_checkpoint_metadata(
                    best_metadata_path,
                    make_checkpoint_metadata(
                        "best",
                        best_metrics,
                        epoch_number=best_epoch,
                        total_steps=best_step,
                        lr_reduction_count=lr_reduction_count,
                        early_stopped=False,
                    ),
                )

                wandb.summary["best_val_rmse"] = best_val_rmse
                wandb.summary["best_val_loss"] = best_metrics["loss"]
                wandb.summary["best_step"] = best_step
                wandb.summary["best_epoch"] = best_epoch

                for metric_name, metric_value in best_metrics.items():
                    wandb.summary[
                        f"best_{metric_name}"
                    ] = metric_value
            else:
                evals_since_improvement += 1

            progress.set_postfix(
                train_loss=f"{loss.item():.4f}",
                val_loss=f"{val_loss:.4f}",
                val_rmse=f"{val_rmse:.1f}",
                lr=f"{new_lr:.2e}",
            )

            if (
                lr_reduction_count >= max_lr_reductions
                and evals_since_improvement
                >= early_stop_patience_evals
            ):
                print(
                    "[INFO] Early stopping: validation RMSE has not "
                    "improved after the required LR reductions and "
                    f"{evals_since_improvement} validation evaluations."
                )
                early_stopped = True
                stop_training = True
                break

            model.train()

    if stop_training:
        break

last_metrics = evaluate_model(
    model,
    val_loader,
    criterion,
)

torch.save(
    model.state_dict(),
    last_checkpoint_path,
)

save_checkpoint_metadata(
    last_metadata_path,
    make_checkpoint_metadata(
        "last",
        last_metrics,
        epoch_number=epoch_progress,
        total_steps=total_steps,
        lr_reduction_count=lr_reduction_count,
        early_stopped=early_stopped,
    ),
)

if not os.path.exists(best_checkpoint_path):
    raise RuntimeError(
        "No best checkpoint was created. "
        "At least one validation evaluation is required."
    )

best_state_dict = torch.load(
    best_checkpoint_path,
    map_location="cuda",
)
model.load_state_dict(best_state_dict)

selected_metrics = evaluate_model(
    model,
    val_loader,
    criterion,
)

torch.save(
    model.state_dict(),
    canonical_checkpoint_path,
)

selected_metadata = make_checkpoint_metadata(
    "selected_best",
    selected_metrics,
    epoch_number=best_epoch,
    total_steps=best_step,
    lr_reduction_count=lr_reduction_count,
    early_stopped=early_stopped,
)

selected_metadata["best_checkpoint_path"] = best_checkpoint_path
selected_metadata["last_checkpoint_path"] = last_checkpoint_path

save_checkpoint_metadata(
    canonical_metadata_path,
    selected_metadata,
)

wandb.summary["block_minutes"] = block_minutes
wandb.summary["split_seed"] = split_seed
wandb.summary["train_seed"] = train_seed
wandb.summary["mode"] = (
    "continuous"
    if args.continuous
    else "classification_based_regression"
)
wandb.summary["input_size"] = len(input_columns)
wandb.summary["trainable_parameters"] = trainable_parameters
wandb.summary["total_steps"] = total_steps
wandb.summary["epochs_completed"] = int(math.ceil(epoch_progress))
wandb.summary["final_epoch_progress"] = float(epoch_progress)
wandb.summary["lr_reduction_count"] = lr_reduction_count
wandb.summary["final_lr"] = optimizer.param_groups[0]["lr"]
wandb.summary["early_stopped"] = early_stopped
wandb.summary["best_step"] = best_step
wandb.summary["best_epoch"] = best_epoch
wandb.summary["best_val_rmse"] = best_val_rmse
wandb.summary["best_val_loss"] = best_metrics["loss"]
wandb.summary["selection_metric"] = "val_rmse"
wandb.summary["min_delta_rmse_k"] = min_delta_rmse_k

for metric_name, metric_value in last_metrics.items():
    wandb.summary[
        f"last_{metric_name}"
    ] = metric_value

for metric_name, metric_value in selected_metrics.items():
    wandb.summary[
        f"selected_best_{metric_name}"
    ] = metric_value

print("\n[INFO] Training complete.")
print(
    f"[INFO] Best checkpoint selected at step {best_step}, "
    f"epoch progress {best_epoch:.6f}, "
    f"validation RMSE {best_val_rmse:.3f} K, "
    f"validation loss {best_metrics['loss']:.6f}"
)
print(
    f"[INFO] Canonical selected checkpoint: "
    f"{canonical_checkpoint_path}"
)
print(
    f"[INFO] Best checkpoint: "
    f"{best_checkpoint_path}"
)
print(
    f"[INFO] Last training-state checkpoint: "
    f"{last_checkpoint_path}"
)
print(
    f"[INFO] Checkpoint metadata: "
    f"{canonical_metadata_path}"
)

wandb.finish()
