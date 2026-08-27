import json
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wandb
import datasets

import models.feed_forward as models
import constants

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description="Train model on seed-split dataset.")
parser.add_argument("--seed", type=int, default=int(os.environ.get("SPLIT_SEED", 0)), help="Seed ID for dataset (0, 1, or 2)")
parser.add_argument("--block-hours", type=int, default=int(os.environ.get("BLOCK_HOURS", 12)))
parser.add_argument("--continuous", action="store_true", help="Train continuous regression variant (MSE loss, 1 output) instead of classification.")
args = parser.parse_args()

seed = args.seed
block_hours = args.block_hours
base_model_name = '2_0'
# Distinct model name so continuous doesn't overwrite classification checkpoints
model_name = (
    f"{base_model_name}_{block_hours}h_s{seed}_continuous"
    if args.continuous
    else f"{base_model_name}_{block_hours}h_s{seed}"
)

dataset_dir = f"dataset/processed_dataset_blocksplit_{block_hours}h_s{seed}"

print(f"[INFO] Running training for Seed {seed}")
print(f"[INFO] Mode: {'Continuous Regression' if args.continuous else 'Classification'}")
print(f"[INFO] Model checkpoint tag: {model_name}")
print(f"[INFO] Reading dataset from: {dataset_dir}")

# Use allocated CPUs instead of total node CPUs
num_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 8

# Hyperparameters
batch_size = 512
num_epochs = 100
max_lr = 8e-4
min_lr = max_lr / 1000
log_every_step = 10

input_columns = [
    'Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 
    'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 
    'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 
    'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index'
]
output_columns = ['Te1']

# Load train chunks dynamically
train_path = os.path.join(dataset_dir, "train_chunks")
train_datasets = []
for folder in sorted(os.listdir(train_path)):
    chunk_full_path = os.path.join(train_path, folder)
    if os.path.isdir(chunk_full_path):
        chunk = datasets.Dataset.load_from_disk(chunk_full_path)
        train_datasets.append(chunk)

train_ds = datasets.concatenate_datasets(train_datasets)
print("Length train ds:", len(train_ds))

val_ds = datasets.Dataset.load_from_disk(os.path.join(dataset_dir, "val-blocks"))

# Strip metadata columns
cols_to_remove = ['DateTimeFormatted', 'Ne1', 'Pv1', 'Te2', 'Ne2', 'Pv2', 'Te3', 'Ne3', 'Pv3', 'I1', 'I2', 'I3']
train_ds = train_ds.remove_columns([c for c in cols_to_remove if c in train_ds.column_names])
val_ds = val_ds.remove_columns([c for c in cols_to_remove if c in val_ds.column_names])

all_columns = input_columns + output_columns
assert set(train_ds.column_names) == set(all_columns), "Mismatch in train columns"
assert set(val_ds.column_names) == set(all_columns), "Mismatch in val columns"

def normalize_batch(batch):
    for col, norm_func in constants.NORMALIZATIONS.items():
        if col in batch:
            batch[col] = norm_func(batch[col])
    return batch

train_ds = train_ds.map(normalize_batch, batched=True, batch_size=10000, num_proc=num_workers)
val_ds = val_ds.map(normalize_batch, batched=True, batch_size=10000, num_proc=num_workers)

# Normalization stats per seed
columns_to_normalize = [col for col in input_columns if col.startswith('AL_index') or col.startswith('SYM_H') or col.startswith('f107_index')]
index_groups = {
    'AL_index': [col for col in columns_to_normalize if col.startswith('AL_index')],
    'SYM_H': [col for col in columns_to_normalize if col.startswith('SYM_H')],
    'f107_index': [col for col in columns_to_normalize if col.startswith('f107_index')]
}

os.makedirs('checkpoints', exist_ok=True)
# Both classification and continuous share the exact same inputs
# We use the base model stats file so it can load the precomputed normalization constants
stats_file = (
    f'checkpoints/{base_model_name}_{block_hours}h_s{seed}_norm_stats.json'
)

if os.path.exists(stats_file):
    print(f"[INFO] Loading existing normalization stats from {stats_file}")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
        means = stats['mean']
        stds = stats['std']
else:
    means, stds = {}, {}
    for group_name, group_cols in tqdm(index_groups.items(), desc="Calculating group stats"):
        group_values = np.concatenate([train_ds.with_format("pandas")[col].values for col in group_cols])
        means[group_name] = float(np.mean(group_values))
        stds[group_name] = float(np.std(group_values))

    with open(stats_file, 'w') as f:
        json.dump({'mean': means, 'std': stds}, f)

group_cols = [col for cols in index_groups.values() for col in cols]
def normalize_group(batch):
    for col in group_cols:
        group_name = '_'.join(col.split('_')[:-1]) if col.split('_')[-1].isdigit() else col
        values = np.array(batch[col], dtype=np.float32)
        batch[col] = (values - means[group_name]) / stds[group_name]
    return batch

train_ds = train_ds.map(normalize_group, batched=True, batch_size=10000, num_proc=num_workers)
val_ds = val_ds.map(normalize_group, batched=True, batch_size=10000, num_proc=num_workers)

def convert_to_tensor(row):
    input_ids = torch.tensor([row[col] for col in input_columns], dtype=torch.float32)
    label = torch.tensor([row[col] for col in output_columns], dtype=torch.float32)
    
    if args.continuous:
        # Keep as raw temperature values (no binning)
        label = label.squeeze() 
    else:
        # CrossEntropy class binning
        label = (label // 100).clamp(0, 149).long().squeeze()
        
    return {"input_ids": input_ids, "label": label}

val_ds = val_ds.map(convert_to_tensor, num_proc=num_workers, remove_columns=all_columns)
train_ds = train_ds.map(convert_to_tensor, num_proc=num_workers, remove_columns=all_columns)

val_ds.set_format(type="torch")
train_ds.set_format(type="torch")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Initialize Model Architecture based on continuous vs classification flag
output_dim = 1 if args.continuous else 150
model = models.FeedForwardNetwork(len(input_columns), 2048, output_dim).to("cuda")

if args.continuous:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.AdamW(model.parameters(), lr=max_lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    threshold=1e-4,
    threshold_mode="rel",
    min_lr=min_lr,
)

wandb.init(
    project="clare",
    name=model_name,
    config={
        "block_hours": block_hours,
        "seed": seed,
        "mode": "continuous" if args.continuous else "classification",
        "dataset_size": len(train_ds),
        "validation_size": len(val_ds),
    }
)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch["input_ids"].to("cuda")
            y = batch["label"].to("cuda")
            
            y_pred = model(x)
            
            # Align output shapes safely for MSE
            if args.continuous:
                preds = y_pred.squeeze(-1)
                loss = criterion(preds, y)

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
            else:
                loss = criterion(y_pred, y)

                all_preds.append(y_pred.argmax(dim=1).cpu())
                all_targets.append(y.cpu())
                
            total_loss += loss.item()
            
    metrics = {
        "loss": total_loss / len(data_loader)
    }

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    if args.continuous:
        metrics["rmse"] = float(np.sqrt(np.mean((preds - targets) ** 2)))
        metrics["acc_10pct"] = float(
            np.mean(np.abs(preds - targets) <= 0.10 * np.abs(targets))
        )
    else:
        metrics["accuracy"] = float(accuracy_score(targets, preds))
        metrics["macro_f1"] = float(f1_score(targets, preds, average="macro"))
        
        pred_temp = preds * 100.0 + 50.0
        true_temp = targets * 100.0 + 50.0
        metrics["acc_10pct"] = float(
            np.mean(np.abs(pred_temp - true_temp) <= 0.10 * np.abs(true_temp))
        )

    return metrics

total_steps = 0
eval_interval = (len(train_loader) + 2) // 3
best_val_loss = float("inf")
min_delta = 1e-4

best_step = 0
best_epoch = 0

best_acc_10pct = None
best_rmse = None
best_accuracy = None
best_macro_f1 = None

early_stop_patience = 12
evals_since_improvement = 0

lr_reduction_count = 0
max_lr_reductions = 4

stop_training = False
early_stopped = False

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x = batch["input_ids"].to("cuda")
        y = batch["label"].to("cuda")

        optimizer.zero_grad()
        y_pred = model(x)

        # Align output shapes safely for MSE
        if args.continuous:
            loss = criterion(y_pred.squeeze(-1), y)
        else:
            loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        total_steps += 1

        if total_steps % log_every_step == 0:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_steps": total_steps
            })

        if total_steps % eval_interval == 0:
            val_metrics = evaluate_model(model, val_loader, criterion)
            val_loss = val_metrics["loss"]

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                lr_reduction_count += 1
            wandb.log({
                "val_loss": val_loss,
                "learning_rate": new_lr,
                "lr_reduction_count": lr_reduction_count,
                "total_steps": total_steps
            })

            if args.continuous:
                wandb.log({
                    "val_rmse": val_metrics["rmse"],
                    "val_acc_10pct": val_metrics["acc_10pct"],
                    "total_steps": total_steps
                })
            else:
                wandb.log({
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_acc_10pct": val_metrics["acc_10pct"],
                    "total_steps": total_steps
                })

            # Save "best" model based on validation subset per Reviewer 2 comment
            if best_val_loss == float("inf") or val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_step = total_steps
                best_epoch = epoch + 1

                best_acc_10pct = val_metrics["acc_10pct"]

                if args.continuous:
                    best_rmse = val_metrics["rmse"]
                else:
                    best_accuracy = val_metrics["accuracy"]
                    best_macro_f1 = val_metrics["macro_f1"]

                evals_since_improvement = 0
                torch.save(model.state_dict(), f'./checkpoints/{model_name}_best.pth')
            else:
                evals_since_improvement += 1

            if (
                lr_reduction_count >= max_lr_reductions
                and evals_since_improvement >= early_stop_patience
            ):
                print(
                    f"Early stopping: no improvement after "
                    f"{lr_reduction_count} LR reductions."
                )
                early_stopped = True
                stop_training = True
                break
            
    if stop_training:
        break

# Final Evaluation & Save
final_metrics = evaluate_model(model, val_loader, criterion)
final_val_loss = final_metrics["loss"]
print(f"\nFinal Validation Loss (Seed {seed}): {final_val_loss:.4f}")
wandb.log({"final_val_loss": final_val_loss})

wandb.summary["block_hours"] = block_hours
wandb.summary["seed"] = seed
wandb.summary["mode"] = "continuous" if args.continuous else "classification"

wandb.summary["best_val_loss"] = best_val_loss
wandb.summary["best_acc_10pct"] = best_acc_10pct
wandb.summary["final_val_loss"] = final_val_loss

if args.continuous:
    wandb.summary["best_rmse"] = best_rmse
    wandb.summary["final_rmse"] = final_metrics["rmse"]
    wandb.summary["final_acc_10pct"] = final_metrics["acc_10pct"]
else:
    wandb.summary["best_accuracy"] = best_accuracy
    wandb.summary["best_macro_f1"] = best_macro_f1
    wandb.summary["final_accuracy"] = final_metrics["accuracy"]
    wandb.summary["final_macro_f1"] = final_metrics["macro_f1"]
    wandb.summary["final_acc_10pct"] = final_metrics["acc_10pct"]

wandb.summary["best_step"] = best_step
wandb.summary["best_epoch"] = best_epoch
wandb.summary["total_steps"] = total_steps

wandb.summary["lr_reduction_count"] = lr_reduction_count
wandb.summary["final_lr"] = optimizer.param_groups[0]["lr"]

wandb.summary["early_stopped"] = early_stopped

checkpoint_path = f'./checkpoints/{model_name}.pth'
torch.save(model.state_dict(), checkpoint_path)
print(f"[INFO] Saved final epoch model to {checkpoint_path}")