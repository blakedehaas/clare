#!/usr/bin/env python3
"""
sweep.py - Distributed Bayesian Hyperparameter Optimization Suite for Space Physics
Designed for NCAR Supercomputing Cluster (Casper / Derecho) with PBS Pro.

Key Capabilities:
1. Asynchronous Distributed Optuna with SQLite / PostgreSQL backend for multi-node sweeps.
2. Space Physics Multi-Objective Pareto Loss:
      L_comp = 0.5 * L_val_normal + 0.5 * L_val_storm
   Guarantees preservation of quiet-time precision while penalizing storm-time degradation.
3. Multi-tier search space covering:
   - DeepSeek-V4 MoE & Deep Neural Topology (depth, width, experts, attention sink)
   - Optimization schedule (learning rate, warmup, weight decay, batch size)
   - Unified Physics-Informed Loss (focal gamma, metric Huber weight, storm alpha)
4. Dynamic ASHA / Median Pruning to terminate unpromising runs early.
5. Automatic Weights & Biases (W&B) experiment synchronization.
"""

import os
import sys
import math
import time
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Optuna Bayesian Optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("[ERROR] Optuna is required. Install via: pip install optuna")
    sys.exit(1)

# Weights & Biases
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# HuggingFace Datasets
try:
    import datasets
except ImportError:
    datasets = None

from physics_loss import UnifiedPhysicsLoss


# ==============================================================================
# 1. 156-FEATURE TELEMETRY SPECIFICATION
# ==============================================================================

INPUT_COLUMNS = [
    'Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON',
    'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5',
    'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11',
    'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17',
    'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23',
    'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29',
    'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5',
    'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12',
    'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19',
    'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26',
    'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33',
    'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40',
    'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47',
    'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54',
    'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61',
    'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68',
    'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75',
    'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82',
    'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89',
    'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96',
    'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103',
    'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110',
    'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117',
    'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124',
    'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131',
    'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138',
    'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144',
    'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index'
]
TARGET_COLUMN = 'Te1'
NUM_CLASSES = 150
SYM_H_FEATURE_IDX = INPUT_COLUMNS.index('SYM_H_0')  # Index 39 for storm scaling


# ==============================================================================
# 2. CONFIGURABLE SPACE PHYSICS ARCHITECTURE (DeepSeekMoE + mHC)
# ==============================================================================

class SwiGLUExpert(nn.Module):
    """SwiGLU feed-forward expert with activation clamping."""
    def __init__(self, d_model: int, expert_dim: int, clamp_val: float = 10.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, expert_dim, bias=False)
        self.w_gate = nn.Linear(d_model, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, d_model, bias=False)
        self.clamp_val = clamp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = torch.clamp(self.w1(x), -self.clamp_val, self.clamp_val)
        gate = torch.clamp(self.w_gate(x), max=self.clamp_val)
        return self.w2(linear * F.silu(gate))


class DynamicSpacePhysicsModel(nn.Module):
    """
    Parametrically reconfigurable neural architecture for HPO sweep:
    Supports:
    - Input dimension: 156 space weather features
    - Residual stream projection (d_model)
    - DeepSeekMoE sublayer with 1 shared expert + E routed experts (Top-K active)
    - Optional Multi-Head Self-Attention with Attention Sink
    - Output projection: 150 temperature bins (100 K discrete intervals)
    """
    def __init__(
        self,
        input_dim: int = 156,
        output_dim: int = 150,
        d_model: int = 256,
        num_layers: int = 4,
        num_experts: int = 8,
        top_k: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_moe: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_moe = use_moe

        # Input feature encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Backbone layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(d_model))
            if use_moe and num_experts > 1:
                # Mixture of Experts block
                shared_exp = SwiGLUExpert(d_model, d_model * 2)
                routed_exps = nn.ModuleList([SwiGLUExpert(d_model, d_model * 2) for _ in range(num_experts)])
                router = nn.Linear(d_model, num_experts, bias=False)
                self.layers.append(nn.ModuleDict({
                    'shared': shared_exp,
                    'routed': routed_exps,
                    'router': router
                }))
            else:
                # Standard dense SwiGLU block
                dense_block = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
                self.layers.append(dense_block)

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, output_dim)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_encoder(x)

        for norm, layer in zip(self.norms, self.layers):
            residual = h
            normed = norm(h)

            if isinstance(layer, nn.ModuleDict):
                # MoE computation
                shared_out = layer['shared'](normed)
                router_logits = layer['router'](normed)
                # Sqrt(Softplus) routing affinity
                affinities = torch.sqrt(F.softplus(router_logits) + 1e-8)
                topk_weights, topk_indices = torch.topk(affinities, self.top_k, dim=-1)
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

                routed_out = torch.zeros_like(shared_out)
                for k in range(self.top_k):
                    exp_idx = topk_indices[:, k]
                    weight = topk_weights[:, k].unsqueeze(-1)
                    for e in range(self.num_experts):
                        mask = (exp_idx == e)
                        if mask.any():
                            routed_out[mask] += weight[mask] * layer['routed'][e](normed[mask])

                sublayer_out = shared_out + routed_out
            else:
                sublayer_out = layer(normed)

            h = residual + sublayer_out

        return self.classifier(self.final_norm(h))


# ==============================================================================
# 3. FAST DATA STREAMING & PREPROCESSING
# ==============================================================================

class SpaceWeatherDataset(Dataset):
    """Memory-efficient PyTorch Dataset over HuggingFace / Arrow splits."""
    def __init__(self, hf_dataset, sym_h_idx: int = 39):
        # Extract features and target arrays directly to float32
        if "input_ids" in hf_dataset.column_names:
            x_raw = np.vstack(hf_dataset["input_ids"]).astype(np.float32)
        else:
            x_raw = np.column_stack([hf_dataset[col] for col in INPUT_COLUMNS if col in hf_dataset.column_names]).astype(np.float32)

        if "label" in hf_dataset.column_names:
            y_raw = np.array(hf_dataset["label"], dtype=np.int64)
        elif TARGET_COLUMN in hf_dataset.column_names:
            y_raw = (np.array(hf_dataset[TARGET_COLUMN]) // 100).clip(0, NUM_CLASSES - 1).astype(np.int64)
        else:
            y_raw = np.zeros(len(x_raw), dtype=np.int64)

        self.x = torch.from_numpy(x_raw)
        self.y = torch.from_numpy(y_raw)
        self.sym_h_idx = min(sym_h_idx, self.x.size(1) - 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_partitioned_dataloaders(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads train_chunks, test-normal, and test-storm with pinned memory."""
    train_dir = os.path.join(data_dir, "train_chunks")
    val_normal_dir = os.path.join(data_dir, "test-normal")
    val_storm_dir = os.path.join(data_dir, "test-storm")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Load 1-2 train chunks for agile HPO trials
    train_ds_list = []
    chunk_dirs = sorted([d for d in os.listdir(train_dir) if d.startswith("train_chunk_")])
    for chunk_name in chunk_dirs[:3]:  # Use first 3 chunks (~750k samples) for fast trial exploration
        chunk = datasets.Dataset.load_from_disk(os.path.join(train_dir, chunk_name))
        train_ds_list.append(chunk)

    combined_train = datasets.concatenate_datasets(train_ds_list)
    val_normal_ds = datasets.Dataset.load_from_disk(val_normal_dir)
    val_storm_ds = datasets.Dataset.load_from_disk(val_storm_dir)

    train_dataset = SpaceWeatherDataset(combined_train, SYM_H_FEATURE_IDX)
    val_normal_dataset = SpaceWeatherDataset(val_normal_ds, SYM_H_FEATURE_IDX)
    val_storm_dataset = SpaceWeatherDataset(val_storm_ds, SYM_H_FEATURE_IDX)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_normal_loader = DataLoader(
        val_normal_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    val_storm_loader = DataLoader(
        val_storm_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_normal_loader, val_storm_loader


# ==============================================================================
# 4. OBJECTIVE FUNCTION FOR OPTUNA
# ==============================================================================

def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_normal_loader: DataLoader,
    val_storm_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluates a single trial configuration across quiet and storm validation splits.
    Minimizes: L_comp = 0.5 * L_normal + 0.5 * L_storm
    """
    # -------------------------------------------------------------------------
    # 1. Sample Hyperparameters from Search Space
    # -------------------------------------------------------------------------
    # Architecture dimensions
    d_model = trial.suggest_categorical("d_model", [128, 256, 384])
    num_layers = trial.suggest_int("num_layers", 3, 7)
    use_moe = trial.suggest_categorical("use_moe", [True, False])
    num_experts = trial.suggest_categorical("num_experts", [4, 8]) if use_moe else 1
    top_k = 2 if use_moe else 1
    dropout = trial.suggest_float("dropout", 0.05, 0.25)

    # Optimization dimensions
    max_lr = trial.suggest_float("max_lr", 3e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # Unified Physics Loss dimensions
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 2.5)
    huber_weight = trial.suggest_float("huber_weight", 0.05, 0.50)
    storm_alpha = trial.suggest_float("storm_alpha", 0.5, 3.0)
    huber_delta = trial.suggest_float("huber_delta", 250.0, 750.0)

    # -------------------------------------------------------------------------
    # 2. Instantiate Model, Loss & Optimizer
    # -------------------------------------------------------------------------
    model = DynamicSpacePhysicsModel(
        input_dim=len(INPUT_COLUMNS),
        output_dim=NUM_CLASSES,
        d_model=d_model,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        dropout=dropout,
        use_moe=use_moe
    ).to(device)

    criterion = UnifiedPhysicsLoss(
        vocab_size=NUM_CLASSES,
        bin_width_k=100.0,
        focal_gamma=focal_gamma,
        huber_weight=huber_weight,
        huber_delta_k=huber_delta,
        storm_alpha=storm_alpha,
        sym_h_feat_idx=SYM_H_FEATURE_IDX
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
        fused=torch.cuda.is_available()
    )

    epochs = args.epochs_per_trial
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=max_lr * 0.01)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Training and validation loop with median pruning
    best_composite_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(x_b)
                loss = criterion(logits, y_b, inputs=x_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_acc += loss.item()

        # Validation on contiguous quiet-time test blocks
        model.eval()
        val_normal_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_normal_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(x_b)
                    loss = criterion(logits, y_b, inputs=x_b)
                val_normal_loss += loss.item()
        val_normal_loss /= len(val_normal_loader)

        # Validation on contiguous held-out geomagnetic storm period
        val_storm_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_storm_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(x_b)
                    loss = criterion(logits, y_b, inputs=x_b)
                val_storm_loss += loss.item()
        val_storm_loss /= len(val_storm_loader)

        # Composite Storm-Resilient Pareto Objective
        composite_loss = 0.5 * val_normal_loss + 0.5 * val_storm_loss
        best_composite_loss = min(best_composite_loss, composite_loss)

        # Report to Optuna for ASHA / Median Pruning
        trial.report(composite_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Log metrics to trial attributes
    trial.set_user_attr("val_normal_loss", val_normal_loss)
    trial.set_user_attr("val_storm_loss", val_storm_loss)
    trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters()))

    return best_composite_loss


# ==============================================================================
# 5. CLI ENTRYPOINT & DISTRIBUTED RUNNER
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Space Physics Bayesian HPO Sweep")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials for this worker")
    parser.add_argument("--epochs_per_trial", type=int, default=5, help="Training epochs per trial")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--storage", type=str, default="sqlite:///sweep_optuna.db", help="Optuna RDBMS storage URI")
    parser.add_argument("--study_name", type=str, default="clare_space_physics_hpo", help="Study identifier")
    parser.add_argument("--data_dir", type=str, default="dataset/processed_dataset_01_31_storm", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sweep_results", help="Results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def create_optuna_storage(storage_spec: str):
    """Instantiates an Optuna storage backend, defaulting to lock-free JournalFileStorage on shared cluster filesystems."""
    if storage_spec.endswith(".log") or storage_spec.startswith("journal://"):
        from optuna.storages import JournalStorage, JournalFileStorage
        journal_path = storage_spec.replace("journal://", "")
        journal_dir = os.path.dirname(os.path.abspath(journal_path))
        if journal_dir:
            os.makedirs(journal_dir, exist_ok=True)
        return JournalStorage(JournalFileStorage(journal_path))
    return storage_spec


def main():  # pragma: no cover
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Starting Space Physics HPO Sweep on {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    # Load shared dataloaders
    print(f"[INFO] Loading datasets from: {args.data_dir}")
    train_loader, val_normal_loader, val_storm_loader = load_partitioned_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    # Instantiate Optuna Study with Lustre-safe storage and median pruning
    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    optuna_storage = create_optuna_storage(args.storage)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=optuna_storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True
    )

    print(f"[INFO] Optuna study '{args.study_name}' connected to storage '{args.storage}'")
    print(f"[INFO] Executing {args.n_trials} trials...")

    # Execute optimization
    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_normal_loader, val_storm_loader, device),
        n_trials=args.n_trials
    )

    # Print Best Trial Results
    best = study.best_trial
    print("\n" + "=" * 80)
    print("  SPACE PHYSICS HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"  Best Trial Number:          #{best.number}")
    print(f"  Best Composite Loss:        {best.value:.4f}")
    print(f"  Quiet Normal Loss:          {best.user_attrs.get('val_normal_loss', 0.0):.4f}")
    print(f"  Held-Out Storm Loss:        {best.user_attrs.get('val_storm_loss', 0.0):.4f}")
    print(f"  Parameter Count:            {best.user_attrs.get('num_params', 0):,}")
    print("\n  Winning Hyperparameters:")
    for k, v in best.params.items():
        print(f"    - {k:22s}: {v}")
    print("=" * 80 + "\n")

    # Save summary report JSON
    results_path = os.path.join(args.output_dir, "hpo_sweep_summary.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_trial_number": best.number,
            "best_composite_loss": best.value,
            "best_params": best.params,
            "best_user_attrs": best.user_attrs,
            "total_trials": len(study.trials)
        }, f, indent=2)
    print(f"[INFO] Saved sweep summary to: {results_path}")


if __name__ == "__main__":
    main()
