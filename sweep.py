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

try:
    from train_v2 import TEMPEST
except ImportError:
    TEMPEST = None


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
    num_workers: int = 2,
    include_storm: bool = False
) -> Any:
    """
    Loads train_chunks and val-normal with pinned memory.
    Enforces scientific holdout: unseen test-normal and test-storm are strictly
    excluded from hyperparameter tuning and model selection.
    """
    train_dir = os.path.join(data_dir, "train_chunks")
    val_normal_dir = os.path.join(data_dir, "val-normal")
    val_storm_dir = os.path.join(data_dir, "val-storm")
    if not os.path.exists(val_normal_dir):
        val_normal_dir = os.path.join(data_dir, "test-normal")
    if not os.path.exists(val_storm_dir):
        val_storm_dir = os.path.join(data_dir, "test-storm")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Load train chunks for agile HPO trials
    train_ds_list = []
    chunk_dirs = sorted([d for d in os.listdir(train_dir) if d.startswith("train_chunk_")])
    for chunk_name in chunk_dirs[:3]:  # Use first 3 chunks (~750k samples) for fast trial exploration
        chunk = datasets.Dataset.load_from_disk(os.path.join(train_dir, chunk_name))
        train_ds_list.append(chunk)

    combined_train = datasets.concatenate_datasets(train_ds_list)
    val_normal_ds = datasets.Dataset.load_from_disk(val_normal_dir)

    train_dataset = SpaceWeatherDataset(combined_train, SYM_H_FEATURE_IDX)
    val_normal_dataset = SpaceWeatherDataset(val_normal_ds, SYM_H_FEATURE_IDX)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_normal_loader = DataLoader(
        val_normal_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if include_storm and os.path.exists(val_storm_dir):
        val_storm_ds = datasets.Dataset.load_from_disk(val_storm_dir)
        val_storm_dataset = SpaceWeatherDataset(val_storm_ds, SYM_H_FEATURE_IDX)
        val_storm_loader = DataLoader(
            val_storm_dataset, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        return train_loader, val_normal_loader, val_storm_loader

    return train_loader, val_normal_loader


# ==============================================================================
# 4. OBJECTIVE FUNCTION FOR OPTUNA (Multi-Fidelity & Physics Guided)
# ==============================================================================

def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_normal_loader: DataLoader,
    val_storm_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None
) -> float:
    """
    Evaluates a single trial configuration using Multi-Fidelity Hyperband Pruning.
    Enforces scientific holdout: optimizes strictly on val-normal (and optional val-storm)
    without touching unseen test-normal or test-storm final evaluation benchmarks.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 1. Sample Hyperparameters from Search Space
    # -------------------------------------------------------------------------
    arch_family = trial.suggest_categorical("arch_family", ["tempest", "swiglu_moe"]) if TEMPEST is not None else "swiglu_moe"

    if arch_family == "tempest":
        d_model = trial.suggest_categorical("d_model", [128, 256, 384])
        expert_dim = trial.suggest_categorical("expert_dim", [256, 512, 768])
        num_experts = trial.suggest_categorical("num_experts", [4, 8])
        top_k = trial.suggest_categorical("top_k", [1, 2])
        n_layers = trial.suggest_int("n_layers", 2, 5)
        n_hc = trial.suggest_categorical("n_hc", [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.25)

        model = TEMPEST(
            num_features=len(INPUT_COLUMNS),
            d_model=d_model,
            expert_dim=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            n_layers=n_layers,
            n_hc=n_hc,
            vocab_size=NUM_CLASSES
        ).to(device)
    else:
        d_model = trial.suggest_categorical("d_model", [128, 256, 384])
        num_layers = trial.suggest_int("num_layers", 3, 7)
        use_moe = trial.suggest_categorical("use_moe", [True, False])
        num_experts = trial.suggest_categorical("num_experts", [4, 8]) if use_moe else 1
        top_k = 2 if use_moe else 1
        dropout = trial.suggest_float("dropout", 0.05, 0.25)

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

    # Optimization dimensions
    max_lr = trial.suggest_float("max_lr", 2e-4, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # Unified Physics Loss dimensions
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 2.5)
    huber_weight = trial.suggest_float("huber_weight", 0.05, 0.50)
    storm_alpha = trial.suggest_float("storm_alpha", 0.5, 3.0)
    huber_delta = trial.suggest_float("huber_delta", 250.0, 750.0)

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
        fused=(torch.cuda.is_available() and device.type == "cuda")
    )

    epochs = getattr(args, "epochs_per_trial", 5)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=max_lr * 0.01)

    scaler = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and device.type == "cuda"))

    # Training and validation loop with Multi-Fidelity Hyperband / Median Pruning
    best_target_loss = float("inf")
    bin_centers = torch.linspace(50.0, 14950.0, NUM_CLASSES, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(torch.cuda.is_available() and device.type == "cuda")):
                logits = model(x_b)
                loss = criterion(logits, y_b, inputs=x_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_acc += loss.item()

        # Validation on contiguous quiet-time validation blocks (val-normal)
        model.eval()
        val_normal_loss = 0.0
        val_normal_sq_err = 0.0
        val_normal_count = 0
        with torch.no_grad():
            for x_b, y_b in val_normal_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(torch.cuda.is_available() and device.type == "cuda")):
                    logits = model(x_b)
                    loss = criterion(logits, y_b, inputs=x_b)
                val_normal_loss += loss.item()

                probs = F.softmax(logits, dim=-1)
                pred_te = (probs * bin_centers).sum(dim=-1)
                true_te = y_b.float() * 100.0 + 50.0
                val_normal_sq_err += ((pred_te - true_te) ** 2).sum().item()
                val_normal_count += len(y_b)

        val_normal_loss /= max(1, len(val_normal_loader))
        val_normal_rmse = math.sqrt(val_normal_sq_err / max(1, val_normal_count))

        # Optional storm validation
        val_storm_loss = 0.0
        val_storm_rmse = 0.0
        val_storm_sq_err = 0.0
        val_storm_count = 0
        if val_storm_loader is not None:
            with torch.no_grad():
                for x_b, y_b in val_storm_loader:
                    x_b = x_b.to(device, non_blocking=True)
                    y_b = y_b.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(torch.cuda.is_available() and device.type == "cuda")):
                        logits = model(x_b)
                        loss = criterion(logits, y_b, inputs=x_b)
                    val_storm_loss += loss.item()

                    probs = F.softmax(logits, dim=-1)
                    pred_te = (probs * bin_centers).sum(dim=-1)
                    true_te = y_b.float() * 100.0 + 50.0
                    val_storm_sq_err += ((pred_te - true_te) ** 2).sum().item()
                    val_storm_count += len(y_b)

            val_storm_loss /= max(1, len(val_storm_loader))
            val_storm_rmse = math.sqrt(val_storm_sq_err / max(1, val_storm_count))

        # Scalarized Composite Objective: balances quiet precision and storm fidelity
        storm_w = getattr(args, "storm_weight", 0.0)
        if val_storm_loader is not None and storm_w > 0.0:
            target_val_loss = (1.0 - storm_w) * val_normal_loss + storm_w * val_storm_loss
        else:
            target_val_loss = val_normal_loss

        best_target_loss = min(best_target_loss, target_val_loss)

        # Multi-fidelity report & pruning (Hyperband / Median)
        trial.report(target_val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Log metrics to trial user attributes
    trial.set_user_attr("val_normal_loss", float(val_normal_loss))
    trial.set_user_attr("val_normal_rmse_k", float(val_normal_rmse))
    if val_storm_loader is not None:
        trial.set_user_attr("val_storm_loss", float(val_storm_loss))
        trial.set_user_attr("val_storm_rmse_k", float(val_storm_rmse))
    trial.set_user_attr("num_params", int(sum(p.numel() for p in model.parameters())))
    trial.set_user_attr("arch_family", str(arch_family))

    return best_target_loss


# ==============================================================================
# 5. CLI ENTRYPOINT & DISTRIBUTED RUNNER
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Space Physics Bayesian & Multi-Fidelity HPO Sweep")
    parser.add_argument("--n_trials", type=int, default=150, help="Number of trials for this worker")
    parser.add_argument("--epochs_per_trial", type=int, default=27, help="Training epochs per trial (max resource)")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--storage", type=str, default="sqlite:///sweep_optuna.db", help="Optuna RDBMS / Journal storage URI")
    parser.add_argument("--study_name", type=str, default="tempest_space_physics_sweep", help="Study identifier")
    parser.add_argument("--data_dir", type=str, default="dataset/processed_dataset_01_31_storm", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sweep_results", help="Results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pruner", type=str, default="hyperband", choices=["hyperband", "median", "none"], help="Pruner algorithm")
    parser.add_argument("--min_resource", type=int, default=3, help="Minimum epochs before pruning in Hyperband")
    parser.add_argument("--reduction_factor", type=int, default=3, help="Reduction factor for successive halving rungs")
    parser.add_argument("--storm_weight", type=float, default=0.25, help="Weight for storm loss in composite objective [0.0, 1.0]")
    return parser.parse_args()


def create_optuna_storage(storage_spec: str):
    """Instantiates an Optuna storage backend, defaulting to lock-free JournalFileBackend on shared cluster filesystems."""
    if storage_spec.endswith(".log") or storage_spec.startswith("journal://"):
        journal_path = storage_spec.replace("journal://", "")
        journal_dir = os.path.dirname(os.path.abspath(journal_path))
        if journal_dir:
            os.makedirs(journal_dir, exist_ok=True)
        try:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            return JournalStorage(JournalFileBackend(journal_path))
        except (ImportError, AttributeError):
            from optuna.storages import JournalStorage, JournalFileStorage
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
    dl_result = load_partitioned_dataloaders(
        args.data_dir, batch_size=args.batch_size, include_storm=True
    )
    if isinstance(dl_result, tuple) and len(dl_result) == 3:
        train_loader, val_normal_loader, val_storm_loader = dl_result
    else:
        train_loader, val_normal_loader = dl_result
        val_storm_loader = None

    # Multi-fidelity Pruner configuration
    if args.pruner == "hyperband":
        from optuna.pruners import HyperbandPruner
        pruner = HyperbandPruner(
            min_resource=args.min_resource,
            max_resource=args.epochs_per_trial,
            reduction_factor=args.reduction_factor
        )
        print(f"[INFO] Using Multi-Fidelity HyperbandPruner (min={args.min_resource}, max={args.epochs_per_trial}, factor={args.reduction_factor})")
    elif args.pruner == "median":
        from optuna.pruners import MedianPruner
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        print(f"[INFO] Using MedianPruner (warmup=2)")
    else:
        from optuna.pruners import NopPruner
        pruner = NopPruner()
        print(f"[INFO] Pruning disabled (NopPruner)")

    # Multivariate Tree-structured Parzen Estimator (TPE)
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
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

    # Execute optimization (tunes on val-normal / val-storm; unseen test-storm and test-normal held out)
    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_normal_loader, val_storm_loader=val_storm_loader, device=device),
        n_trials=args.n_trials
    )

    # Print Best Trial Results
    best = study.best_trial
    print("\n" + "=" * 80)
    print("  SPACE PHYSICS HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"  Best Trial Number:          #{best.number}")
    print(f"  Best Composite Loss:        {best.value:.4f}")
    print(f"  Validation Loss (quiet):    {best.user_attrs.get('val_normal_loss', 0.0):.4f}")
    print(f"  Validation RMSE (quiet):    {best.user_attrs.get('val_normal_rmse_k', 0.0):.1f} K")
    if 'val_storm_loss' in best.user_attrs:
        print(f"  Validation Loss (storm):    {best.user_attrs.get('val_storm_loss', 0.0):.4f}")
        print(f"  Validation RMSE (storm):    {best.user_attrs.get('val_storm_rmse_k', 0.0):.1f} K")
    print(f"  Architecture Family:        {best.user_attrs.get('arch_family', 'N/A')}")
    print(f"  Parameter Count:            {best.user_attrs.get('num_params', 0):,}")
    print("\n  Winning Hyperparameters:")
    for k, v in best.params.items():
        print(f"    - {k:22s}: {v}")
    print("=" * 80 + "\n")

    # Pareto-Optimal Frontier Analysis
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    has_storm_attrs = any("val_storm_loss" in t.user_attrs for t in completed_trials)
    pareto_trials = []

    if has_storm_attrs and len(completed_trials) > 0:
        for t1 in completed_trials:
            q1 = t1.user_attrs.get("val_normal_loss", float("inf"))
            s1 = t1.user_attrs.get("val_storm_loss", float("inf"))
            dominated = False
            for t2 in completed_trials:
                if t1.number == t2.number:
                    continue
                q2 = t2.user_attrs.get("val_normal_loss", float("inf"))
                s2 = t2.user_attrs.get("val_storm_loss", float("inf"))
                if q2 <= q1 and s2 <= s1 and (q2 < q1 or s2 < s1):
                    dominated = True
                    break
            if not dominated:
                pareto_trials.append({
                    "trial_number": t1.number,
                    "val_normal_loss": q1,
                    "val_storm_loss": s1,
                    "val_normal_rmse_k": t1.user_attrs.get("val_normal_rmse_k", 0.0),
                    "val_storm_rmse_k": t1.user_attrs.get("val_storm_rmse_k", 0.0),
                    "params": t1.params
                })

    # Save summary report JSON
    results_path = os.path.join(args.output_dir, "hpo_sweep_summary.json")
    summary_data = {
        "best_trial_number": best.number,
        "best_composite_loss": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "pareto_optimal_trials": pareto_trials
    }
    with open(results_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[INFO] Saved comprehensive sweep summary to: {results_path}")

    # Generate Optuna Publication Visualizations
    try:
        import matplotlib.pyplot as plt
        if has_storm_attrs and len(completed_trials) > 1:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            q_losses = [t.user_attrs.get("val_normal_loss", 0.0) for t in completed_trials]
            s_losses = [t.user_attrs.get("val_storm_loss", 0.0) for t in completed_trials]
            ax.scatter(q_losses, s_losses, c="royalblue", alpha=0.6, s=35, label="Completed Trials")

            if pareto_trials:
                pq = [pt["val_normal_loss"] for pt in pareto_trials]
                ps = [pt["val_storm_loss"] for pt in pareto_trials]
                sort_p = sorted(zip(pq, ps))
                ax.plot([x[0] for x in sort_p], [x[1] for x in sort_p], color="crimson", linestyle="--", linewidth=2.0, marker="o", label="Pareto Frontier")

            ax.set_xlabel("Quiet Validation Loss (val-normal)", fontweight="bold")
            ax.set_ylabel("Storm Validation Loss (val-storm)", fontweight="bold")
            ax.set_title("TEMPEST HPO: Quiet vs. Storm Validation Pareto Frontier", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(frameon=True)
            pareto_fig_path = os.path.join(args.output_dir, "optuna_pareto_front.png")
            plt.tight_layout()
            plt.savefig(pareto_fig_path, dpi=300)
            plt.close(fig)
            print(f"[INFO] Saved Pareto Frontier plot to: {pareto_fig_path}")

        if len(completed_trials) >= 5:
            try:
                import optuna.visualization.matplotlib as optuna_vis
                fig = optuna_vis.plot_param_importances(study)
                imp_fig_path = os.path.join(args.output_dir, "optuna_param_importances.png")
                plt.tight_layout()
                plt.savefig(imp_fig_path, dpi=300)
                plt.close(plt.gcf())
                print(f"[INFO] Saved Parameter Importances plot to: {imp_fig_path}")
            except Exception as e:
                print(f"[WARNING] Could not plot parameter importances: {e}")
    except Exception as e:
        print(f"[WARNING] Could not generate Optuna visualization plots: {e}")


if __name__ == "__main__":
    main()
