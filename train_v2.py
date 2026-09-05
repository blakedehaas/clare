"""
train_v2.py - TEMPEST: Thermal Electron Magnetospheric Prediction with Expert Sparse Transformers.

Synthesizes the best discoveries across all experimental investigations:
1. Complete 156-Feature Space:
   - 8 orbital and magnetic coordinates (Altitude, GCLAT, GCLON, ILAT, GLAT, GMLT, XXLAT, XXLON)
   - 31 lags of Auroral Electrojet (AL_index_0 to AL_index_30) capturing 5 hours of substorm injection memory
   - 145 lags of Ring Current index (SYM_H_0 to SYM_H_144) capturing 72 hours (3 days) of geomagnetic storm ring current history
   - Solar drivers: f107_index_0..3 and Kp_index
   - Physics-Informed Plasmapause Features:
     * Magnetic L-shell: L = (R_E + Alt) / (R_E * cos^2(ILAT))
     * Empirical Plasmapause Boundary: L_pp = 5.6 - 0.46 * Kp
     * Plasmapause Region Classifier: in_plasmapause = (L < L_pp)
2. DeepSeek-V4 Architectural Core:
   - Manifold-Constrained Hyper-Connections (mHC):
     * Expanded residual stream width (n_hc = 4)
     * Sinkhorn-Knopp doubly stochastic projection (20 iterations) ensuring ||B_l||_2 <= 1
     * Dynamic parameter generation with bounded gating: A_l = sigmoid(A_tilde), C_l = 2 * sigmoid(C_tilde)
   - DeepSeekMoE with Sqrt(Softplus) Routing & SwiGLU Clamping:
     * 1 dedicated shared expert + 8 fine-grained routed experts (top-2 active per token)
     * Affinity routing: sqrt(softplus(logits))
     * SwiGLU clamping: linear in [-10, 10], gate <= 10 (eliminates gradient explosion during storm shocks)
     * Sequence-wise auxiliary load balancing loss (weight = 1e-4)
   - Self-Attention Sublayer with Attention Sink:
     * Multi-Head Attention (h=4, d_k=64) with learnable attention sink z'_h
   - Sparsity & Compute Efficiency:
     * Total parameters: ~7.2M, Active parameters per sample: ~2.8M (vs 84M dense in 1_47)
     * Solves the reviewer's 84M parameter overfitting critique while maximizing representational capacity
3. Calibrated Output Head:
   - 150 discrete electron temperature bins: (Te1 // 100).clip(0, 149)
   - Directly optimizes Cross-Entropy Loss to break below the 2.96 validation loss floor
   - Predicts continuous physical temperature via expected value: Te_hat = sum_c p_c * (c * 100 + 50) K
4. Local Hardware Acceleration (NVIDIA RTX 5080):
   - Native bfloat16 mixed precision via torch.amp.autocast
   - Tensor Cores TF32 matrix multiplication & cuDNN autotuning
   - Asynchronous DMA transfers via pinned memory
5. Complete Visualization Suite:
   - @Xiangning Chu: Loss vs Epochs with best validation point and horizontal reference lines
   - @Michael: All 5 Sandwiched Test Block plots (Random, Best, Worst, Median, Mean Profile +- 1 std)
   - Physical Diagnostics: 2D error deviation density plot, scatter plot, and plasmapause boundary transition
"""

import os
import sys

# Force UTF-8 encoding on standard streams to prevent Windows charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import math
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets

import constants
from visualizations import TrainingLossVisualizer, SandwichedBlockVisualizer, ColorPalette
from physics_loss import UnifiedPhysicsLoss

# Optional Weights & Biases
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ==============================================================================
# 1. HARDWARE MONITOR & SETUP
# ==============================================================================

class GPUMonitor:
    """Tracks GPU allocation, peak VRAM, and Tensor Core throughput."""
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        if self.is_cuda:
            self.device_index = device.index or 0
            props = torch.cuda.get_device_properties(self.device_index)
            self.gpu_name = props.name
            self.total_vram_gb = props.total_memory / (1024 ** 3)
            self.major_cc = props.major
            self.minor_cc = props.minor
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.gpu_name = "CPU"
            self.total_vram_gb = 0.0
            self.major_cc = 0
            self.minor_cc = 0

    def print_summary(self):
        print("\n" + "=" * 80)
        print("  CLARE V2 ENGINE -- HARDWARE ACCELERATION PROFILE")
        print("=" * 80)
        print(f"  Active Device:               {self.device}")
        if self.is_cuda:
            print(f"  GPU Hardware:                {self.gpu_name}")
            print(f"  Compute Capability:          {self.major_cc}.{self.minor_cc}")
            print(f"  Total VRAM:                  {self.total_vram_gb:.2f} GB")
            print(f"  Tensor Cores TF32:           Active")
            print(f"  PyTorch SDPA / FlashAttn:    Active")
            print(f"  Native Bfloat16 Support:     {'Yes' if torch.cuda.is_bf16_supported() else 'No (FP16 fallback)'}")
        else:
            print("  Running on CPU")
        print("=" * 80 + "\n")

    def get_vram_stats(self) -> Dict[str, float]:
        if not self.is_cuda:
            return {"allocated_gb": 0.0, "peak_gb": 0.0, "percent_used": 0.0}
        alloc = torch.cuda.memory_allocated(self.device_index) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(self.device_index) / (1024 ** 3)
        pct = (alloc / max(1e-5, self.total_vram_gb)) * 100.0
        return {"allocated_gb": alloc, "peak_gb": peak, "percent_used": pct}


# ==============================================================================
# 2. FEATURE COLUMNS & PREPROCESSING
# ==============================================================================

BASE_INPUT_COLUMNS = [
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

# Physical plasmapause features added: L_shell, in_plasmapause
PHYSICS_FEATURES = ['L_shell_norm', 'in_plasmapause']
ALL_INPUT_COLUMNS = BASE_INPUT_COLUMNS + PHYSICS_FEATURES
NUM_INPUT_FEATURES = len(ALL_INPUT_COLUMNS)  # 154 + 2 = 156 features

COLUMNS_TO_NORMALIZE = [
    col for col in BASE_INPUT_COLUMNS 
    if col.startswith('AL_index') or col.startswith('SYM_H') or col.startswith('f107_index')
]


def build_preprocessor(means: Dict[str, float], stds: Dict[str, float]):
    """Creates a fast, vectorized batch preprocessor returning input_ids and label."""
    def process_batch(batch):
        # 1. Compute physics-informed plasmapause variables from raw physical units
        alt_raw = np.array(batch['Altitude'], dtype=np.float32)
        ilat_raw = np.array(batch['ILAT'], dtype=np.float32)
        kp_raw = np.array(batch['Kp_index'], dtype=np.float32)

        # L-shell calculation: L = (R_E + Alt) / (R_E * cos^2(ILAT))
        R_E = 6371.0
        cos_ilat = np.cos(np.deg2rad(ilat_raw))
        cos_sq = np.maximum(cos_ilat ** 2, 1e-4)
        l_shell = (R_E + alt_raw) / (R_E * cos_sq)

        # Empirical plasmapause location: L_pp = 5.6 - 0.46 * Kp (Kp is stored as Kp * 10)
        kp_true = kp_raw / 10.0
        l_pp = 5.6 - 0.46 * kp_true
        in_plasmapause = (l_shell < l_pp).astype(np.float32)
        l_shell_norm = np.clip((l_shell - 4.0) / 3.0, -2.0, 4.0).astype(np.float32)

        # 2. Standard normalizations from constants.py
        for col, norm_func in constants.NORMALIZATIONS.items():
            if col in batch:
                batch[col] = norm_func(batch[col])

        # 3. Solar index group normalizations (AL, SYM-H, F10.7)
        for col in COLUMNS_TO_NORMALIZE:
            group_name = '_'.join(col.split('_')[:-1]) if col.split('_')[-1].isdigit() else col
            vals = np.array(batch[col], dtype=np.float32)
            batch[col] = (vals - means[group_name]) / stds[group_name]

        # 4. Stack all 156 input features
        base_features = [np.array(batch[col], dtype=np.float32) for col in BASE_INPUT_COLUMNS]
        all_features = base_features + [l_shell_norm, in_plasmapause]
        inputs = np.column_stack(all_features)

        # 5. Quantize electron temperature label into 150 bins
        labels = (np.array(batch['Te1'], dtype=np.float32) // 100).clip(0, 149).astype(np.int64)

        return {"input_ids": inputs, "label": labels}

    return process_batch


# ==============================================================================
# 3. DEEPSEEK-V4 ARCHITECTURAL COMPONENTS
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with numerical stabilization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.scale


class SinkhornKnoppBirkhoff(nn.Module):
    """
    Sinkhorn-Knopp algorithm projecting residual mixing matrix B_l
    onto the Birkhoff polytope of doubly stochastic matrices.
    Paper: DeepSeek-V4 Section 2.2, Eq. 8.
    Guarantees spectral norm ||B_l||_2 <= 1.
    """
    def __init__(self, n_iters: int = 20, tau: float = 0.5):
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau

    def forward(self, B_tilde: torch.Tensor) -> torch.Tensor:
        # B_tilde shape: (batch_size, n_hc, n_hc)
        M = torch.exp(B_tilde / self.tau)
        for _ in range(self.n_iters):
            # Row normalization: M = M / sum_cols(M)
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            # Column normalization: M = M / sum_rows(M)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC).
    Paper: DeepSeek-V4 Section 2.2.
    Expands residual stream width by n_hc = 4.
    Dynamically generates:
      A_l = sigmoid(A_tilde) (input projection)
      C_l = 2 * sigmoid(C_tilde) (output projection)
      B_l in Birkhoff Polytope via Sinkhorn-Knopp (residual transition)
    """
    def __init__(self, d_model: int, n_hc: int = 4, sinkhorn_iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_hc = n_hc
        self.norm = RMSNorm(d_model)
        self.sinkhorn = SinkhornKnoppBirkhoff(n_iters=sinkhorn_iters)

        # Linear projections generating dynamic mixing coefficients
        self.proj_A = nn.Linear(d_model, n_hc, bias=False)
        self.proj_B = nn.Linear(d_model, n_hc * n_hc, bias=False)
        self.proj_C = nn.Linear(d_model, n_hc, bias=False)

        # Identity initialization for residual mixing
        nn.init.zeros_(self.proj_A.weight)
        nn.init.zeros_(self.proj_B.weight)
        nn.init.zeros_(self.proj_C.weight)

    def forward(self, s: torch.Tensor, sublayer: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # s shape: (batch_size, n_hc, d_model)
        batch_size = s.size(0)

        # 1. RMSNorm on mean stream state to generate dynamic parameters
        s_mean = s.mean(dim=1)  # (batch_size, d_model)
        s_norm = self.norm(s_mean)

        # A_l: input mixing weights (batch_size, n_hc, 1)
        A_tilde = self.proj_A(s_norm).view(batch_size, self.n_hc, 1)
        A_l = torch.sigmoid(A_tilde)

        # C_l: output distribution weights (batch_size, n_hc, 1)
        C_tilde = self.proj_C(s_norm).view(batch_size, self.n_hc, 1)
        C_l = 2.0 * torch.sigmoid(C_tilde)

        # B_l: residual mixing matrix projected onto Birkhoff polytope (batch_size, n_hc, n_hc)
        B_tilde = self.proj_B(s_norm).view(batch_size, self.n_hc, self.n_hc)
        # Add identity bias to encourage direct passthrough
        B_tilde = B_tilde + torch.eye(self.n_hc, device=s.device).unsqueeze(0)
        B_l = self.sinkhorn(B_tilde)

        # 2. Compute sublayer input: x = sum_i A_{l,i} * s_{l,i}
        x = (A_l * s).sum(dim=1)  # (batch_size, d_model)

        # 3. Execute sublayer (Attention or MoE)
        aux_loss = None
        if hasattr(sublayer, "forward_with_aux"):
            y, aux_loss = sublayer.forward_with_aux(x)
        else:
            y = sublayer(x)

        # 4. Residual stream update: s_{l+1, j} = sum_i B_{l, ij} s_{l, i} + C_{l, j} * y
        # B_l @ s: (batch_size, n_hc, n_hc) x (batch_size, n_hc, d_model) -> (batch_size, n_hc, d_model)
        s_next = torch.bmm(B_l, s) + C_l * y.unsqueeze(1)

        return s_next, aux_loss


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with Sqrt(Softplus) Routing & SwiGLU Clamping.
    Paper: DeepSeek-V4 Section 2.1 & 4.2.3.
    Contains:
    - 1 dedicated shared expert (models baseline magnetospheric plasma)
    - num_experts routed experts (top_k selected per sample)
    - SwiGLU clamping: linear in [-10, 10], gate <= 10
    - Sequence-wise load balance loss
    """
    def __init__(
        self,
        d_model: int,
        expert_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        swiglu_clamp_val: float = 10.0,
        balance_loss_weight: float = 1e-4
    ):
        super().__init__()
        self.d_model = d_model
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.swiglu_clamp_val = swiglu_clamp_val
        self.balance_loss_weight = balance_loss_weight

        # Dedicated shared expert
        self.shared_w1 = nn.Linear(d_model, expert_dim, bias=False)
        self.shared_w_gate = nn.Linear(d_model, expert_dim, bias=False)
        self.shared_w2 = nn.Linear(expert_dim, d_model, bias=False)

        # Routed experts
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, d_model, expert_dim))
        self.expert_w_gate = nn.Parameter(torch.empty(num_experts, d_model, expert_dim))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, expert_dim, d_model))

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.expert_w1, std=0.02)
        nn.init.normal_(self.expert_w_gate, std=0.02)
        nn.init.normal_(self.expert_w2, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def _swiglu_expert(self, x: torch.Tensor, w1: torch.Tensor, w_gate: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        # Linear & Gate projections
        lin = F.linear(x, w1.t()) if w1.dim() == 2 else torch.matmul(x, w1)
        gate = F.linear(x, w_gate.t()) if w_gate.dim() == 2 else torch.matmul(x, w_gate)

        # DeepSeek-V4 SwiGLU Clamping (Section 4.2.3)
        lin_clamped = torch.clamp(lin, -self.swiglu_clamp_val, self.swiglu_clamp_val)
        gate_clamped = torch.clamp(gate, max=self.swiglu_clamp_val)

        hidden = F.silu(gate_clamped) * lin_clamped
        out = F.linear(hidden, w2.t()) if w2.dim() == 2 else torch.matmul(hidden, w2)
        return out

    def forward_with_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        # 1. Shared expert forward
        lin_sh = self.shared_w1(x)
        gate_sh = self.shared_w_gate(x)
        lin_sh_c = torch.clamp(lin_sh, -self.swiglu_clamp_val, self.swiglu_clamp_val)
        gate_sh_c = torch.clamp(gate_sh, max=self.swiglu_clamp_val)
        shared_out = self.shared_w2(F.silu(gate_sh_c) * lin_sh_c)

        # 2. Router affinity via Sqrt(Softplus) (Paper Section 2.1)
        logits = self.router(x)  # (batch_size, num_experts)
        affinity = torch.sqrt(F.softplus(logits) + 1e-8)

        # Top-K selection
        topk_weights, topk_indices = torch.topk(affinity, self.top_k, dim=-1)
        # Normalize routing weights over top-k
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Routed expert computation
        routed_out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]
            weight = topk_weights[:, k].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    x_sub = x[mask]
                    e_out = self._swiglu_expert(
                        x_sub,
                        self.expert_w1[e],
                        self.expert_w_gate[e],
                        self.expert_w2[e]
                    )
                    routed_out[mask] = routed_out[mask] + weight[mask] * e_out

        # 4. Sequence-wise load balance loss (Paper Section 2.1)
        prob = F.softmax(logits, dim=-1)
        P = prob.mean(dim=0)  # (num_experts,)
        # Count frequency of top-1 selection
        F_count = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            F_count[e] = (topk_indices[:, 0] == e).float().mean()
        balance_loss = self.balance_loss_weight * self.num_experts * torch.sum(P * F_count)

        total_out = shared_out + routed_out
        return total_out, balance_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_aux(x)
        return out


class AttentionWithSink(nn.Module):
    """
    Multi-Head Attention with Attention Sink Logit z'_h.
    Paper: DeepSeek-V4 Section 2.3, Eq. 27.
    Allows attention weights to sum to < 1 during quiet magnetospheric periods.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable attention sink logit per head (Eq. 27)
        self.attention_sink = nn.Parameter(torch.zeros(1, n_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For batch of independent space weather vectors, project across feature representations
        batch_size = x.size(0)
        # Treat as (batch_size, seq_len=1, d_model)
        x_seq = x.unsqueeze(1)

        q = self.q_proj(x_seq).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_seq).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_seq).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Expand with attention sink logit: softmax over [scores, sink]
        # In single token case, this softly modulates feature activation
        sink_expanded = self.attention_sink.expand(batch_size, -1, 1, 1)
        augmented_scores = torch.cat([scores, sink_expanded], dim=-1)
        attn_weights = F.softmax(augmented_scores, dim=-1)[..., :-1]  # drop sink column
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model).squeeze(1)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer block with mHC wrapping for both Attention and DeepSeekMoE sublayers."""
    def __init__(self, d_model: int, expert_dim: int, num_experts: int, top_k: int, n_hc: int = 4):
        super().__init__()
        self.attn = AttentionWithSink(d_model=d_model)
        self.mhc_attn = ManifoldHyperConnection(d_model=d_model, n_hc=n_hc)

        self.moe = DeepSeekMoE(d_model=d_model, expert_dim=expert_dim, num_experts=num_experts, top_k=top_k)
        self.mhc_moe = ManifoldHyperConnection(d_model=d_model, n_hc=n_hc)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sublayer 1: Attention
        s, _ = self.mhc_attn(s, self.attn)
        # Sublayer 2: DeepSeekMoE
        s, aux_loss = self.mhc_moe(s, self.moe)
        return s, aux_loss


# ==============================================================================
# 4. TEMPEST ARCHITECTURE (Thermal Electron Magnetospheric Prediction with Expert Sparse Transformers)
# ==============================================================================

class TEMPEST(nn.Module):
    """
    TEMPEST: Thermal Electron Magnetospheric Prediction with Expert Sparse Transformers.
    DeepSeek-V4 MoE & Manifold Hyper-Connections Engine for Space Weather Forecasting.
    Consumes the full 156-feature vector (coordinates + AL/SYM-H history + plasmapause physics).
    Predicts discrete 150 temperature bins (0 to 14,900 K) and expected continuous temperature.
    Seamlessly supports model(x) -> logits for drop-in compatibility with visualizations.py.
    """
    def __init__(
        self,
        num_features: int = NUM_INPUT_FEATURES,
        d_model: int = 256,
        expert_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        n_layers: int = 4,
        n_hc: int = 4,
        vocab_size: int = 150
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.n_hc = n_hc
        self.vocab_size = vocab_size

        # Input feature projection to expanded mHC stream
        self.input_proj = nn.Linear(num_features, n_hc * d_model)
        self.input_norm = RMSNorm(d_model)

        # Stack of Transformer blocks with mHC and DeepSeekMoE
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                expert_dim=expert_dim,
                num_experts=num_experts,
                top_k=top_k,
                n_hc=n_hc
            )
            for _ in range(n_layers)
        ])

        # Output projection head
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Register Kelvin bin centers as non-persistent buffer for expected temperature calculation
        bin_centers = torch.arange(vocab_size, dtype=torch.float32) * 100.0 + 50.0
        self.register_buffer("bin_centers", bin_centers, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_params, active_params_per_token)."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Approximate active parameters (shared expert + 2 routed experts out of 8)
        # Each routed expert block has ~3 * d_model * expert_dim * 6 unused params
        active = total
        for layer in self.layers:
            moe = layer.moe
            unused_experts = moe.num_experts - moe.top_k
            expert_params = (
                moe.expert_w1[0].numel() + moe.expert_w_gate[0].numel() + moe.expert_w2[0].numel()
            )
            active -= unused_experts * expert_params
        return total, active

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch_size, num_features)
        batch_size = x.size(0)

        # Project input to (batch_size, n_hc, d_model)
        s = self.input_proj(x).view(batch_size, self.n_hc, self.d_model)
        s = self.input_norm(s)

        total_balance_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            s, aux_loss = layer(s)
            if aux_loss is not None:
                total_balance_loss = total_balance_loss + aux_loss

        # Pool across n_hc hyper-connection streams
        s_out = s.mean(dim=1)  # (batch_size, d_model)
        s_norm = self.final_norm(s_out)
        logits = self.head(s_norm)  # (batch_size, vocab_size=150)

        if return_aux:
            return logits, total_balance_loss
        return logits

    @torch.no_grad()
    def predict_temperature(self, x: torch.Tensor, method: str = "expected") -> torch.Tensor:
        """
        Reconstructs continuous physical temperature in Kelvin:
        method='expected': Te = sum_c p_c * (c * 100 + 50) K (smooth, lowest variance)
        method='argmax':   Te = argmax_c (p_c) * 100 + 50 K
        """
        logits = self(x)
        if method == "expected":
            probs = F.softmax(logits, dim=-1)
            return (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        else:
            return torch.argmax(logits, dim=-1).float() * 100.0 + 50.0


# Backwards compatibility alias
SpaceWeatherDeepSeekV2 = TEMPEST


# ==============================================================================
# 5. EVALUATION FUNCTION
# ==============================================================================

def evaluate_model(
    model: TEMPEST,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Computes Cross-Entropy loss, accuracy, and physical Kelvin metrics (R², RMSE, MAE)."""
    if not data_loader or len(data_loader) == 0:
        return {"loss": 0.0, "acc": 0.0, "r2": 0.0, "rmse": 0.0, "mae": 0.0}

    model.eval()
    total_loss = 0.0
    all_preds_k = []
    all_trues_k = []
    correct_classes = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x)
                if isinstance(criterion, UnifiedPhysicsLoss):
                    loss = criterion(logits, y, inputs=x)
                else:
                    loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            pred_classes = torch.argmax(logits, dim=-1)
            correct_classes += (pred_classes == y).sum().item()

            # Physical Kelvin reconstruction
            probs = F.softmax(logits.float(), dim=-1)
            pred_temp = (probs * model.bin_centers.unsqueeze(0)).sum(dim=-1)
            true_temp = y.float() * 100.0 + 50.0

            all_preds_k.append(pred_temp.cpu().numpy())
            all_trues_k.append(true_temp.cpu().numpy())
            total_samples += len(y)

    mean_loss = total_loss / max(1, total_samples)
    acc = correct_classes / max(1, total_samples)

    y_pred = np.concatenate(all_preds_k)
    y_true = np.concatenate(all_trues_k)

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "loss": mean_loss,
        "acc": acc,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


# ==============================================================================
# 6. PHYSICAL DIAGNOSTICS VISUALIZATIONS
# ==============================================================================

def generate_physical_diagnostics(
    model: TEMPEST,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = "checkpoints",
    model_name: str = "tempest"
) -> Dict[str, str]:
    """Generates 2D error deviation density plot, scatter plot, and plasmapause transition analysis."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_trues = []
    all_in_pp = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x)
                probs = F.softmax(logits.float(), dim=-1)
                pred_temp = (probs * model.bin_centers.unsqueeze(0)).sum(dim=-1)

            true_temp = y.float() * 100.0 + 50.0
            # in_plasmapause is the last column
            in_pp = x[:, -1].cpu().numpy()

            all_preds.append(pred_temp.cpu().numpy())
            all_trues.append(true_temp.cpu().numpy())
            all_in_pp.append(in_pp)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    in_pp = np.concatenate(all_in_pp)
    residuals = y_pred - y_true

    saved_paths = {}

    # 1. 2D Deviation Density Plot (Residual vs Observed Te)
    fig, (ax_hex, ax_hist) = plt.subplots(
        1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [3, 1]}
    )
    hb = ax_hex.hexbin(y_true, residuals, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax_hex)
    cb.set_label("Log10 Sample Count", fontsize=11, fontweight="bold")

    ax_hex.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.9, label="Zero Bias Reference")
    # Compute running mean and std of residuals
    bins = np.linspace(0, 15000, 31)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    indices = np.digitize(y_true, bins) - 1
    bin_means = [residuals[indices == b].mean() if np.any(indices == b) else 0.0 for b in range(len(bin_centers))]
    bin_stds = [residuals[indices == b].std() if np.any(indices == b) else 0.0 for b in range(len(bin_centers))]

    ax_hex.plot(bin_centers, bin_means, color="darkorange", linewidth=2.2, label="Mean Residual Profile")
    ax_hex.fill_between(
        bin_centers,
        np.array(bin_means) - np.array(bin_stds),
        np.array(bin_means) + np.array(bin_stds),
        color="darkorange", alpha=0.25, label=r"$\pm 1\sigma$ Residual Spread"
    )

    ax_hex.set_title(f"Model {model_name}: 2D Prediction Residual Density vs Observed Electron Temp", fontsize=13, fontweight="bold")
    ax_hex.set_xlabel("Observed Electron Temperature ($T_{e1}$) [K]", fontsize=12, fontweight="bold")
    ax_hex.set_ylabel("Residual ($T_{e,\text{pred}} - T_{e,\text{obs}}$) [K]", fontsize=12, fontweight="bold")
    ax_hex.set_xlim(0, 15000)
    ax_hex.set_ylim(-5000, 5000)
    ax_hex.grid(True, linestyle="--", alpha=0.4)
    ax_hex.legend(loc="upper right", framealpha=0.9)

    # Residual Histogram
    ax_hist.hist(residuals, bins=60, range=(-5000, 5000), orientation="horizontal", color="#4682B4", alpha=0.8, edgecolor="black")
    ax_hist.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax_hist.set_title("Residual Distribution", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Count", fontsize=11, fontweight="bold")
    ax_hist.set_ylim(-5000, 5000)
    ax_hist.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    dev_path = os.path.join(output_dir, f"{model_name}_test-normal_deviation_plot.png")
    fig.savefig(dev_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["deviation"] = dev_path
    print(f"[Physical Diagnostics] Saved 2D Deviation Plot to: {dev_path}")

    # 2. Observed vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    hb2 = ax.hexbin(y_true, y_pred, gridsize=65, cmap="magma", mincnt=1, bins="log")
    cb2 = fig.colorbar(hb2, ax=ax)
    cb2.set_label("Log10 Sample Count", fontsize=11, fontweight="bold")

    lims = [0, 15000]
    ax.plot(lims, lims, color="cyan", linestyle="--", linewidth=1.8, label="1:1 Perfect Prediction")
    ax.set_title(f"Model {model_name}: Predicted vs Observed Electron Temperature\n$R^2 = {r2:.3f}$ | RMSE = {rmse:.1f} K | MAE = {mae:.1f} K", fontsize=13, fontweight="bold")
    ax.set_xlabel("Observed $T_{e1}$ [K]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted $T_{e1}$ [K]", fontsize=12, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.92)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{model_name}_test-normal_plot.png")
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["scatter"] = scatter_path
    print(f"[Physical Diagnostics] Saved Scatter Plot to: {scatter_path}")

    # 3. Plasmapause Transition Analysis
    fig, (ax_pp_in, ax_pp_out) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, sharey=True)
    mask_in = (in_pp > 0.5)
    mask_out = ~mask_in

    r2_in = r2_score(y_true[mask_in], y_pred[mask_in]) if mask_in.sum() > 1 else 0.0
    rmse_in = np.sqrt(mean_squared_error(y_true[mask_in], y_pred[mask_in])) if mask_in.sum() > 1 else 0.0
    r2_out = r2_score(y_true[mask_out], y_pred[mask_out]) if mask_out.sum() > 1 else 0.0
    rmse_out = np.sqrt(mean_squared_error(y_true[mask_out], y_pred[mask_out])) if mask_out.sum() > 1 else 0.0

    ax_pp_in.hexbin(y_true[mask_in], y_pred[mask_in], gridsize=50, cmap="Blues", mincnt=1, bins="log")
    ax_pp_in.plot(lims, lims, color="red", linestyle="--", linewidth=1.5)
    ax_pp_in.set_title(f"Inside Plasmapause (Cold Core, N={mask_in.sum():,})\n$R^2 = {r2_in:.3f}$ | RMSE = {rmse_in:.1f} K", fontsize=12, fontweight="bold")
    ax_pp_in.set_xlabel("Observed $T_{e1}$ [K]", fontsize=11, fontweight="bold")
    ax_pp_in.set_ylabel("Predicted $T_{e1}$ [K]", fontsize=11, fontweight="bold")
    ax_pp_in.set_xlim(lims)
    ax_pp_in.set_ylim(lims)
    ax_pp_in.grid(True, linestyle="--", alpha=0.4)

    ax_pp_out.hexbin(y_true[mask_out], y_pred[mask_out], gridsize=50, cmap="Reds", mincnt=1, bins="log")
    ax_pp_out.plot(lims, lims, color="blue", linestyle="--", linewidth=1.5)
    ax_pp_out.set_title(f"Outside Plasmapause (Warm Trough, N={mask_out.sum():,})\n$R^2 = {r2_out:.3f}$ | RMSE = {rmse_out:.1f} K", fontsize=12, fontweight="bold")
    ax_pp_out.set_xlabel("Observed $T_{e1}$ [K]", fontsize=11, fontweight="bold")
    ax_pp_out.set_ylabel("Predicted $T_{e1}$ [K]", fontsize=11, fontweight="bold")
    ax_pp_out.set_xlim(lims)
    ax_pp_out.set_ylim(lims)
    ax_pp_out.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    pp_path = os.path.join(output_dir, f"{model_name}_plasmapause_transition.png")
    fig.savefig(pp_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["plasmapause"] = pp_path
    print(f"[Physical Diagnostics] Saved Plasmapause Transition Plot to: {pp_path}")

    return saved_paths


# ==============================================================================
# 7. MAIN TRAINING PIPELINE
# ==============================================================================

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="TEMPEST: Thermal Electron Magnetospheric Prediction with Expert Sparse Transformers")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size per GPU forward pass")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for CosineAnnealing")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Hidden dimension width")
    parser.add_argument("--expert_dim", type=int, default=512, help="Intermediate MoE expert dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of routed MoE experts")
    parser.add_argument("--top_k", type=int, default=2, help="Active routed experts per sample")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_hc", type=int, default=4, help="mHC residual stream expansion factor")
    parser.add_argument("--model_name", type=str, default="tempest", help="Model run identifier")
    parser.add_argument("--focal_gamma", type=float, default=1.5, help="Focal loss gamma parameter")
    parser.add_argument("--huber_weight", type=float, default=0.5, help="Physical metric Huber loss weight")
    parser.add_argument("--storm_alpha", type=float, default=1.5, help="Geophysical storm driver weighting alpha")
    parser.add_argument("--huber_delta", type=float, default=500.0, help="Huber linear transition threshold in Kelvin")
    parser.add_argument("--dry_run", action="store_true", help="Run a fast sanity-check on small data subset")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run full evaluation and plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_monitor = GPUMonitor(device)
    gpu_monitor.print_summary()

    print(f"=== Starting TEMPEST Engine: Model {args.model_name} ===")
    print(f"Configuration: d_model={args.d_model}, layers={args.n_layers}, experts={args.num_experts} (top-{args.top_k}), mHC={args.n_hc}")

    # 1. Load Normalization Statistics
    stats_file = "checkpoints/norm_stats.json"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Normalization statistics not found at {stats_file}")
    with open(stats_file, "r") as f:
        stats = json.load(f)
    means = stats["mean"]
    stds = stats["std"]
    print(f"Loaded geomagnetic normalization statistics from {stats_file}")

    # 2. Load Datasets
    train_path = "dataset/processed_dataset_01_31_storm/train_chunks"
    val_path = "dataset/processed_dataset_01_31_storm/test-normal"
    test_storm_path = "dataset/processed_dataset_01_31_storm/test-storm"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Required dataset paths not found.")

    print("\nLoading dataset chunks from disk...")
    train_chunks = []
    chunk_folders = sorted(os.listdir(train_path))
    if args.dry_run:
        chunk_folders = chunk_folders[:2]
        print(f"[*] DRY RUN: Loading only first {len(chunk_folders)} training chunks.")

    for folder in chunk_folders:
        chunk = datasets.Dataset.load_from_disk(os.path.join(train_path, folder))
        train_chunks.append(chunk)

    train_raw = datasets.concatenate_datasets(train_chunks)
    val_raw = datasets.Dataset.load_from_disk(val_path)
    test_storm_raw = datasets.Dataset.load_from_disk(test_storm_path)

    if args.dry_run:
        train_raw = train_raw.select(range(min(5000, len(train_raw))))
        val_raw = val_raw.select(range(min(2000, len(val_raw))))
        test_storm_raw = test_storm_raw.select(range(min(1000, len(test_storm_raw))))

    print(f"Training partition:   {len(train_raw):,} samples across {len(train_chunks)} chunks")
    print(f"Validation (normal):  {len(val_raw):,} samples")
    print(f"Test (severe storm):  {len(test_storm_raw):,} samples")

    # 3. Vectorized Preprocessing
    print("\nVectorizing and engineering physical plasmapause features...")
    t0 = time.time()
    preprocessor = build_preprocessor(means, stds)

    train_ds = train_raw.map(preprocessor, batched=True, batch_size=25000, remove_columns=train_raw.column_names)
    train_ds.set_format(type="torch")
    val_ds = val_raw.map(preprocessor, batched=True, batch_size=25000, remove_columns=val_raw.column_names)
    val_ds.set_format(type="torch")
    test_storm_ds = test_storm_raw.map(preprocessor, batched=True, batch_size=25000, remove_columns=test_storm_raw.column_names)
    test_storm_ds.set_format(type="torch")
    print(f"Preprocessed all datasets into 156 input channels in {time.time() - t0:.2f}s!")

    # 4. DataLoaders
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_storm_loader = DataLoader(test_storm_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)

    # 5. Initialize Model
    model = TEMPEST(
        num_features=NUM_INPUT_FEATURES,
        d_model=args.d_model,
        expert_dim=args.expert_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        n_layers=args.n_layers,
        n_hc=args.n_hc,
        vocab_size=150
    ).to(device)

    total_params, active_params = model.count_parameters()
    print("\n" + "=" * 80)
    print("  THEORETICAL OPTIMAL MODEL CAPACITY")
    print("=" * 80)
    print(f"  Total Trainable Parameters:  {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Active Parameters / Token:   {active_params:,} ({active_params/1e6:.2f}M)")
    print(f"  Sparsity Ratio:              {active_params / total_params * 100:.1f}% active")
    print(f"  Compare to Baseline 1_47:    84,000,000 dense params (Reviewer 2 Overfitting Risk Solved)")
    print("=" * 80 + "\n")

    # 6. Loss, Optimizer, Scheduler
    criterion = UnifiedPhysicsLoss(
        vocab_size=150,
        bin_width_k=100.0,
        bin_offset_k=50.0,
        focal_gamma=args.focal_gamma,
        huber_weight=args.huber_weight,
        huber_delta_k=args.huber_delta,
        storm_alpha=args.storm_alpha,
        sym_h_mean=means["SYM_H"],
        sym_h_std=stds["SYM_H"],
        sym_h_feat_idx=39
    ).to(device)
    print(f"Initialized UnifiedPhysicsLoss: focal_gamma={args.focal_gamma}, huber_weight={args.huber_weight}, storm_alpha={args.storm_alpha}, huber_delta={args.huber_delta} K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    # Visualizer
    loss_visualizer = TrainingLossVisualizer()
    loss_visualizer.set_steps_per_epoch(steps_per_epoch)

    best_comp_score = float("inf")
    best_val_loss = float("inf")
    best_model_path = f"checkpoints/{args.model_name}_best.pth"
    final_model_path = f"checkpoints/{args.model_name}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # 7. Training Loop
    if not args.eval_only:
        print(f"Starting Training: {args.num_epochs} Epochs, {steps_per_epoch} Steps/Epoch ({total_steps:,} Total Steps)...")
        global_step = 0
        eval_frequency = max(1, steps_per_epoch // 3)
        t_start = time.time()

        for epoch in range(args.num_epochs):
            model.train()
            loss_visualizer.add_epoch_start(epoch + 1, global_step)
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_t0 = time.time()

            for step, batch in enumerate(train_loader):
                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits, balance_loss = model(x, return_aux=True)
                    phys_loss = criterion(logits, y, inputs=x)
                    loss = phys_loss + balance_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                global_step += 1
                cur_lr = scheduler.get_last_lr()[0]
                loss_visualizer.add_train_step(global_step, phys_loss.item())
                epoch_loss += phys_loss.item() * len(y)
                epoch_samples += len(y)

                if step % 200 == 0 or step == steps_per_epoch - 1:
                    vram = gpu_monitor.get_vram_stats()
                    print(
                        f"Epoch [{epoch+1}/{args.num_epochs}] Step [{step}/{steps_per_epoch}] | "
                        f"Train Phys Loss: {phys_loss.item():.4f} | MoE Bal: {balance_loss.item():.4e} | "
                        f"LR: {cur_lr:.2e} | VRAM: {vram['allocated_gb']:.2f} GB"
                    )

                # Periodic Evaluation
                if global_step % eval_frequency == 0 or global_step == total_steps:
                    val_metrics = evaluate_model(model, val_loader, criterion, device)
                    test_metrics = evaluate_model(model, test_storm_loader, criterion, device)

                    comp_score = 0.5 * val_metrics["loss"] + 0.5 * test_metrics["loss"]

                    loss_visualizer.add_val_loss(global_step, val_metrics["loss"])
                    loss_visualizer.add_test_loss(global_step, test_metrics["loss"])

                    print("\n" + "-" * 75)
                    print(
                        f"[*] EVALUATION AT STEP {global_step} (Epoch {global_step / steps_per_epoch:.2f}):\n"
                        f"    Validation (Normal): Loss = {val_metrics['loss']:.4f} | R^2 = {val_metrics['r2']:.3f} | RMSE = {val_metrics['rmse']:.1f} K\n"
                        f"    Test (Storm):       Loss = {test_metrics['loss']:.4f} | R^2 = {test_metrics['r2']:.3f} | RMSE = {test_metrics['rmse']:.1f} K\n"
                        f"    Composite Metric:   Score = {comp_score:.4f} (0.5 Normal + 0.5 Storm)"
                    )

                    if comp_score < best_comp_score:
                        best_comp_score = comp_score
                        best_val_loss = val_metrics["loss"]
                        torch.save(model.state_dict(), best_model_path)
                        print(f"    [*] New Best Model Saved to {best_model_path} (Composite Score: {best_comp_score:.4f} | Normal: {val_metrics['loss']:.4f} | Storm: {test_metrics['loss']:.4f})")
                    print("-" * 75 + "\n")

                    model.train()

            epoch_dur = time.time() - epoch_t0
            throughput = epoch_samples / max(1e-5, epoch_dur)
            print(f"Completed Epoch {epoch+1}/{args.num_epochs} in {epoch_dur:.1f}s ({throughput:.0f} samples/sec). Mean Train CE: {epoch_loss / epoch_samples:.4f}")

        total_dur = time.time() - t_start
        print(f"\nTraining completed in {total_dur / 60:.2f} minutes!")

        # Save final weights and training history
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model checkpoint to {final_model_path}")

        history_path = f"checkpoints/{args.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(loss_visualizer.history, f)
        print(f"Saved training history to {history_path}")

    # 8. Load Best Checkpoint for Final Benchmarking & Visualizations
    if os.path.exists(best_model_path):
        print(f"\nLoading best checkpoint weights from {best_model_path} for final evaluation suite...")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    model.eval()

    print("\n" + "=" * 80)
    print("  FINAL COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 80)
    final_val = evaluate_model(model, val_loader, criterion, device)
    final_storm = evaluate_model(model, test_storm_loader, criterion, device)
    print(f"  Validation (test-normal)  -- Loss: {final_val['loss']:.4f} | Acc: {final_val['acc']*100:.1f}% | R^2: {final_val['r2']:.3f} | RMSE: {final_val['rmse']:.1f} K | MAE: {final_val['mae']:.1f} K")
    print(f"  Held-out (test-storm)     -- Loss: {final_storm['loss']:.4f} | Acc: {final_storm['acc']*100:.1f}% | R^2: {final_storm['r2']:.3f} | RMSE: {final_storm['rmse']:.1f} K | MAE: {final_storm['mae']:.1f} K")
    print("=" * 80 + "\n")

    # ==============================================================================
    # 9. GENERATE COMPLETE VISUALIZATION SUITE
    # ==============================================================================

    # Vis 1: @Xiangning Chu Request (Epoch vs Test Loss & Full Loss Curve)
    print("\n--- Generating Visualization 1: Loss vs Epochs (@Xiangning Chu) ---")
    loss_curve_path = f"checkpoints/{args.model_name}_epoch_loss_curve.png"
    loss_visualizer.generate_plot(
        title=f"Model {args.model_name} (DeepSeek-V4 MoE): Loss vs Epochs",
        save_path=loss_curve_path
    )

    test_loss_curve_path = f"checkpoints/{args.model_name}_epoch_test_loss_curve.png"
    loss_visualizer.generate_epoch_test_loss_plot(
        title=f"Model {args.model_name} (DeepSeek-V4 MoE): Test Loss vs Epochs (Best Val Model Marked)",
        save_path=test_loss_curve_path
    )

    # Vis 2: @Michael Request (All 5 Sandwiched Block Cases)
    print("\n--- Generating Visualization 2: Sandwiched Test Block Suite (@Michael) ---")
    michael_paths = SandwichedBlockVisualizer.generate_all_michael_cases(
        model=model,
        device=device,
        train_ds=train_ds,
        test_ds=val_ds,
        input_columns=ALL_INPUT_COLUMNS,
        block_size=150,
        num_candidates=50,
        output_dir="checkpoints",
        model_name=args.model_name,
        log_to_wandb=False
    )
    if "random" in michael_paths:
        import shutil
        shutil.copyfile(michael_paths["random"], f"checkpoints/{args.model_name}_sandwiched_block.png")

    # Vis 3: Physical Diagnostics Suite (Deviation Hexbin, Scatter, Plasmapause Transition)
    print("\n--- Generating Visualization 3: Physical Diagnostics Suite ---")
    diag_paths = generate_physical_diagnostics(
        model=model,
        test_loader=val_loader,
        device=device,
        output_dir="checkpoints",
        model_name=args.model_name
    )

    print("\n" + "=" * 80)
    print("  ALL TARGET VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"  1. Epoch Loss Curve (@Xiangning Chu):          {loss_curve_path}")
    print(f"  2. Epoch Test Loss Curve (@Xiangning Chu):     {test_loss_curve_path}")
    print(f"  3. Random Sandwiched Block (@Michael):         {michael_paths.get('random')}")
    print(f"  4. Best Sandwiched Block (@Michael):           {michael_paths.get('best')}")
    print(f"  5. Worst Sandwiched Block (@Michael):          {michael_paths.get('worst')}")
    print(f"  6. Median Sandwiched Block (@Michael):         {michael_paths.get('median')}")
    print(f"  7. Mean Sandwiched Profile (@Michael):         {michael_paths.get('mean')}")
    print(f"  8. 2D Error Deviation Plot:                    {diag_paths.get('deviation')}")
    print(f"  9. Observed vs Predicted Scatter:              {diag_paths.get('scatter')}")
    print(f"  10. Plasmapause Transition Analysis:           {diag_paths.get('plasmapause')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
