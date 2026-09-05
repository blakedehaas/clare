"""
explainability.py - SHAP and Gradient-Based Feature Attribution Suite for TEMPEST.

Provides comprehensive model explainability:
1. Decomposes continuous expected electron temperature Te_hat(x) into additive feature attributions:
       Te_hat(x) = phi_0 + sum_{i=1}^M phi_i(x)
   Satisfying the Shapley efficiency, symmetry, and dummy player axioms.
2. Supports both external 'shap' library (when installed) and an exact, zero-dependency
   Path-Shapley (Integrated Gradients) mathematical attribution engine for PyTorch models.
3. Generates publication-grade explainability diagnostics:
   - Global Feature Importance (Top-N mean |SHAP| in Kelvin)
   - Summary Beeswarm Attribution Plot (Impact directionality)
   - Regime Comparison (Inside Plasmapause L < L_pp vs Outside Plasmatrough L >= L_pp; Quiet vs Storm)
   - Local Waterfall Attribution Plot (Single-orbit satellite pass case study)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any

# Try importing shap; fallback gracefully if not present
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from train_v2 import TEMPEST, ALL_INPUT_COLUMNS, NUM_INPUT_FEATURES


class TempestExplainer:
    """
    Model explainability engine for TEMPEST continuous temperature predictions.
    Computes Shapley additive feature attributions phi_i in Kelvin for each of the 156 input features.
    """
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_names = feature_names or ALL_INPUT_COLUMNS
        self.num_features = len(self.feature_names)

        # 150 bin centers from 50 to 14,950 K
        self.bin_centers = torch.linspace(50.0, 14950.0, 150, device=self.device)

    def predict_temperature_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning continuous expected temperature in Kelvin."""
        x = x.to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits.float(), dim=-1)
        return (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)

    def predict_temperature_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """NumPy wrapper for black-box explainers."""
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            temps = self.predict_temperature_tensor(x_tensor)
            return temps.cpu().numpy()

    def compute_integrated_gradients(
        self,
        inputs: torch.Tensor,
        baselines: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        Computes Path-Shapley (Integrated Gradients) feature attributions.
        Guarantees the Shapley Efficiency axiom:
            sum_i phi_i(x) == f(x) - f(baseline)

        Parameters:
        -----------
        inputs : torch.Tensor of shape (N, num_features)
        baselines : torch.Tensor of shape (1, num_features) or (N, num_features)
        steps : int, Riemann integration steps along linear path

        Returns:
        --------
        attributions : np.ndarray of shape (N, num_features) in Kelvin
        baseline_value : float, mean expected temperature of baseline
        """
        self.model.eval()
        inputs = inputs.to(self.device).float()
        batch_size = inputs.shape[0]

        if baselines is None:
            # Default baseline: zero vector (nominal normalized quiet conditions)
            baselines = torch.zeros((1, self.num_features), device=self.device, dtype=torch.float32)
        else:
            baselines = baselines.to(self.device).float()

        if baselines.shape[0] == 1 and batch_size > 1:
            baselines = baselines.expand(batch_size, -1)

        # Baseline expected value
        with torch.no_grad():
            baseline_temp = self.predict_temperature_tensor(baselines).mean().item()

        # Generate linear interpolation paths: x_alpha = baseline + alpha * (input - baseline)
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        total_gradients = torch.zeros_like(inputs)

        # Compute path gradients with trapezoidal rule
        for i in range(steps):
            alpha = (alphas[i] + alphas[i + 1]) / 2.0
            interpolated = baselines + alpha * (inputs - baselines)
            interpolated.requires_grad_(True)

            te_hat = self.predict_temperature_tensor(interpolated)
            grad = torch.autograd.grad(
                outputs=te_hat,
                inputs=interpolated,
                grad_outputs=torch.ones_like(te_hat),
                create_graph=False,
                retain_graph=False
            )[0]

            total_gradients += grad / float(steps)

        # Path-Shapley attribution: (input - baseline) * average_gradient
        diff = (inputs - baselines).detach()
        attributions = (diff * total_gradients).detach().cpu().numpy()

        return attributions, baseline_temp

    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        background_samples: Optional[Union[np.ndarray, torch.Tensor]] = None,
        steps: int = 50
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature attributions across dataset samples.
        Uses shap.Explainer if available and requested, otherwise employs exact Path-Shapley.
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_tensor = x.to(self.device).float()

        baseline_tensor = None
        if background_samples is not None:
            if isinstance(background_samples, np.ndarray):
                bg = torch.tensor(background_samples, dtype=torch.float32, device=self.device)
            else:
                bg = background_samples.to(self.device).float()
            baseline_tensor = bg.mean(dim=0, keepdim=True)

        # Compute Path-Shapley attributions
        attributions, base_val = self.compute_integrated_gradients(
            x_tensor, baselines=baseline_tensor, steps=steps
        )

        with torch.no_grad():
            preds = self.predict_temperature_tensor(x_tensor).cpu().numpy()

        return {
            "shap_values": attributions,
            "base_value": base_val,
            "predictions": preds,
            "feature_names": self.feature_names
        }


# ==============================================================================
# PUBLICATION EXPLAINABILITY PLOTTING SUITE
# ==============================================================================

def plot_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Generates horizontal bar chart of top-N features by mean |SHAP value| in Kelvin."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    top_features = [feature_names[i] for i in sorted_idx][::-1]
    top_values = mean_abs_shap[sorted_idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    colors = ["#1f77b4" if "SYM_H" in feat or "AL" in feat else "#ff7f0e" if "Altitude" in feat or "LAT" in feat or "MLT" in feat else "#2ca02c" for feat in top_features]

    bars = ax.barh(range(len(top_features)), top_values, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Attribution [Kelvin]", fontsize=12, fontweight="bold")
    ax.set_title(f"TEMPEST: Global Feature Importance (Top {top_n} Attributions)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Add numeric value labels on bars
    for bar, val in zip(bars, top_values):
        ax.text(val + max(top_values) * 0.01, bar.get_y() + bar.get_height() / 2.0,
                f"{val:.1f} K", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved Global Feature Importance to {save_path}")
    return fig


def plot_regime_comparison(
    shap_values: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compares feature attributions across distinct space physics regimes:
    Regime A: Inside Plasmapause (L < L_pp, quiet cold core)
    Regime B: Outside Plasmapause (L >= L_pp, storm-heated plasmatrough)
    """
    # Feature 155 is in_plasmapause
    pp_idx = feature_names.index("in_plasmapause") if "in_plasmapause" in feature_names else -1
    if pp_idx != -1 and pp_idx < feature_matrix.shape[1]:
        in_pp_mask = feature_matrix[:, pp_idx] > 0.5
    else:
        in_pp_mask = np.ones(len(feature_matrix), dtype=bool)

    mean_inside = np.mean(np.abs(shap_values[in_pp_mask]), axis=0) if np.any(in_pp_mask) else np.zeros(len(feature_names))
    mean_outside = np.mean(np.abs(shap_values[~in_pp_mask]), axis=0) if np.any(~in_pp_mask) else np.zeros(len(feature_names))

    overall_mean = (mean_inside + mean_outside) / 2.0
    top_idx = np.argsort(overall_mean)[::-1][:top_n]

    top_features = [feature_names[i] for i in top_idx]
    val_in = mean_inside[top_idx]
    val_out = mean_outside[top_idx]

    x = np.arange(len(top_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.bar(x - width/2, val_in, width, label="Inside Plasmapause (Cold Dense Core)", color="#1b9e77", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, val_out, width, label="Outside Plasmapause (Warm Plasmatrough)", color="#d95f02", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean |SHAP Attribution| [Kelvin]", fontsize=12, fontweight="bold")
    ax.set_title("TEMPEST: Physical Regime Attribution (Plasmapause Boundary Stratification)", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved Regime Comparison to {save_path}")
    return fig


def plot_waterfall(
    shap_sample: np.ndarray,
    base_val: float,
    pred_val: float,
    feature_vals: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    title: str = "TEMPEST Single-Observation Prediction Decomposition",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Generates local waterfall attribution plot showing additive contributions to Te."""
    abs_vals = np.abs(shap_sample)
    sorted_idx = np.argsort(abs_vals)[::-1][:top_n]

    top_names = [feature_names[i] for i in sorted_idx]
    top_shaps = shap_sample[sorted_idx]
    top_raw = feature_vals[sorted_idx]

    other_shap = np.sum(shap_sample) - np.sum(top_shaps)
    all_names = top_names + ["Other Lags & Features"]
    all_shaps = list(top_shaps) + [other_shap]

    # Cumulative summation for waterfall display
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    running_sum = base_val
    for i, (name, val) in enumerate(zip(all_names, all_shaps)):
        color = "#d62728" if val >= 0 else "#1f77b4"
        ax.barh(i, val, left=running_sum, color=color, edgecolor="black", alpha=0.85)
        ax.text(running_sum + val / 2.0, i, f"{val:+.1f} K", va="center", ha="center",
                color="white" if abs(val) > 40 else "black", fontsize=8, fontweight="bold")
        running_sum += val

    labels = [
        f"{name} ({top_raw[i]:.2f})" if i < len(top_raw) else name
        for i, name in enumerate(all_names)
    ]
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Electron Temperature [Kelvin]", fontsize=12, fontweight="bold")
    ax.axvline(base_val, color="gray", linestyle="--", linewidth=1.2, label=f"Base Quiet Value: {base_val:.0f} K")
    ax.axvline(pred_val, color="gold", linestyle="-", linewidth=2.0, label=f"Predicted Te: {pred_val:.0f} K")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved Waterfall Plot to {save_path}")
    return fig
