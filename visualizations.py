"""
visualizations.py - Object-Oriented Visualizations and CLI for the CLARE Project.

Provides:
- TrainingLossVisualizer: Plots epoch-marked loss curves across steps with perceptually
  uniform red color scheme, best model validation point, horizontal best reference lines,
  and comprehensive statistics legend (@Xiangning Chu).
- SandwichedBlockVisualizer: Slices and plots performance across a test block sandwiched
  between two train blocks ([Train Block 1] -> [Test Block] -> [Train Block 2]), with
  shaded regions and residual analysis (@Michael).
- DataSliceVisualizer: Interactive/custom slice visualizer for arbitrary time/index ranges.
- Standalone CLI interface.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==============================================================================
# 1. Perceptually Uniform Red Color Schemes & Styling
# ==============================================================================

class ColorPalette:
    """Perceptually uniform red color scheme for loss curves and multi-line plots."""
    # Sequential red palette with clear lightness/perceptual separation
    DARK_RED = "#67001F"       # Deep burgundy / dark crimson (Test Loss)
    CRIMSON = "#D62728"        # Vivid crimson red (Validation Loss)
    CORAL_RED = "#FB6A4A"      # Soft coral / light red (Train Loss)
    GOLD_STAR = "#FFD700"      # Gold highlight for best model checkpoint
    
    # Sandwiched block shading colors
    TRAIN_BLOCK_SHADE = "#E8EEF5"  # Soft cool blue/gray for seen train data
    TEST_BLOCK_SHADE = "#FDE8E8"   # Warm peach/red for unseen test data
    BOUNDARY_LINE = "#333333"      # Dark gray boundary delimiter

    @classmethod
    def get_red_palette(cls, n_shades: int = 3) -> List[str]:
        """Returns n perceptually spaced shades of red using matplotlib Reds colormap."""
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps["Reds"]
        else:
            cmap = cm.get_cmap("Reds")
        # Sample between 0.45 and 0.95 to avoid invisible whites and muddy darks
        return [mcolors.to_hex(cmap(val)) for val in np.linspace(0.45, 0.95, n_shades)]


# ==============================================================================
# 2. TrainingLossVisualizer (@Xiangning Chu)
# ==============================================================================

class TrainingLossVisualizer:
    """
    Plots training, validation, and test loss curves across steps/epochs.
    
    Features:
    - X axis: Epochs (with continuous step mapping)
    - Y axis: Loss values (focused on Test Loss, Val Loss, Train Loss)
    - Vertical dotted black lines marking the start of each epoch
    - Point highlighting the Validation Loss of the 'best' model checkpoint
    - Horizontal dashed lines for best test, val, and train losses
    - Colored legend displaying key statistics with a perceptually uniform red scheme
    - Logging directly to Weights & Biases and saving to disk
    """

    def __init__(self, history: Optional[Dict[str, Any]] = None):
        """
        history format:
        {
            "steps": List[int],
            "train_loss": List[float],
            "val_loss": List[Tuple[int, float]],   # [(step, loss), ...]
            "test_loss": List[Tuple[int, float]],  # [(step, loss), ...]
            "epoch_starts": List[int],             # [step_ep0, step_ep1, ...]
            "steps_per_epoch": int,
            "best_val_step": Optional[int],
            "best_val_loss": Optional[float],
        }
        """
        self.history = history or {
            "steps": [],
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "epoch_starts": [],
            "steps_per_epoch": 1,
            "best_val_step": None,
            "best_val_loss": None,
        }

    def add_train_step(self, step: int, loss: float):
        self.history["steps"].append(step)
        self.history["train_loss"].append(loss)

    def add_epoch_start(self, epoch: int, step: int):
        self.history["epoch_starts"].append(step)

    def add_val_loss(self, step: int, loss: float):
        self.history["val_loss"].append((step, loss))
        current_best = self.history.get("best_val_loss")
        if current_best is None or loss < current_best:
            self.history["best_val_loss"] = loss
            self.history["best_val_step"] = step

    def add_test_loss(self, step: int, loss: float):
        self.history["test_loss"].append((step, loss))

    def set_steps_per_epoch(self, steps_per_epoch: int):
        self.history["steps_per_epoch"] = max(1, steps_per_epoch)

    def generate_plot(
        self,
        title: str = "Training & Evaluation Loss vs Epochs",
        figsize: Tuple[int, int] = (14, 7),
        dpi: int = 300,
        smooth_train_window: int = 15,
        save_path: Optional[str] = None,
        log_to_wandb: bool = False,
        wandb_key: str = "epoch_loss_curve"
    ) -> Figure:
        """
        Generates the formatted loss plot meeting all criteria.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        steps_per_epoch = self.history.get("steps_per_epoch", 1)
        steps = np.array(self.history.get("steps", []))
        train_loss = np.array(self.history.get("train_loss", []))
        val_loss_entries = self.history.get("val_loss", [])
        test_loss_entries = self.history.get("test_loss", [])
        epoch_starts = self.history.get("epoch_starts", [])

        # Map steps to fractional epochs for the continuous X axis
        epochs_train = steps / steps_per_epoch if len(steps) > 0 else np.array([])

        palette = ColorPalette.get_red_palette(3)
        train_color = palette[0]    # Soft coral red
        val_color = palette[1]      # Crimson red
        test_color = palette[2]     # Dark ruby red

        legend_stats = []

        # 1. Plot Train Loss
        if len(train_loss) > 0:
            min_train_idx = np.argmin(train_loss)
            min_train_loss = train_loss[min_train_idx]
            min_train_ep = epochs_train[min_train_idx]

            # Raw steps in faint line, smoothed in solid line if long enough
            if len(train_loss) > smooth_train_window:
                smoothed = pd.Series(train_loss).rolling(smooth_train_window, min_periods=1).mean().values
                ax.plot(epochs_train, train_loss, color=train_color, alpha=0.25, linewidth=0.8)
                ax.plot(epochs_train, smoothed, color=train_color, linewidth=1.8,
                        label=f"Train Loss (min: {min_train_loss:.4f} @ Ep {min_train_ep:.1f})")
            else:
                ax.plot(epochs_train, train_loss, color=train_color, linewidth=1.5,
                        label=f"Train Loss (min: {min_train_loss:.4f} @ Ep {min_train_ep:.1f})")

            # Best train loss point & horizontal line
            ax.scatter([min_train_ep], [min_train_loss], color=train_color, s=70, zorder=5, marker="o")
            ax.axhline(y=min_train_loss, color=train_color, linestyle="--", linewidth=0.9, alpha=0.6)
            legend_stats.append(f"Best Train Loss: {min_train_loss:.4f}")

        # 2. Plot Validation Loss
        if len(val_loss_entries) > 0:
            val_steps = np.array([e[0] for e in val_loss_entries])
            val_losses = np.array([e[1] for e in val_loss_entries])
            val_epochs = val_steps / steps_per_epoch

            min_val_idx = np.argmin(val_losses)
            min_val_loss = val_losses[min_val_idx]
            min_val_ep = val_epochs[min_val_idx]
            min_val_step = val_steps[min_val_idx]

            ax.plot(val_epochs, val_losses, color=val_color, linewidth=2.2, marker="s", markersize=6,
                    label=f"Val Loss (min: {min_val_loss:.4f} @ Ep {min_val_ep:.1f})")
            ax.axhline(y=min_val_loss, color=val_color, linestyle="--", linewidth=1.1, alpha=0.7)
            legend_stats.append(f"Best Val Loss: {min_val_loss:.4f}")

            # Highlight Point for Validation Loss of 'best' model checkpoint (@Xiangning Chu)
            best_val_loss = self.history.get("best_val_loss", min_val_loss)
            best_val_step = self.history.get("best_val_step", min_val_step)
            best_val_ep = best_val_step / steps_per_epoch

            ax.scatter([best_val_ep], [best_val_loss], color=ColorPalette.GOLD_STAR, edgecolors=val_color,
                       s=200, zorder=10, marker="*",
                       label=f"★ Best Model (Val Loss: {best_val_loss:.4f} @ Ep {best_val_ep:.1f})")

            # Callout annotation
            ax.annotate(
                f"Best Model\nVal: {best_val_loss:.4f}\nEp: {best_val_ep:.1f}",
                xy=(best_val_ep, best_val_loss),
                xytext=(15, 25), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9E6", edgecolor=val_color, alpha=0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color=val_color, lw=1.5),
                fontsize=9, fontweight="bold", color="#333333"
            )

        # 3. Plot Test Loss (@Xiangning Chu)
        if len(test_loss_entries) > 0:
            test_steps = np.array([e[0] for e in test_loss_entries])
            test_losses = np.array([e[1] for e in test_loss_entries])
            test_epochs = test_steps / steps_per_epoch

            min_test_idx = np.argmin(test_losses)
            min_test_loss = test_losses[min_test_idx]
            min_test_ep = test_epochs[min_test_idx]

            ax.plot(test_epochs, test_losses, color=test_color, linewidth=2.5, marker="^", markersize=7,
                    label=f"Test Loss (min: {min_test_loss:.4f} @ Ep {min_test_ep:.1f})")
            ax.axhline(y=min_test_loss, color=test_color, linestyle="--", linewidth=1.1, alpha=0.7)
            ax.scatter([min_test_ep], [min_test_loss], color=test_color, s=90, zorder=6, marker="^")
            legend_stats.append(f"Best Test Loss: {min_test_loss:.4f}")

        # 4. Vertical dotted black lines marking the start of each epoch
        max_epoch = int(np.ceil(epochs_train[-1])) if len(epochs_train) > 0 else len(epoch_starts)
        marked_epochs = set()

        line_alpha = 0.35 if max_epoch > 15 else 0.7
        ep_interval = 5 if max_epoch > 50 else (2 if max_epoch > 20 else 1)

        # If explicit epoch_starts recorded
        for ep_idx, ep_step in enumerate(epoch_starts):
            ep_val = ep_step / steps_per_epoch
            if round(ep_val) % ep_interval == 0:
                ax.axvline(x=ep_val, color="black", linestyle=":", linewidth=1.0, alpha=line_alpha)
            marked_epochs.add(round(ep_val))

        # Also ensure every integer epoch line exists up to max_epoch
        for ep in range(1, max_epoch + 1):
            if ep not in marked_epochs and ep % ep_interval == 0:
                ax.axvline(x=ep, color="black", linestyle=":", linewidth=1.0, alpha=line_alpha)

        # Add single representative dotted black line to legend
        ax.plot([], [], color="black", linestyle=":", linewidth=1.2, label=f"Epoch Boundary (interval: {ep_interval})")

        # 5. Axes & Styling
        ax.set_xlabel("Epochs", fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(epochs_train) > 0:
            ax.set_xlim(left=0, right=max(1.0, float(np.max(epochs_train))))

        # Key Statistics Legend
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="#FAFAFA",
                  edgecolor="#CCCCCC", fontsize=10)

        plt.tight_layout()

        # 6. Save and Log
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[TrainingLossVisualizer] Saved plot to: {save_path}")

        if log_to_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({wandb_key: wandb.Image(fig)})
            print(f"[TrainingLossVisualizer] Logged plot to W&B as '{wandb_key}'")

        return fig

    def generate_epoch_test_loss_plot(
        self,
        title: str = "Test Loss vs Epochs (Best Val Model Marked)",
        figsize: Tuple[int, int] = (14, 7),
        dpi: int = 300,
        save_path: Optional[str] = None,
        log_to_wandb: bool = False,
        wandb_key: str = "epoch_test_loss_curve"
    ) -> Figure:
        """
        Generates the target visualization requested by @Xiangning Chu:
        - X axis: Epochs
        - Y axis: Test Loss
        - Point for Validation Loss of 'best' model checkpoint prominently marked
        - Vertical dotted black lines at epoch boundaries
        - Horizontal dashed line for minimum test loss
        - Perceptually uniform red color scheme
        - Clear legend and key metrics summary box
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        steps_per_epoch = self.history.get("steps_per_epoch", 1)
        test_loss_entries = self.history.get("test_loss", [])
        val_loss_entries = self.history.get("val_loss", [])
        train_loss = np.array(self.history.get("train_loss", []))
        steps = np.array(self.history.get("steps", []))
        epoch_starts = self.history.get("epoch_starts", [])

        palette = ColorPalette.get_red_palette(3)
        train_color = palette[0]    # Soft coral red
        val_color = palette[1]      # Crimson red
        test_color = palette[2]     # Dark ruby/burgundy red

        # 1. Primary Curve: Test Loss (@Xiangning Chu)
        min_test_loss, min_test_ep = None, None
        if len(test_loss_entries) > 0:
            test_steps = np.array([e[0] for e in test_loss_entries])
            test_losses = np.array([e[1] for e in test_loss_entries])
            test_epochs = test_steps / steps_per_epoch

            min_test_idx = np.argmin(test_losses)
            min_test_loss = test_losses[min_test_idx]
            min_test_ep = test_epochs[min_test_idx]

            ax.plot(test_epochs, test_losses, color=test_color, linewidth=2.8,
                    marker="o", markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                    label=f"Test Loss (min: {min_test_loss:.4f} @ Ep {min_test_ep:.2f})", zorder=8)
            ax.axhline(y=min_test_loss, color=test_color, linestyle="--", linewidth=1.2, alpha=0.75,
                       label=f"Min Test Loss Ref ({min_test_loss:.4f})")
            ax.scatter([min_test_ep], [min_test_loss], color=test_color, s=120, zorder=9, marker="o", edgecolors="black")

        # 2. Context Curve: Validation Loss
        best_val_loss, best_val_ep = None, None
        if len(val_loss_entries) > 0:
            val_steps = np.array([e[0] for e in val_loss_entries])
            val_losses = np.array([e[1] for e in val_loss_entries])
            val_epochs = val_steps / steps_per_epoch

            min_val_idx = np.argmin(val_losses)
            min_val_loss = val_losses[min_val_idx]
            min_val_ep = val_epochs[min_val_idx]

            ax.plot(val_epochs, val_losses, color=val_color, linewidth=1.8, linestyle="-.",
                    marker="s", markersize=5, alpha=0.85,
                    label=f"Validation Loss (min: {min_val_loss:.4f} @ Ep {min_val_ep:.2f})", zorder=7)

            # Point for Validation Loss of 'best' model checkpoint (@Xiangning Chu)
            best_val_loss = self.history.get("best_val_loss", min_val_loss)
            best_val_step = self.history.get("best_val_step", val_steps[min_val_idx])
            best_val_ep = best_val_step / steps_per_epoch

            ax.scatter([best_val_ep], [best_val_loss], color=ColorPalette.GOLD_STAR, edgecolors="#B8860B",
                       s=260, zorder=15, marker="*", linewidths=1.2,
                       label=f"★ Best Model Checkpoint (Val Loss: {best_val_loss:.4f} @ Ep {best_val_ep:.2f})")

            # Callout annotation
            ax.annotate(
                f"Best Model Checkpoint\nVal Loss: {best_val_loss:.4f}\nEpoch: {best_val_ep:.2f}",
                xy=(best_val_ep, best_val_loss),
                xytext=(25, 30), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9E6", edgecolor="#D4AF37", lw=1.5, alpha=0.95),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#B8860B", lw=2.0),
                fontsize=9.5, fontweight="bold", color="#222222", zorder=16
            )

        # 3. Epoch boundaries (vertical dotted black lines)
        epochs_train = steps / steps_per_epoch if len(steps) > 0 else np.array([])
        max_epoch = int(np.ceil(epochs_train[-1])) if len(epochs_train) > 0 else len(epoch_starts)
        if max_epoch == 0 and len(test_loss_entries) > 0:
            max_epoch = int(np.ceil(test_epochs[-1]))
        max_epoch = max(1, max_epoch)

        for ep in range(1, max_epoch + 1):
            ax.axvline(x=ep, color="black", linestyle=":", linewidth=1.2, alpha=0.6, zorder=3)

        ax.plot([], [], color="black", linestyle=":", linewidth=1.2, label="Epoch Boundary")

        # 4. Labels, Title, and Legend
        ax.set_xlabel("Epochs", fontsize=13, fontweight="bold")
        ax.set_ylabel("Test Loss", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=1)
        ax.set_xlim(left=0, right=max_epoch)

        ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="#FAFAFA",
                  edgecolor="#CCCCCC", fontsize=10)

        plt.tight_layout()

        # 5. Save and Log
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[TrainingLossVisualizer] Saved Epoch Test Loss plot to: {save_path}")

        if log_to_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({wandb_key: wandb.Image(fig)})
            print(f"[TrainingLossVisualizer] Logged Epoch Test Loss plot to W&B as '{wandb_key}'")

        return fig


# ==============================================================================
# 3. SandwichedBlockVisualizer (@Michael)
# ==============================================================================

class SandwichedBlockVisualizer:
    """
    Visualizes model performance over a test block sandwiched between two train blocks:
    [Train Block 1] -> [Sandwiched Test Block] -> [Train Block 2]
    
    Features:
    - Top panel: Observed vs Predicted values over time with shaded backgrounds:
        - Train Block 1 (light cool background, labeled)
        - Sandwiched Test Block (distinct warm highlight, labeled as Unseen)
        - Train Block 2 (light cool background, labeled)
    - Vertical dashed boundary separators between blocks
    - Bottom panel: Deviation / Residual curve (Pred - Obs) across the transition
    - Summary metrics box (R², RMSE, MAE) for each block
    - Saves figure and optionally logs to Weights & Biases
    """

    def __init__(
        self,
        time_col: str = "DateTimeFormatted",
        target_col: str = "Te1",
        pred_col: str = "Te1_pred"
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.pred_col = pred_col

    @staticmethod
    def extract_sandwiched_slice(
        df_combined: pd.DataFrame,
        split_col: str = "split",
        train_label: str = "train",
        test_label: str = "test",
        min_block_size: int = 50,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Finds contiguous blocks in chronological order and picks a test block that is
        literally sandwiched between two train blocks.
        """
        df = df_combined.copy()
        if "DateTimeFormatted" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["DateTimeFormatted"]):
            df["DateTimeFormatted"] = pd.to_datetime(df["DateTimeFormatted"])
            df.sort_values(by="DateTimeFormatted", inplace=True)

        # Identify contiguous blocks
        df["_block_id"] = (df[split_col] != df[split_col].shift(1)).cumsum()
        grouped = df.groupby("_block_id").agg(
            split=(split_col, "first"),
            count=(split_col, "count"),
            start_idx=("_block_id", lambda x: x.index[0]),
            end_idx=("_block_id", lambda x: x.index[-1])
        ).reset_index()

        # Find candidates: index i is test, i-1 is train, i+1 is train
        candidates = []
        for i in range(1, len(grouped) - 1):
            if (grouped.loc[i, "split"] == test_label and
                grouped.loc[i-1, "split"] == train_label and
                grouped.loc[i+1, "split"] == train_label and
                grouped.loc[i, "count"] >= min_block_size):
                candidates.append(i)

        rng = np.random.RandomState(random_state)
        if candidates:
            chosen_idx = rng.choice(candidates)
            train1_block = grouped.loc[chosen_idx - 1]
            test_block = grouped.loc[chosen_idx]
            train2_block = grouped.loc[chosen_idx + 1]

            slice_df = df.loc[train1_block["start_idx"]:train2_block["end_idx"]].copy()
            meta = {
                "train1_len": train1_block["count"],
                "test_len": test_block["count"],
                "train2_len": train2_block["count"],
                "test_start_time": slice_df.loc[test_block["start_idx"], "DateTimeFormatted"],
                "test_end_time": slice_df.loc[test_block["end_idx"], "DateTimeFormatted"],
            }
            return slice_df, meta
        else:
            # Fallback: create a partition by slicing a contiguous window
            n = len(df)
            block_size = max(min_block_size, n // 10)
            mid = n // 2
            t1_start = max(0, mid - block_size)
            t2_end = min(n, mid + 2 * block_size)
            sub_df = df.iloc[t1_start:t2_end].copy()
            
            # Label middle third as test, flanks as train
            n_sub = len(sub_df)
            third = n_sub // 3
            sub_df[split_col] = [train_label] * third + [test_label] * third + [train_label] * (n_sub - 2 * third)
            meta = {
                "train1_len": third,
                "test_len": third,
                "train2_len": n_sub - 2 * third,
                "test_start_time": sub_df.iloc[third]["DateTimeFormatted"] if "DateTimeFormatted" in sub_df else None,
                "test_end_time": sub_df.iloc[2 * third]["DateTimeFormatted"] if "DateTimeFormatted" in sub_df else None,
            }
            return sub_df, meta

    def generate_plot(
        self,
        df_slice: pd.DataFrame,
        train1_mask: np.ndarray,
        test_mask: np.ndarray,
        train2_mask: np.ndarray,
        title: str = "Performance Across Sandwiched Test Block",
        figsize: Tuple[int, int] = (16, 9),
        dpi: int = 300,
        log_scale_y: bool = True,
        save_path: Optional[str] = None,
        log_to_wandb: bool = False,
        wandb_key: str = "sandwiched_test_block"
    ) -> Figure:
        """
        Renders the sandwiched block visualization.
        """
        df = df_slice.copy()
        if self.time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            df[self.time_col] = pd.to_datetime(df[self.time_col])

        has_time = self.time_col in df.columns
        x_values = df[self.time_col].values if has_time else np.arange(len(df))

        y_true = df[self.target_col].values
        y_pred = df[self.pred_col].values
        residuals = y_pred - y_true

        # Calculate metrics for each block
        def calc_metrics(m):
            if np.sum(m) == 0:
                return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
            t, p = y_true[m], y_pred[m]
            return {
                "r2": r2_score(t, p) if len(t) > 1 else 0.0,
                "rmse": np.sqrt(mean_squared_error(t, p)),
                "mae": mean_absolute_error(t, p)
            }

        m_tr1 = calc_metrics(train1_mask)
        m_test = calc_metrics(test_mask)
        m_tr2 = calc_metrics(train2_mask)

        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=figsize, dpi=dpi,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )

        # 1. Top Panel: Observed vs Predicted
        ax_main.plot(x_values, y_true, label="Observed (Ground Truth)", color="#1F77B4",
                     linewidth=1.8, alpha=0.9)
        ax_main.plot(x_values, y_pred, label="Model Prediction", color="#D62728",
                     linewidth=2.0, linestyle="--", alpha=0.95)

        if log_scale_y:
            ax_main.set_yscale("log")
            ax_main.set_ylabel(r"$\log_{10}$ Electron Temp ($T_{e1}$) [K]", fontsize=12, fontweight="bold")
        else:
            ax_main.set_ylabel("Electron Temp ($T_{e1}$) [K]", fontsize=12, fontweight="bold")

        ax_main.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

        # 2. Bottom Panel: Residuals
        ax_res.plot(x_values, residuals, color="#8B0000", linewidth=1.2, label="Residual (Pred - Obs)")
        ax_res.axhline(0, color="black", linestyle=":", linewidth=1.2)
        ax_res.set_ylabel("Residual [K]", fontsize=11, fontweight="bold")
        ax_res.grid(True, linestyle="--", alpha=0.4)

        # 3. Apply Sandwiched Shading
        def shade_region(mask, color, label_text):
            if not np.any(mask):
                return
            x_m = x_values[mask]
            start_x, end_x = x_m[0], x_m[-1]
            for ax in (ax_main, ax_res):
                ax.axvspan(start_x, end_x, facecolor=color, alpha=0.35, zorder=0)
            # Add top label text
            mid_x = start_x + (end_x - start_x) / 2
            ax_main.text(
                mid_x, 0.96, label_text, transform=ax_main.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85)
            )

        shade_region(train1_mask, ColorPalette.TRAIN_BLOCK_SHADE, "Train Block 1 (Seen)")
        shade_region(test_mask, ColorPalette.TEST_BLOCK_SHADE, "Sandwiched Test Block (Unseen)")
        shade_region(train2_mask, ColorPalette.TRAIN_BLOCK_SHADE, "Train Block 2 (Seen)")

        # Vertical boundary markers between blocks
        if np.any(train1_mask) and np.any(test_mask):
            b1 = x_values[test_mask][0]
            for ax in (ax_main, ax_res):
                ax.axvline(b1, color=ColorPalette.BOUNDARY_LINE, linestyle="--", linewidth=1.2, alpha=0.8)

        if np.any(test_mask) and np.any(train2_mask):
            b2 = x_values[test_mask][-1]
            for ax in (ax_main, ax_res):
                ax.axvline(b2, color=ColorPalette.BOUNDARY_LINE, linestyle="--", linewidth=1.2, alpha=0.8)

        # 4. Format X-axis
        if has_time:
            ax_res.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            fig.autofmt_xdate()
            ax_res.set_xlabel("Time (UTC)", fontsize=12, fontweight="bold")
        else:
            ax_res.set_xlabel("Sample Index", fontsize=12, fontweight="bold")

        # 5. Legends and Metrics Box
        ax_main.legend(loc="upper left", framealpha=0.92, facecolor="#FAFAFA", edgecolor="#CCCCCC")

        # Metrics text box positioned cleanly without colliding with block headers
        metrics_text = (
            f"SANDWICHED TEST BLOCK\n"
            f"R²: {m_test['r2']:.3f} | RMSE: {m_test['rmse']:.1f} K | MAE: {m_test['mae']:.1f} K\n\n"
            f"TRAIN CONTEXT (FLANKS)\n"
            f"Block 1: R²={m_tr1['r2']:.3f}, RMSE={m_tr1['rmse']:.1f} K\n"
            f"Block 2: R²={m_tr2['r2']:.3f}, RMSE={m_tr2['rmse']:.1f} K"
        )
        ax_main.text(
            0.98, 0.88, metrics_text, transform=ax_main.transAxes,
            ha="right", va="top", fontsize=9.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#B0B0B0", alpha=0.92)
        )

        plt.tight_layout()

        # 6. Save and Log
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SandwichedBlockVisualizer] Saved plot to: {save_path}")

        if log_to_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({wandb_key: wandb.Image(fig)})
            print(f"[SandwichedBlockVisualizer] Logged plot to W&B as '{wandb_key}'")

        return fig

    def generate_mean_sandwiched_plot(
        self,
        candidate_blocks: List[Dict[str, Any]],
        title: str = "Mean Performance Over All Sandwiched Blocks",
        figsize: Tuple[int, int] = (16, 9),
        dpi: int = 300,
        log_scale_y: bool = True,
        save_path: Optional[str] = None,
        log_to_wandb: bool = False,
        wandb_key: str = "sandwiched_mean_all_blocks"
    ) -> Figure:
        """
        Generates the mean performance plot across all candidate sandwiched blocks (@Michael):
        - X axis: Normalized relative sample index within the sandwiched window
        - Top panel: Mean observed profile (± 1 std band) vs Mean predicted profile (± 1 std band)
        - Bottom panel: Mean residual profile (± 1 std band) with zero-reference line
        - Shaded regions for Train Block 1, Sandwiched Test Block, Train Block 2
        - Vertical dashed lines delimiting the test block
        - Summary metrics box reporting mean R², RMSE, MAE (± std) across all blocks
        """
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=figsize, dpi=dpi,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )

        t1_len = candidate_blocks[0]["train1_len"]
        t_len = candidate_blocks[0]["test_len"]
        t2_len = candidate_blocks[0]["train2_len"]
        window_len = t1_len + t_len + t2_len
        x_rel = np.arange(window_len)

        true_arr = np.array([b["y_true"] for b in candidate_blocks])
        pred_arr = np.array([b["y_pred"] for b in candidate_blocks])
        res_arr = pred_arr - true_arr

        mean_true = np.mean(true_arr, axis=0)
        std_true = np.std(true_arr, axis=0)
        mean_pred = np.mean(pred_arr, axis=0)
        std_pred = np.std(pred_arr, axis=0)
        mean_res = np.mean(res_arr, axis=0)
        std_res = np.std(res_arr, axis=0)

        r2_list = [b["r2"] for b in candidate_blocks]
        rmse_list = [b["rmse"] for b in candidate_blocks]
        mae_list = [b["mae"] for b in candidate_blocks]
        mean_r2, std_r2 = float(np.mean(r2_list)), float(np.std(r2_list))
        mean_rmse, std_rmse = float(np.mean(rmse_list)), float(np.std(rmse_list))
        mean_mae, std_mae = float(np.mean(mae_list)), float(np.std(mae_list))

        # Top panel: Mean Observed vs Mean Predicted
        ax_main.plot(x_rel, mean_true, label="Mean Observed (Ground Truth)", color="#1F77B4", linewidth=2.0)
        ax_main.fill_between(x_rel, np.maximum(1, mean_true - std_true), mean_true + std_true,
                             color="#1F77B4", alpha=0.22, label=r"Observed $\pm 1\sigma$")

        ax_main.plot(x_rel, mean_pred, label="Mean Model Prediction", color="#D62728", linewidth=2.0, linestyle="--")
        ax_main.fill_between(x_rel, np.maximum(1, mean_pred - std_pred), mean_pred + std_pred,
                             color="#D62728", alpha=0.22, label=r"Predicted $\pm 1\sigma$")

        if log_scale_y:
            ax_main.set_yscale("log")
            ax_main.set_ylabel(r"$\log_{10}$ Electron Temp ($T_{e1}$) [K]", fontsize=12, fontweight="bold")
        else:
            ax_main.set_ylabel("Electron Temp ($T_{e1}$) [K]", fontsize=12, fontweight="bold")

        ax_main.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

        # Bottom panel: Residuals
        ax_res.plot(x_rel, mean_res, color="#8B0000", linewidth=1.5, label="Mean Residual (Pred - Obs)")
        ax_res.fill_between(x_rel, mean_res - std_res, mean_res + std_res,
                            color="#8B0000", alpha=0.25, label=r"Residual $\pm 1\sigma$")
        ax_res.axhline(0, color="black", linestyle=":", linewidth=1.2)
        ax_res.set_ylabel("Residual [K]", fontsize=11, fontweight="bold")
        ax_res.set_xlabel("Relative Window Sample Index", fontsize=12, fontweight="bold")
        ax_res.grid(True, linestyle="--", alpha=0.4)

        # Apply Sandwiched Shading
        m1 = np.zeros(window_len, dtype=bool); m1[:t1_len] = True
        m2 = np.zeros(window_len, dtype=bool); m2[t1_len:t1_len + t_len] = True
        m3 = np.zeros(window_len, dtype=bool); m3[t1_len + t_len:] = True

        def shade_region(mask, color, label_text):
            x_m = x_rel[mask]
            start_x, end_x = x_m[0], x_m[-1]
            for ax in (ax_main, ax_res):
                ax.axvspan(start_x, end_x, facecolor=color, alpha=0.35, zorder=0)
            mid_x = start_x + (end_x - start_x) / 2
            ax_main.text(
                mid_x, 0.96, label_text, transform=ax_main.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85)
            )

        shade_region(m1, ColorPalette.TRAIN_BLOCK_SHADE, f"Train Block 1 ({t1_len} pts)")
        shade_region(m2, ColorPalette.TEST_BLOCK_SHADE, f"Sandwiched Test Block ({t_len} pts, Unseen)")
        shade_region(m3, ColorPalette.TRAIN_BLOCK_SHADE, f"Train Block 2 ({t2_len} pts)")

        # Vertical boundary markers
        b1 = t1_len
        b2 = t1_len + t_len
        for ax in (ax_main, ax_res):
            ax.axvline(b1, color=ColorPalette.BOUNDARY_LINE, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axvline(b2, color=ColorPalette.BOUNDARY_LINE, linestyle="--", linewidth=1.2, alpha=0.8)

        ax_main.legend(loc="upper left", framealpha=0.92, facecolor="#FAFAFA", edgecolor="#CCCCCC")
        ax_res.legend(loc="upper left", framealpha=0.92, facecolor="#FAFAFA", edgecolor="#CCCCCC")

        metrics_text = (
            f"AGGREGATE SUMMARY ({len(candidate_blocks)} BLOCKS)\n"
            f"Test Block Mean R²:   {mean_r2:.3f} ± {std_r2:.3f}\n"
            f"Test Block Mean RMSE: {mean_rmse:.1f} ± {std_rmse:.1f} K\n"
            f"Test Block Mean MAE:  {mean_mae:.1f} ± {std_mae:.1f} K\n\n"
            f"CONTEXT CONFIGURATION\n"
            f"Train Block 1: {t1_len} pts (Seen)\n"
            f"Sandwiched Test: {t_len} pts (Unseen)\n"
            f"Train Block 2: {t2_len} pts (Seen)"
        )
        ax_main.text(
            0.98, 0.88, metrics_text, transform=ax_main.transAxes,
            ha="right", va="top", fontsize=9.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#B0B0B0", alpha=0.92)
        )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[SandwichedBlockVisualizer] Saved Mean Sandwiched plot to: {save_path}")

        if log_to_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({wandb_key: wandb.Image(fig)})
            print(f"[SandwichedBlockVisualizer] Logged Mean Sandwiched plot to W&B as '{wandb_key}'")

        return fig

    @classmethod
    def generate_all_michael_cases(
        cls,
        model: Any,
        device: Any,
        train_ds: Any,
        test_ds: Any,
        input_columns: Optional[List[str]] = None,
        block_size: int = 150,
        num_candidates: int = 50,
        output_dir: str = "checkpoints",
        model_name: str = "1_47",
        log_to_wandb: bool = False
    ) -> Dict[str, str]:
        """
        Generates all 5 separate target plots requested by @Michael:
        1. Random test block between two train blocks
        2. Best performance test block between two train blocks
        3. Worst performance test block between two train blocks
        4. Median performance test block between two train blocks
        5. Mean performance over all train blocks / sandwiched blocks
        """
        os.makedirs(output_dir, exist_ok=True)
        model.eval()

        def extract_slice_arrays(ds, start, end):
            if isinstance(ds, dict) and "input_ids" in ds:
                n_len = len(ds["input_ids"])
                start = max(0, min(start, n_len - 1))
                end = max(start + 1, min(end, n_len))
                x = ds["input_ids"][start:end]
                y = ds["label"][start:end] if "label" in ds else np.zeros(len(x))
                x_np = x.cpu().numpy().astype(np.float32) if isinstance(x, torch.Tensor) else np.array(x, dtype=np.float32)
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                return x_np, y_np

            n_len = len(ds)
            start = max(0, min(start, n_len - 1))
            end = max(start + 1, min(end, n_len))

            if hasattr(ds, "iloc"):
                sub = ds.iloc[start:end]
                if "input_ids" in sub.columns:
                    x = np.vstack(sub["input_ids"].values)
                elif input_columns:
                    cols = [c for c in input_columns if c in sub.columns]
                    x = sub[cols].values.astype(np.float32)
                else:
                    x = sub.values.astype(np.float32)
                if "label" in sub.columns:
                    y = sub["label"].values
                elif "Te1" in sub.columns:
                    y = (sub["Te1"].values // 100).clip(0, 149)
                else:
                    y = np.zeros(len(x))
                return x, y
            elif hasattr(ds, "__getitem__"):
                sub = ds[start:end]
                if isinstance(sub, tuple):
                    x, y = sub[0], sub[1]
                elif isinstance(sub, dict):
                    x, y = sub["input_ids"], sub["label"]
                else:
                    x, y = sub, np.zeros(len(sub))

                if isinstance(x, torch.Tensor):
                    x_np = x.cpu().numpy().astype(np.float32)
                else:
                    x_np = np.array(x, dtype=np.float32)

                if isinstance(y, torch.Tensor):
                    y_np = y.cpu().numpy()
                else:
                    y_np = np.array(y)

                return x_np, y_np
            raise ValueError(f"Unsupported dataset format: {type(ds)}")

        candidate_blocks = []
        n_train = len(train_ds["input_ids"]) if isinstance(train_ds, dict) and "input_ids" in train_ds else len(train_ds)
        n_test = len(test_ds["input_ids"]) if isinstance(test_ds, dict) and "input_ids" in test_ds else len(test_ds)

        step_test = max(block_size, (n_test - block_size) // max(1, num_candidates))
        total_cand = min(num_candidates, n_test // block_size, (n_train - block_size) // (2 * block_size))
        total_cand = max(5, total_cand)

        with torch.no_grad():
            for k in range(total_cand):
                tr1_start = (k * 2 * block_size) % max(1, (n_train - 2 * block_size))
                tr1_end = tr1_start + block_size
                te_start = (k * step_test) % max(1, (n_test - block_size))
                te_end = te_start + block_size
                tr2_start = tr1_end
                tr2_end = tr2_start + block_size

                x_tr1, y_tr1 = extract_slice_arrays(train_ds, tr1_start, tr1_end)
                x_te, y_te = extract_slice_arrays(test_ds, te_start, te_end)
                x_tr2, y_tr2 = extract_slice_arrays(train_ds, tr2_start, tr2_end)

                t_tr1 = torch.tensor(x_tr1, dtype=torch.float32, device=device)
                t_te = torch.tensor(x_te, dtype=torch.float32, device=device)
                t_tr2 = torch.tensor(x_tr2, dtype=torch.float32, device=device)

                pred_tr1 = torch.argmax(model(t_tr1), dim=1).cpu().numpy() * 100 + 50
                pred_te = torch.argmax(model(t_te), dim=1).cpu().numpy() * 100 + 50
                pred_tr2 = torch.argmax(model(t_tr2), dim=1).cpu().numpy() * 100 + 50

                te_true_k = y_te * 100 + 50
                y_true = np.concatenate([y_tr1 * 100 + 50, te_true_k, y_tr2 * 100 + 50])
                y_pred = np.concatenate([pred_tr1, pred_te, pred_tr2])

                r2 = float(r2_score(te_true_k, pred_te)) if len(te_true_k) > 1 else 0.0
                rmse = float(np.sqrt(mean_squared_error(te_true_k, pred_te)))
                mae = float(mean_absolute_error(te_true_k, pred_te))

                candidate_blocks.append({
                    "candidate_idx": k,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "train1_len": len(x_tr1),
                    "test_len": len(x_te),
                    "train2_len": len(x_tr2),
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                })

        ranked = sorted(candidate_blocks, key=lambda b: b["r2"])
        best_block = ranked[-1]
        worst_block = ranked[0]
        median_block = ranked[len(ranked) // 2]

        rng = np.random.RandomState(42)
        random_idx = rng.randint(0, len(ranked))
        random_block = candidate_blocks[random_idx]

        vis = cls()
        paths = {}

        def plot_single_candidate(cand, title, save_path, wandb_key):
            n = len(cand["y_true"])
            df_plot = pd.DataFrame({
                "Te1": cand["y_true"],
                "Te1_pred": cand["y_pred"],
            })
            m1 = np.zeros(n, dtype=bool); m1[:cand["train1_len"]] = True
            m2 = np.zeros(n, dtype=bool); m2[cand["train1_len"]:cand["train1_len"] + cand["test_len"]] = True
            m3 = np.zeros(n, dtype=bool); m3[cand["train1_len"] + cand["test_len"]:] = True

            vis.generate_plot(
                df_plot, m1, m2, m3,
                title=title,
                save_path=save_path,
                log_to_wandb=log_to_wandb,
                wandb_key=wandb_key
            )
            return save_path

        # 1. Random Test Block
        p_random = os.path.join(output_dir, f"{model_name}_sandwiched_random.png")
        paths["random"] = plot_single_candidate(
            random_block,
            f"Model {model_name}: Random Sandwiched Test Block (R² = {random_block['r2']:.3f})",
            p_random, "sandwiched_random"
        )

        # 2. Best Performance Block
        p_best = os.path.join(output_dir, f"{model_name}_sandwiched_best.png")
        paths["best"] = plot_single_candidate(
            best_block,
            f"Model {model_name}: Best Performance Sandwiched Test Block (R² = {best_block['r2']:.3f})",
            p_best, "sandwiched_best"
        )

        # 3. Worst Performance Block
        p_worst = os.path.join(output_dir, f"{model_name}_sandwiched_worst.png")
        paths["worst"] = plot_single_candidate(
            worst_block,
            f"Model {model_name}: Worst Performance Sandwiched Test Block (R² = {worst_block['r2']:.3f})",
            p_worst, "sandwiched_worst"
        )

        # 4. Median Performance Block
        p_median = os.path.join(output_dir, f"{model_name}_sandwiched_median.png")
        paths["median"] = plot_single_candidate(
            median_block,
            f"Model {model_name}: Median Performance Sandwiched Test Block (R² = {median_block['r2']:.3f})",
            p_median, "sandwiched_median"
        )

        # 5. Mean Performance Over All Blocks
        p_mean = os.path.join(output_dir, f"{model_name}_sandwiched_mean_all_blocks.png")
        vis.generate_mean_sandwiched_plot(
            candidate_blocks,
            title=f"Model {model_name}: Mean Performance Over All Sandwiched Blocks (N={len(candidate_blocks)})",
            save_path=p_mean,
            log_to_wandb=log_to_wandb,
            wandb_key="sandwiched_mean_all_blocks"
        )
        paths["mean"] = p_mean

        return paths


# ==============================================================================
# 4. DataSliceVisualizer
# ==============================================================================

class DataSliceVisualizer:
    """
    Evaluates and plots any arbitrary slice of the data (by time, Kp, altitude, etc.).
    """

    def __init__(self, time_col: str = "DateTimeFormatted", target_col: str = "Te1"):
        self.time_col = time_col
        self.target_col = target_col

    def plot_slice(
        self,
        df_slice: pd.DataFrame,
        pred_col: str = "Te1_pred",
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        has_time = self.time_col in df_slice.columns
        x = df_slice[self.time_col] if has_time else np.arange(len(df_slice))

        ax.plot(x, df_slice[self.target_col], label="Observed", color="#1F77B4", linewidth=1.5)
        if pred_col in df_slice.columns:
            ax.plot(x, df_slice[pred_col], label="Predicted", color="#D62728", linewidth=1.8, linestyle="--")

        ax.set_yscale("log")
        ax.set_ylabel(r"Electron Temp ($T_{e1}$) [K]", fontsize=12)
        ax.set_title(title or f"Data Slice ({len(df_slice)} samples)", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()

        if has_time:
            fig.autofmt_xdate()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


# ==============================================================================
# 5. Standalone CLI
# ==============================================================================

def build_cli():
    parser = argparse.ArgumentParser(description="CLARE Object-Oriented Visualization Suite")
    subparsers = parser.add_subparsers(dest="command", help="Visualization subcommands")

    # Subcommand: loss
    loss_parser = subparsers.add_parser("loss", help="Plot training/val/test loss curve")
    loss_parser.add_argument("--history-json", type=str, help="Path to training history JSON file")
    loss_parser.add_argument("--output", type=str, default="checkpoints/loss_curve.png", help="Output image file path")

    # Subcommand: sandwiched
    sand_parser = subparsers.add_parser("sandwiched", help="Plot sandwiched test block performance")
    sand_parser.add_argument("--data-file", type=str, help="CSV/Parquet file containing time and predictions")
    sand_parser.add_argument("--output", type=str, default="checkpoints/sandwiched_block.png", help="Output image path")
    sand_parser.add_argument("--block-size", type=int, default=200, help="Minimum block size")
    sand_parser.add_argument("--seed", type=int, default=42, help="Random seed for block extraction")

    # Subcommand: demo
    demo_parser = subparsers.add_parser("demo", help="Generate demo plots with synthetic data")
    demo_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")

    return parser


def run_demo(output_dir: str):
    print(f"Generating demo visualizations in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Demo Training Loss Plot
    steps_per_epoch = 100
    num_epochs = 5
    total_steps = steps_per_epoch * num_epochs

    vis_loss = TrainingLossVisualizer()
    vis_loss.set_steps_per_epoch(steps_per_epoch)

    step_loss = 2.5
    for s in range(1, total_steps + 1):
        if (s - 1) % steps_per_epoch == 0:
            vis_loss.add_epoch_start(s // steps_per_epoch, s)
        step_loss = max(0.4, step_loss * 0.995 + np.random.normal(0, 0.02))
        vis_loss.add_train_step(s, step_loss)

        if s % 33 == 0:
            val_loss = step_loss * 1.08 + np.random.normal(0, 0.03)
            test_loss = step_loss * 1.12 + np.random.normal(0, 0.03)
            vis_loss.add_val_loss(s, val_loss)
            vis_loss.add_test_loss(s, test_loss)

    vis_loss.generate_plot(save_path=os.path.join(output_dir, "demo_loss_curve.png"))

    # 2. Demo Sandwiched Block Plot
    n_pts = 600
    dates = pd.date_range("1991-01-31", periods=n_pts, freq="10min")
    t1_mask = np.zeros(n_pts, dtype=bool)
    t1_mask[:200] = True
    test_mask = np.zeros(n_pts, dtype=bool)
    test_mask[200:400] = True
    t2_mask = np.zeros(n_pts, dtype=bool)
    t2_mask[400:] = True

    true_te1 = 2000 + 800 * np.sin(np.linspace(0, 4 * np.pi, n_pts)) + np.random.normal(0, 100, n_pts)
    pred_te1 = true_te1.copy()
    pred_te1[test_mask] += np.random.normal(80, 150, np.sum(test_mask))  # unseen error

    df_demo = pd.DataFrame({
        "DateTimeFormatted": dates,
        "Te1": np.maximum(500, true_te1),
        "Te1_pred": np.maximum(500, pred_te1)
    })

    vis_sand = SandwichedBlockVisualizer()
    vis_sand.generate_plot(
        df_demo, t1_mask, test_mask, t2_mask,
        save_path=os.path.join(output_dir, "demo_sandwiched_block.png")
    )
    print("Demo visualization generation completed successfully!")


def main():  # pragma: no cover
    parser = build_cli()
    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.output_dir)
    elif args.command == "loss":
        if not args.history_json or not os.path.exists(args.history_json):
            print(f"Error: History file '{args.history_json}' not found.")
            sys.exit(1)
        with open(args.history_json, "r") as f:
            hist = json.load(f)
        vis = TrainingLossVisualizer(hist)
        vis.generate_plot(save_path=args.output)
    elif args.command == "sandwiched":
        if not args.data_file or not os.path.exists(args.data_file):
            print(f"Error: Data file '{args.data_file}' not found.")
            sys.exit(1)
        df = pd.read_parquet(args.data_file) if args.data_file.endswith(".parquet") else pd.read_csv(args.data_file)
        slice_df, meta = SandwichedBlockVisualizer.extract_sandwiched_slice(df, min_block_size=args.block_size, random_state=args.seed)
        n = len(slice_df)
        t1_len, t_len, t2_len = meta["train1_len"], meta["test_len"], meta["train2_len"]
        m1 = np.zeros(n, dtype=bool); m1[:t1_len] = True
        m2 = np.zeros(n, dtype=bool); m2[t1_len:t1_len+t_len] = True
        m3 = np.zeros(n, dtype=bool); m3[t1_len+t_len:] = True
        vis = SandwichedBlockVisualizer()
        vis.generate_plot(slice_df, m1, m2, m3, save_path=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
