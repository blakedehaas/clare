"""
generate_full_visualization_suite.py - Generates the complete publication visualization suite for TEMPEST.

Outputs high-resolution (300 DPI) publication figures to paper/figures/:
1. epoch_loss_curve.png: Training, validation, and test loss progression with best model callout.
2. epoch_test_loss_curve.png: Test loss progression highlighting best checkpoint.
3. sandwiched_mean_all_blocks.png: Aggregate performance across all 334 contiguous sandwiched test blocks (mean ± 1σ).
4. sandwiched_median.png: Median performing sandwiched test block.
5. sandwiched_best.png: Best performing sandwiched test block.
6. sandwiched_worst.png: Worst performing sandwiched test block.
7. sandwiched_random.png: Representative random sandwiched test block.
8. test-normal_deviation_plot.png: 2D hexbin residual density vs observed Te, running mean & std profile.
9. test-normal_plot.png: Observed vs predicted scatter plot on unseen test-normal with 1:1 line.
10. plasmapause_transition.png: Plasmapause transition analysis (inside vs outside L_pp).
11. test_storm_timeseries.png: Unseen February 1991 severe storm continuous tracking.
12. kutiev_comparison.png: Side-by-side scatter & residual benchmark against Kutiev et al. (2002).
13. tempest_shap_global_importance.png: Top 20 physical drivers ranked by mean absolute Path-Shapley values.
14. tempest_shap_regime_comparison.png: Quiet-time vs storm-time comparative feature attributions.
15. tempest_shap_beeswarm.png: Beeswarm distribution of feature attributions.
16. tempest_shap_waterfall.png: Step-by-step waterfall attribution for an individual storm observation.
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn.functional as F
import datasets
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from train_v2 import TEMPEST, ALL_INPUT_COLUMNS, NUM_INPUT_FEATURES, build_preprocessor
from baselines.kutiev_2002 import predict_kutiev_2002, evaluate_kutiev_metrics
from explainability import (
    TempestExplainer,
    plot_global_feature_importance,
    plot_beeswarm_summary,
    plot_regime_comparison,
    plot_waterfall
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_core_checkpoints_figures(output_dir: str):
    """Copies pre-computed diagnostic figures from checkpoints/ to paper/figures/."""
    targets = [
        ("epoch_loss_curve.png", "checkpoints/tempest_epoch_loss_curve.png"),
        ("epoch_test_loss_curve.png", "checkpoints/tempest_epoch_test_loss_curve.png"),
        ("sandwiched_mean_all_blocks.png", "checkpoints/tempest_sandwiched_mean_all_blocks.png"),
        ("sandwiched_median.png", "checkpoints/tempest_sandwiched_median.png"),
        ("sandwiched_best.png", "checkpoints/tempest_sandwiched_best.png"),
        ("sandwiched_worst.png", "checkpoints/tempest_sandwiched_worst.png"),
        ("sandwiched_random.png", "checkpoints/tempest_sandwiched_random.png"),
        ("test-normal_deviation_plot.png", "checkpoints/tempest_test-normal_deviation_plot.png"),
        ("test-normal_plot.png", "checkpoints/tempest_test-normal_plot.png"),
        ("plasmapause_transition.png", "checkpoints/tempest_plasmapause_transition.png"),
    ]
    for target_name, src in targets:
        dst = os.path.join(output_dir, target_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"[Copied] {src} -> {dst}")
        else:
            print(f"[Warning] Source figure {src} not found!")


def load_tempest_model(device: torch.device) -> TEMPEST:
    """Loads the trained TEMPEST checkpoint."""
    model = TEMPEST(
        num_features=NUM_INPUT_FEATURES,
        d_model=256,
        expert_dim=512,
        num_experts=8,
        top_k=2,
        n_layers=4,
        n_hc=4,
        vocab_size=150
    ).to(device)
    ckpt_path = "checkpoints/tempest_best.pth"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state if "state_dict" not in state else state["state_dict"], strict=False)
        print(f"Loaded TEMPEST model weights from {ckpt_path}")
    else:
        print(f"[Warning] {ckpt_path} not found, using initialized weights!")
    model.eval()
    return model


def generate_kutiev_comparison_figure(
    model: TEMPEST,
    device: torch.device,
    means: dict,
    stds: dict,
    output_path: str
):
    """Generates a side-by-side scatter and residual benchmark against Kutiev et al. (2002)."""
    print("Generating Kutiev et al. (2002) comparison figure...")
    val_raw = datasets.Dataset.load_from_disk("dataset/processed_dataset_01_31_storm/test-normal")
    sub_raw = val_raw.select(range(min(6000, len(val_raw))))

    preprocessor = build_preprocessor(means, stds)
    sub_ds = sub_raw.map(preprocessor, batched=True, batch_size=2000, remove_columns=sub_raw.column_names)
    sub_ds.set_format("torch")

    x_tensor = torch.tensor(np.array(sub_ds["input_ids"]), dtype=torch.float32, device=device)
    y_true_k = np.array(sub_ds["label"], dtype=np.float32) * 100.0 + 50.0

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(x_tensor)
            probs = F.softmax(logits.float(), dim=-1)
            pred_tempest_k = (probs * model.bin_centers.unsqueeze(0)).sum(dim=-1).cpu().numpy()

    # Kutiev et al. (2002) predictions
    alt = np.array(sub_raw["Altitude"])
    ilat = np.array(sub_raw["ILAT"])
    gmlt = np.array(sub_raw["GMLT"])
    kp = np.array(sub_raw["Kp_index"])

    pred_kutiev_k = predict_kutiev_2002(altitude=alt, ilat=ilat, gmlt=gmlt, kp=kp)

    m_tempest = {
        "r2": r2_score(y_true_k, pred_tempest_k),
        "rmse": np.sqrt(mean_squared_error(y_true_k, pred_tempest_k)),
        "mae": mean_absolute_error(y_true_k, pred_tempest_k),
        "acc10": np.mean(np.abs(pred_tempest_k - y_true_k) <= 0.10 * y_true_k) * 100.0
    }
    m_kutiev = evaluate_kutiev_metrics(y_true_k, pred_kutiev_k)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    lims = [500, 12000]

    # Panel 1: TEMPEST Scatter
    hb1 = ax1.hexbin(y_true_k, pred_tempest_k, gridsize=50, cmap="Blues", mincnt=1, bins="log")
    ax1.plot(lims, lims, color="black", linestyle="--", linewidth=1.5, label="1:1 Reference")
    ax1.plot(lims, [l * 1.10 for l in lims], color="gray", linestyle=":", label="±10% Error Bounds")
    ax1.plot(lims, [l * 0.90 for l in lims], color="gray", linestyle=":")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(
        f"(a) TEMPEST (DeepSeek-V4 MoE)\n$R^2 = {m_tempest['r2']:.3f}$ | RMSE = {m_tempest['rmse']:.1f} K | MAE = {m_tempest['mae']:.1f} K | Acc(±10%) = {m_tempest['acc10']:.1f}%",
        fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Observed Electron Temperature $T_e$ [K]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted $T_e$ [K]", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    cb1 = fig.colorbar(hb1, ax=ax1)
    cb1.set_label("Log10 Count", fontsize=10)
    ax1.legend(loc="upper left")

    # Panel 2: Kutiev et al. (2002) Scatter
    hb2 = ax2.hexbin(y_true_k, pred_kutiev_k, gridsize=50, cmap="Reds", mincnt=1, bins="log")
    ax2.plot(lims, lims, color="black", linestyle="--", linewidth=1.5, label="1:1 Reference")
    ax2.plot(lims, [l * 1.10 for l in lims], color="gray", linestyle=":", label="±10% Error Bounds")
    ax2.plot(lims, [l * 0.90 for l in lims], color="gray", linestyle=":")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title(
        f"(b) Kutiev et al. (2002) Empirical Akebono Model\n$R^2 = {m_kutiev['r2']:.3f}$ | RMSE = {m_kutiev['rmse']:.1f} K | MAE = {m_kutiev['mae']:.1f} K | Acc(±10%) = {m_kutiev['acc_10']:.1f}%",
        fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Observed Electron Temperature $T_e$ [K]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted $T_e$ [K]", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    cb2 = fig.colorbar(hb2, ax=ax2)
    cb2.set_label("Log10 Count", fontsize=10)
    ax2.legend(loc="upper left")

    # Panel 3: Residual Distributions
    res_tempest = pred_tempest_k - y_true_k
    res_kutiev = pred_kutiev_k - y_true_k
    bins_res = np.linspace(-3000, 3000, 61)
    ax3.hist(res_tempest, bins=bins_res, density=True, alpha=0.6, color="#1F77B4", label=f"TEMPEST (Std: {np.std(res_tempest):.1f} K)")
    ax3.hist(res_kutiev, bins=bins_res, density=True, alpha=0.5, color="#D62728", label=f"Kutiev (2002) (Std: {np.std(res_kutiev):.1f} K)")
    ax3.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax3.set_title("(c) Residual Error Probability Densities", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Residual Error ($T_{e,\\text{pred}} - T_{e,\\text{obs}}$) [K]", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Panel 4: Error vs Altitude Profile
    alt_bins = np.linspace(1000, 9000, 17)
    alt_centers = 0.5 * (alt_bins[:-1] + alt_bins[1:])
    alt_idx = np.digitize(alt, alt_bins) - 1

    tempest_rmse_alt = [np.sqrt(mean_squared_error(y_true_k[alt_idx == b], pred_tempest_k[alt_idx == b])) if np.sum(alt_idx == b) > 10 else np.nan for b in range(len(alt_centers))]
    kutiev_rmse_alt = [np.sqrt(mean_squared_error(y_true_k[alt_idx == b], pred_kutiev_k[alt_idx == b])) if np.sum(alt_idx == b) > 10 else np.nan for b in range(len(alt_centers))]

    ax4.plot(alt_centers, tempest_rmse_alt, marker="o", linewidth=2.2, color="#1F77B4", label="TEMPEST RMSE")
    ax4.plot(alt_centers, kutiev_rmse_alt, marker="s", linewidth=2.0, linestyle="--", color="#D62728", label="Kutiev (2002) RMSE")
    ax4.set_title("(d) Altitude-Stratified Root Mean Square Error", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Spacecraft Altitude [km]", fontsize=11, fontweight="bold")
    ax4.set_ylabel("RMSE [K]", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Kutiev Comparison] Saved to: {output_path}")


def generate_storm_timeseries_figure(
    model: TEMPEST,
    device: torch.device,
    means: dict,
    stds: dict,
    output_path: str
):
    """Generates continuous time-series tracking of the unseen February 1991 severe storm."""
    print("Generating February 1991 severe storm tracking figure...")
    storm_raw = datasets.Dataset.load_from_disk("dataset/processed_dataset_01_31_storm/test-storm")
    sub_raw = storm_raw.select(range(min(1200, len(storm_raw))))

    preprocessor = build_preprocessor(means, stds)
    storm_ds = sub_raw.map(preprocessor, batched=True, batch_size=2000, remove_columns=storm_raw.column_names)
    storm_ds.set_format("torch")

    x_tensor = torch.tensor(np.array(storm_ds["input_ids"]), dtype=torch.float32, device=device)
    y_true_k = np.array(storm_ds["label"], dtype=np.float32) * 100.0 + 50.0

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(x_tensor)
            probs = F.softmax(logits.float(), dim=-1)
            pred_k = (probs * model.bin_centers.unsqueeze(0)).sum(dim=-1).cpu().numpy()

    # Extract geomagnetic indices
    sym_h = np.array(sub_raw["SYM_H_0"])
    al = np.array(sub_raw["AL_index_0"])
    time_idx = np.arange(len(y_true_k))

    fig, (ax_te, ax_res, ax_sym, ax_al) = plt.subplots(4, 1, figsize=(16, 12), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2, 1.0, 1.0]})

    # Panel 1: Observed vs Predicted Te
    ax_te.plot(time_idx, y_true_k, label="Observed $T_e$ (Akebono TED)", color="#1F77B4", linewidth=1.6)
    ax_te.plot(time_idx, pred_k, label="TEMPEST Predicted $\\mathbb{E}[T_e]$", color="#D62728", linewidth=1.8, linestyle="--")
    ax_te.set_ylabel("Electron Temp ($T_e$) [K]", fontsize=11, fontweight="bold")
    ax_te.set_title("TEMPEST Severe Storm Tracking: Unseen February 1991 Benchmark ($SYM\\text{-}H < -200\\text{ nT}$)", fontsize=14, fontweight="bold", pad=10)
    ax_te.grid(True, linestyle="--", alpha=0.4)
    ax_te.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Residual Error
    residuals = pred_k - y_true_k
    ax_res.plot(time_idx, residuals, color="#8B0000", linewidth=1.2, label="Residual ($T_{e,\\text{pred}} - T_{e,\\text{obs}}$)")
    ax_res.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax_res.set_ylabel("Residual [K]", fontsize=11, fontweight="bold")
    ax_res.grid(True, linestyle="--", alpha=0.4)
    ax_res.legend(loc="upper right", framealpha=0.9)

    # Panel 3: SYM-H Index
    ax_sym.plot(time_idx, sym_h, color="#2CA02C", linewidth=1.5, label="SYM-H Index (Ring Current)")
    ax_sym.axhline(-100, color="orange", linestyle="--", alpha=0.7, label="Strong Storm (-100 nT)")
    ax_sym.axhline(-200, color="red", linestyle="--", alpha=0.7, label="Severe Storm (-200 nT)")
    ax_sym.set_ylabel("SYM-H [nT]", fontsize=11, fontweight="bold")
    ax_sym.grid(True, linestyle="--", alpha=0.4)
    ax_sym.legend(loc="lower left", framealpha=0.9)

    # Panel 4: AL Index
    ax_al.plot(time_idx, al, color="#9467BD", linewidth=1.2, label="AL Index (Auroral Electrojet)")
    ax_al.set_ylabel("AL [nT]", fontsize=11, fontweight="bold")
    ax_al.set_xlabel("Orbital Sequential Sample Index (February 1991 Passes)", fontsize=11, fontweight="bold")
    ax_al.grid(True, linestyle="--", alpha=0.4)
    ax_al.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Storm Timeseries] Saved to: {output_path}")


def generate_shap_suite_figures(
    model: TEMPEST,
    device: torch.device,
    means: dict,
    stds: dict,
    output_dir: str
):
    """Generates the full SHAP explainability suite using TempestExplainer."""
    print("Generating SHAP Explainability suite figures...")
    explainer = TempestExplainer(model, device=device)

    val_raw = datasets.Dataset.load_from_disk("dataset/processed_dataset_01_31_storm/test-normal")
    preprocessor = build_preprocessor(means, stds)
    val_ds = val_raw.select(range(200)).map(preprocessor, batched=True, batch_size=200, remove_columns=val_raw.column_names)
    val_ds.set_format("torch")
    x_val = torch.tensor(np.array(val_ds["input_ids"]), dtype=torch.float32, device=device)
    storm_raw = datasets.Dataset.load_from_disk("dataset/processed_dataset_01_31_storm/test-storm")
    storm_ds = storm_raw.select(range(200)).map(preprocessor, batched=True, batch_size=200, remove_columns=storm_raw.column_names)
    storm_ds.set_format("torch")
    x_storm = torch.tensor(np.array(storm_ds["input_ids"]), dtype=torch.float32, device=device)

    # Baseline quiet state
    baseline_vector = x_val.mean(dim=0, keepdim=True)

    # Compute SHAP for a sample of validation points
    sample_eval = x_val[:50]
    sample_eval_np = sample_eval.cpu().numpy()
    res_val = explainer.explain(sample_eval, background_samples=x_val.mean(dim=0, keepdim=True), steps=20)
    shap_vals = res_val["shap_values"]

    feature_names = ALL_INPUT_COLUMNS

    # 1. Global Feature Importance
    p_global = os.path.join(output_dir, "tempest_shap_global_importance.png")
    plot_global_feature_importance(shap_vals, feature_names=feature_names, top_n=20, save_path=p_global)

    # 2. Beeswarm Summary Plot
    p_bee = os.path.join(output_dir, "tempest_shap_beeswarm.png")
    plot_beeswarm_summary(shap_vals, sample_eval_np, feature_names=feature_names, top_n=15, save_path=p_bee)

    # 3. Regime Comparison (Inside vs Outside Plasmapause)
    p_regime = os.path.join(output_dir, "tempest_shap_regime_comparison.png")
    plot_regime_comparison(shap_vals, sample_eval_np, feature_names=feature_names, top_n=12, save_path=p_regime)

    # 4. Waterfall Plot for a Storm Observation
    storm_eval = x_storm[:20]
    res_storm = explainer.explain(storm_eval, background_samples=x_val.mean(dim=0, keepdim=True), steps=20)
    idx_case = 5
    shap_sample = res_storm["shap_values"][idx_case]
    base_val = res_storm["base_value"]
    pred_val = res_storm["predictions"][idx_case]
    feat_vals = storm_eval[idx_case].cpu().numpy()

    p_waterfall = os.path.join(output_dir, "tempest_shap_waterfall.png")
    plot_waterfall(
        shap_sample=shap_sample,
        base_val=base_val,
        pred_val=pred_val,
        feature_vals=feat_vals,
        feature_names=feature_names,
        top_n=12,
        title="TEMPEST Local Waterfall Attribution (Severe Storm Observation)",
        save_path=p_waterfall
    )



def main():
    output_dir = "paper/figures"
    ensure_dir(output_dir)

    print("=== Copying Existing Core Diagnostic Visualizations ===")
    copy_core_checkpoints_figures(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on compute device: {device}")
    model = load_tempest_model(device)

    with open("checkpoints/norm_stats.json", "r") as f:
        stats = json.load(f)
    means, stds = stats["mean"], stats["std"]

    print("\n=== Generating Kutiev et al. (2002) Comparison Figure ===")
    generate_kutiev_comparison_figure(
        model, device, means, stds,
        os.path.join(output_dir, "kutiev_comparison.png")
    )

    print("\n=== Generating Unseen February 1991 Storm Tracking Figure ===")
    generate_storm_timeseries_figure(
        model, device, means, stds,
        os.path.join(output_dir, "test_storm_timeseries.png")
    )

    print("\n=== Generating SHAP Explainability Suite Figures ===")
    generate_shap_suite_figures(model, device, means, stds, output_dir)

    print("\n[SUCCESS] Full TEMPEST visualization suite generated in 'paper/figures/'!")


if __name__ == "__main__":
    main()
