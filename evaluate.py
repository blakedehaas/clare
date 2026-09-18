"""Evaluate the canonical CLARE and Continuous models on validation and storm test sets."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score

import constants
import models.feed_forward as models

BASE_MODEL_NAME = "2"
TE_MIN_K = 0.0
TE_BIN_WIDTH_K = 100.0
TE_NUM_CLASSES = 150
TE_MAX_K_EXCLUSIVE = 15000.0
SOFT_TARGET_SIGMA_K = 100.0
CENTRAL_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)

RAW_INPUT_COLUMNS = [
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
INPUT_COLUMNS = [
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
GROUP_COLUMNS = {
    "AL_index": [f"AL_index_{i}" for i in range(31)],
    "SYM_H": [f"SYM_H_{i}" for i in range(145)],
    "f107_index": [f"f107_index_{i}" for i in range(4)],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SPLIT_SEED", 0)))
    parser.add_argument("--block-minutes", type=int, default=int(os.environ.get("BLOCK_MINUTES", 212)))
    parser.add_argument("--train-seed", type=int, default=int(os.environ.get("TRAIN_SEED", 0)))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_outputs/paper_statistics.json"))
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_state_dict(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return state


def model_parameter_count(state_dict):
    return int(sum(value.numel() for value in state_dict.values() if torch.is_tensor(value)))


def prepare_features(batch, means, stds):
    values = {key: np.asarray(value) for key, value in batch.items()}
    missing = set(RAW_INPUT_COLUMNS + ["Te1"]) - set(values)
    if missing:
        raise ValueError(f"Evaluation batch is missing columns: {sorted(missing)}")

    for column, function in constants.NORMALIZATIONS.items():
        values[column] = function(values[column])
    for column, function in constants.CIRCULAR_ENCODINGS.items():
        values.update(function(values[column]))
    for group_name, columns in GROUP_COLUMNS.items():
        for column in columns:
            values[column] = (
                (np.asarray(values[column], dtype=np.float32) - means[group_name])
                / stds[group_name]
            ).astype(np.float32)

    x = np.column_stack([values[column] for column in INPUT_COLUMNS]).astype(np.float32)
    y = np.asarray(values["Te1"], dtype=np.float32)
    if x.shape[1] != len(INPUT_COLUMNS) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid evaluation inputs after preprocessing")
    return x, y


def point_metrics(predicted, observed):
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_k": float(np.sqrt(np.mean(error**2))),
        "mae_k": float(np.mean(np.abs(error))),
        "bias_k": float(np.mean(error)),
        "r2": float(r2_score(observed, predicted)),
        "pearson_r": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if np.std(observed) > 0 and np.std(predicted) > 0
            else None
        ),
        "within_10pct": float(np.mean(np.abs(error) <= 0.10 * np.abs(observed))),
    }


def evaluate_continuous(model, ds, means, stds, batch_size, device):
    predictions = []
    observations = []
    with torch.no_grad():
        for batch in ds.iter(batch_size=batch_size):
            x, y = prepare_features(batch, means, stds)
            output = model(torch.from_numpy(x).to(device)).squeeze(-1)
            predictions.append(output.cpu().numpy())
            observations.append(y)
    predicted = np.concatenate(predictions)
    observed = np.concatenate(observations)
    metrics = point_metrics(predicted, observed)
    metrics["mse_k2"] = float(metrics["rmse_k"] ** 2)
    return metrics


def evaluate_clare(model, ds, means, stds, batch_size, device):
    centers = torch.arange(TE_NUM_CLASSES, device=device, dtype=torch.float32) * TE_BIN_WIDTH_K + 50.0
    predictions = []
    observations = []
    hard_predictions = []
    hard_targets = []

    soft_ce_sum = 0.0
    crps_sum = 0.0
    entropy_sum = 0.0
    predictive_std_sum = 0.0
    max_probability_sum = 0.0
    coverage_counts = {level: 0 for level in CENTRAL_COVERAGE_LEVELS}
    total = 0

    with torch.no_grad():
        for batch in ds.iter(batch_size=batch_size):
            x, y_np = prepare_features(batch, means, stds)
            y = torch.from_numpy(y_np).to(device)
            logits = model(torch.from_numpy(x).to(device))
            probabilities = torch.softmax(logits, dim=1)
            log_probabilities = torch.log_softmax(logits, dim=1)

            prediction = probabilities @ centers
            hard_prediction = probabilities.argmax(dim=1)
            hard_target = torch.floor((y - TE_MIN_K) / TE_BIN_WIDTH_K).long()
            if (hard_target < 0).any() or (hard_target >= TE_NUM_CLASSES).any():
                raise ValueError("Te1 is outside the fixed CLARE bin range [0, 15000) K")

            soft_targets = torch.softmax(
                -0.5 * ((centers.unsqueeze(0) - y.unsqueeze(1)) / SOFT_TARGET_SIGMA_K) ** 2,
                dim=1,
            )
            soft_ce = -(soft_targets * log_probabilities).sum(dim=1)

            cdf = probabilities.cumsum(dim=1)
            indicator = (centers.unsqueeze(0) >= y.unsqueeze(1)).to(probabilities.dtype)
            crps = TE_BIN_WIDTH_K * torch.square(cdf - indicator).sum(dim=1)

            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
            variance = (probabilities * torch.square(centers.unsqueeze(0) - prediction.unsqueeze(1))).sum(dim=1)
            predictive_std = torch.sqrt(variance.clamp_min(0.0))
            max_probability = probabilities.max(dim=1).values

            for level in CENTRAL_COVERAGE_LEVELS:
                tail = (1.0 - level) / 2.0
                lower_index = (cdf >= tail).to(torch.int64).argmax(dim=1)
                upper_index = (cdf >= 1.0 - tail).to(torch.int64).argmax(dim=1)
                lower_edge = TE_MIN_K + lower_index.to(torch.float32) * TE_BIN_WIDTH_K
                upper_edge = TE_MIN_K + (upper_index.to(torch.float32) + 1.0) * TE_BIN_WIDTH_K
                coverage_counts[level] += int(((y >= lower_edge) & (y <= upper_edge)).sum().item())

            n = y.shape[0]
            total += n
            soft_ce_sum += float(soft_ce.sum().item())
            crps_sum += float(crps.sum().item())
            entropy_sum += float(entropy.sum().item())
            predictive_std_sum += float(predictive_std.sum().item())
            max_probability_sum += float(max_probability.sum().item())

            predictions.append(prediction.cpu().numpy())
            observations.append(y_np)
            hard_predictions.append(hard_prediction.cpu().numpy())
            hard_targets.append(hard_target.cpu().numpy())

    predicted = np.concatenate(predictions)
    observed = np.concatenate(observations)
    hard_predictions = np.concatenate(hard_predictions)
    hard_targets = np.concatenate(hard_targets)

    metrics = point_metrics(predicted, observed)
    metrics.update(
        {
            "soft_target_cross_entropy": soft_ce_sum / total,
            "hard_bin_accuracy": float(accuracy_score(hard_targets, hard_predictions)),
            "macro_f1": float(f1_score(hard_targets, hard_predictions, average="macro", zero_division=0)),
            "crps_k": crps_sum / total,
            "mean_predictive_entropy_nats": entropy_sum / total,
            "mean_predictive_std_k": predictive_std_sum / total,
            "mean_max_bin_probability": max_probability_sum / total,
            "central_interval_coverage": {
                f"{int(level * 100)}pct": coverage_counts[level] / total
                for level in CENTRAL_COVERAGE_LEVELS
            },
        }
    )
    return metrics


def timestamp_string(value):
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def dataset_context(dataset_dir, block_minutes, split_seed, val_ds, storm_ds):
    summary_path = dataset_dir / f"split_summary_{block_minutes}m_s{split_seed}.csv"
    assignment_path = dataset_dir / f"block_assignment_{block_minutes}m_s{split_seed}.csv"
    summary = pd.read_csv(summary_path)
    assignment = pd.read_csv(assignment_path)

    expected_rows = {"val": len(val_ds), "test-storm": len(storm_ds)}
    summary_index = summary.set_index("split")
    for name, count in expected_rows.items():
        if int(summary_index.loc[name, "n_samples"]) != count:
            raise ValueError(f"Split summary count mismatch for {name}")

    splits = {}
    for _, row in summary.iterrows():
        splits[str(row["split"])] = {
            "n_samples": int(row["n_samples"]),
            "n_blocks": int(row["n_blocks"]),
            "time_start": timestamp_string(row["time_start"]),
            "time_end": timestamp_string(row["time_end"]),
        }

    return {
        "dataset_dir": str(dataset_dir),
        "block_minutes": int(block_minutes),
        "split_seed": int(split_seed),
        "full_dataset_definition": "all filtered rows after NaN removal, before storm holdout and embargo removal",
        "splits": splits,
        "development": {
            "n_samples": int(splits["train"]["n_samples"] + splits["val"]["n_samples"]),
            "n_blocks": int(len(assignment)),
            "train_fraction": float(splits["train"]["n_samples"] / (splits["train"]["n_samples"] + splits["val"]["n_samples"])),
        },
        "storm_window": {
            "start_inclusive": timestamp_string(summary["storm_start"].iloc[0]),
            "end_exclusive": timestamp_string(summary["storm_end_exclusive"].iloc[0]),
        },
        "post_storm_embargo_end_exclusive": timestamp_string(summary["post_storm_embargo_end_exclusive"].iloc[0]),
    }


def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = args.dataset_root / f"processed_dataset_blocksplit_{args.block_minutes}m_s{args.seed}"
    val_ds = datasets.Dataset.load_from_disk(str(dataset_dir / "val-blocks"))
    storm_ds = datasets.Dataset.load_from_disk(str(dataset_dir / "test-storm"))

    stats_path = args.checkpoints_dir / f"{BASE_MODEL_NAME}_{args.block_minutes}m_s{args.seed}_norm_stats.json"
    stats = load_json(stats_path)
    means = stats["mean"]
    stds = stats["std"]

    model_prefix = f"{BASE_MODEL_NAME}_{args.block_minutes}m_s{args.seed}_ts{args.train_seed}"
    specs = {
        "CLARE": {
            "checkpoint": args.checkpoints_dir / f"{model_prefix}.pth",
            "metadata": args.checkpoints_dir / f"{model_prefix}_metadata.json",
            "output_dim": TE_NUM_CLASSES,
        },
        "Continuous": {
            "checkpoint": args.checkpoints_dir / f"{model_prefix}_continuous.pth",
            "metadata": args.checkpoints_dir / f"{model_prefix}_continuous_metadata.json",
            "output_dim": 1,
        },
    }

    models_loaded = {}
    model_context = {}
    for label, spec in specs.items():
        state = load_state_dict(spec["checkpoint"])
        model = models.FeedForwardNetwork(len(INPUT_COLUMNS), 2048, spec["output_dim"])
        model.load_state_dict(state)
        model.to(device).eval()
        metadata = load_json(spec["metadata"])
        if int(metadata.get("input_size", len(INPUT_COLUMNS))) != len(INPUT_COLUMNS):
            raise ValueError(f"{label} metadata input size does not match evaluator")
        models_loaded[label] = model
        model_context[label] = {
            "checkpoint": str(spec["checkpoint"]),
            "metadata": str(spec["metadata"]),
            "normalization_stats": str(stats_path),
            "parameter_count": model_parameter_count(state),
            "selected_step": metadata.get("total_steps"),
            "selected_epoch": metadata.get("epoch"),
            "selection_criterion": metadata.get("selection_criterion"),
            "mode": metadata.get("mode"),
        }

    split_datasets = {"validation": val_ds, "test-storm": storm_ds}
    results = {}
    for split_name, ds in split_datasets.items():
        print(f"Evaluating {split_name}: {len(ds):,} rows")
        results[split_name] = {
            "CLARE": evaluate_clare(models_loaded["CLARE"], ds, means, stds, args.batch_size, device),
            "Continuous": evaluate_continuous(models_loaded["Continuous"], ds, means, stds, args.batch_size, device),
        }

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": "Canonical paper evaluation statistics for CLARE and Continuous",
        "dataset": dataset_context(dataset_dir, args.block_minutes, args.seed, val_ds, storm_ds),
        "target": {
            "quantity": "electron temperature",
            "units": "K",
            "range_k": [TE_MIN_K, TE_MAX_K_EXCLUSIVE],
            "clare_bins": TE_NUM_CLASSES,
            "clare_bin_width_k": TE_BIN_WIDTH_K,
            "clare_soft_target_sigma_k": SOFT_TARGET_SIGMA_K,
            "clare_scalar_decoder": "softmax-weighted expected bin center",
        },
        "training_seed": int(args.train_seed),
        "models": model_context,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(clean_for_json(output), handle, indent=2, allow_nan=False)

    print(f"Saved paper statistics to {args.output}")
    for split_name, split_results in results.items():
        print(f"\n{split_name}")
        for label, metrics in split_results.items():
            print(
                f"  {label}: RMSE={metrics['rmse_k']:.3f} K, "
                f"MAE={metrics['mae_k']:.3f} K, "
                f"R2={metrics['r2']:.5f}, "
                f"within10={100.0 * metrics['within_10pct']:.3f}%"
            )


if __name__ == "__main__":
    main()
