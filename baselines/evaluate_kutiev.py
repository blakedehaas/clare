"""Evaluate the Kutiev et al. (2002) baseline on a CLARE test dataset."""

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.kutiev_2002 import predict_kutiev_2002


REQUIRED_COLUMNS = ("Altitude", "GLAT", "ILAT", "GMLT", "Te1")


def calculate_metrics(observed_k, predicted_k):
    """Calculate the three metrics reported for CLARE and its baselines."""

    observed = np.asarray(observed_k, dtype=float)
    predicted = np.asarray(predicted_k, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0.0)
    observed = observed[finite]
    predicted = predicted[finite]
    if observed.size < 2:
        raise ValueError("At least two finite, positive observations are required")

    relative_error = np.abs(predicted - observed) / observed
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    total_sum_squares = np.sum((observed - np.mean(observed)) ** 2)
    if total_sum_squares == 0.0:
        r2 = 1.0 if residual_sum_squares == 0.0 else 0.0
    else:
        r2 = 1.0 - residual_sum_squares / total_sum_squares
    return {
        "accuracy_within_10_percent": float(np.mean(relative_error <= 0.10) * 100.0),
        "r2": float(r2),
        "rmse_k": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "sample_count": int(observed.size),
    }


def evaluate_dataset(dataset):
    """Evaluate Kutiev only on rows inside its published applicability domain."""

    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    predictions = predict_kutiev_2002(
        dataset["Altitude"],
        dataset["GLAT"],
        dataset["ILAT"],
        dataset["GMLT"],
    )
    observed = np.asarray(dataset["Te1"], dtype=float)
    comparison_mask = predictions.eligible & np.isfinite(observed) & (observed > 0.0)
    eligible_count = int(np.count_nonzero(comparison_mask))
    total_count = len(dataset)
    if eligible_count < 2:
        raise ValueError("Fewer than two rows fall inside the Kutiev applicability domain")

    metrics = calculate_metrics(
        observed[comparison_mask],
        predictions.temperature_k[comparison_mask],
    )
    metrics.update(
        {
            "dataset_rows": total_count,
            "eligible_rows": eligible_count,
            "coverage_percent": float(eligible_count / total_count * 100.0),
            "scope": "1000-10000 km, |GLAT| <= 70 deg, L <= 3, MLT 09-16 or 22-04",
            "source_doi": "10.1029/2002JA009494",
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="dataset/processed_dataset/test-normal",
        help="Path to a Hugging Face Dataset saved with save_to_disk",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for a machine-readable metrics report",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import datasets

    dataset = datasets.load_from_disk(args.dataset)
    report = evaluate_dataset(dataset)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
