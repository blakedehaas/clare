"""Evaluate the Kutiev et al. (2002) baseline on a CLARE test dataset."""

import argparse
import csv
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


def _comparison_mask_and_predictions(dataset):
    """Return Kutiev predictions and the valid rows in its published domain."""

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
    if np.count_nonzero(comparison_mask) < 2:
        raise ValueError("Fewer than two rows fall inside the Kutiev applicability domain")
    return observed, predictions, comparison_mask


def evaluate_dataset(dataset, comparison_predictions=None, comparison_name="comparison_model"):
    """Evaluate Kutiev and, optionally, another model on exactly the same rows."""

    observed, predictions, comparison_mask = _comparison_mask_and_predictions(dataset)
    eligible_count = int(np.count_nonzero(comparison_mask))
    total_count = len(dataset)

    metrics = calculate_metrics(
        observed[comparison_mask],
        predictions.temperature_k[comparison_mask],
    )
    metrics.update(
        {
            "dataset_rows": total_count,
            "eligible_rows": eligible_count,
            "coverage_percent": float(eligible_count / total_count * 100.0),
            "scope": (
                "1000-10000 km for L < 2; 1000-6370 km for 2 <= L <= 3; "
                "|GLAT| <= 70 deg; MLT 09-16 or 22-04"
            ),
            "source_doi": "10.1029/2002JA009494",
        }
    )

    if comparison_predictions is not None:
        comparison = np.asarray(comparison_predictions, dtype=float)
        if comparison.shape != observed.shape:
            raise ValueError(
                "Comparison predictions must have one value per dataset row "
                f"(expected {observed.shape}, got {comparison.shape})"
            )
        paired_mask = comparison_mask & np.isfinite(comparison)
        if np.count_nonzero(paired_mask) < 2:
            raise ValueError("Fewer than two rows have paired finite predictions")
        metrics["paired_comparison"] = {
            "sample_count": int(np.count_nonzero(paired_mask)),
            "kutiev_2002": calculate_metrics(
                observed[paired_mask], predictions.temperature_k[paired_mask]
            ),
            comparison_name: calculate_metrics(observed[paired_mask], comparison[paired_mask]),
        }
    return metrics


def write_comparison_rows(
    output_path,
    dataset,
    comparison_predictions=None,
    comparison_name="comparison_model",
):
    """Export the exact aligned rows used for reproducible model comparison."""

    observed, predictions, comparison_mask = _comparison_mask_and_predictions(dataset)
    comparison = None
    if comparison_predictions is not None:
        comparison = np.asarray(comparison_predictions, dtype=float)
        if comparison.shape != observed.shape:
            raise ValueError(
                "Comparison predictions must have one value per dataset row "
                f"(expected {observed.shape}, got {comparison.shape})"
            )
        comparison_mask &= np.isfinite(comparison)

    datetimes = (
        dataset["DateTimeFormatted"]
        if "DateTimeFormatted" in dataset.column_names
        else None
    )
    fieldnames = ["dataset_index", "observed_te_k", "kutiev_2002_te_k", "kutiev_zone"]
    if datetimes is not None:
        fieldnames.insert(1, "datetime")
    if comparison is not None:
        fieldnames.append(f"{comparison_name}_te_k")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in np.flatnonzero(comparison_mask):
            row = {
                "dataset_index": int(index),
                "observed_te_k": float(observed[index]),
                "kutiev_2002_te_k": float(predictions.temperature_k[index]),
                "kutiev_zone": str(predictions.zone[index]),
            }
            if datetimes is not None:
                row["datetime"] = datetimes[index]
            if comparison is not None:
                row[f"{comparison_name}_te_k"] = float(comparison[index])
            writer.writerow(row)


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
    parser.add_argument(
        "--comparison-predictions",
        help="Optional .npy file containing one aligned prediction per dataset row",
    )
    parser.add_argument(
        "--comparison-name",
        default="comparison_model",
        help="Label used for the aligned comparison model in reports and row exports",
    )
    parser.add_argument(
        "--output-comparison-csv",
        help="Optional CSV containing the exact eligible rows and aligned predictions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import datasets

    dataset = datasets.load_from_disk(args.dataset)
    comparison_predictions = (
        np.load(args.comparison_predictions)
        if args.comparison_predictions
        else None
    )
    report = evaluate_dataset(dataset, comparison_predictions, args.comparison_name)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.output_comparison_csv:
        write_comparison_rows(
            args.output_comparison_csv,
            dataset,
            comparison_predictions,
            args.comparison_name,
        )


if __name__ == "__main__":
    main()
