"""Report Kutiev (2002) metrics on full and paper-supported storm cohorts."""

import argparse
import json

import numpy as np
from datasets import Dataset

from baselines.kutiev_2002 import evaluate_kutiev_metrics, predict_kutiev_2002


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Hugging Face test-storm dataset directory")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    dataset = Dataset.load_from_disk(args.dataset)
    altitude = np.asarray(dataset["Altitude"], dtype=float)
    glat = np.asarray(dataset["GLAT"], dtype=float)
    gmlt = np.asarray(dataset["GMLT"], dtype=float)
    observed = np.asarray(dataset["Te1"], dtype=float)

    supported = predict_kutiev_2002(altitude, glat, gmlt)
    extrapolated = predict_kutiev_2002(altitude, glat, gmlt, extrapolate=True)
    report = {
        "dataset_rows": len(dataset),
        "full_test_storm_extrapolated": evaluate_kutiev_metrics(observed, extrapolated),
        "paper_supported_subset": evaluate_kutiev_metrics(observed, supported),
        "warning": (
            "The full-set result is a sensitivity analysis: it extrapolates beyond 6370 km "
            "and L=3 and assigns transition hours to the nearer published day/night sector. "
            "Two samples above L=28 contribute 99.994% of its squared error."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
