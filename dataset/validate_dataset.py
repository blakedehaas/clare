#!/usr/bin/env python3

"""
Dataset validation for configurable block-split experiments.

This validator:

1. Discovers all datasets matching:
       processed_dataset_blocksplit_<block_hours>h_s<seed>

2. Groups datasets by block size.

3. Performs:
       - Per-dataset validation
       - Cross-seed validation
       - Statistical validation
       - Block integrity validation
       - Storm integrity validation

4. Writes:
       dataset_validation/
           <block_size>h/
               summary.txt
               seed_<seed>/
                   validation_report.txt
                   timeline.png
                   year_coverage.png
                   cdf_<variable>.png

Every validation produces:
    PASS
    FAIL
    INFO

A final failure summary is included at the end of every report.
"""

import os
import re
import hashlib
from itertools import combinations

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


# ======================================================================================
# Configuration
# ======================================================================================

DATASET_PATTERN = re.compile(
    r"processed_dataset_blocksplit_(\d+)h_s(\d+)$"
)

VALIDATION_ROOT = "dataset_validation"

STORM_START = pd.Timestamp("1991-01-31")
STORM_END = pd.Timestamp("1991-02-07")

KEY_FEATURES = [
    "Te1",
    "Kp_index",
    "f107_index_0",
    "GMLT",
]

EXPECTED_TRAIN_FRAC = 0.8
EXPECTED_VAL_FRAC = 0.1
EXPECTED_TEST_FRAC = 0.1


os.makedirs(VALIDATION_ROOT, exist_ok=True)


# ======================================================================================
# Logging Helpers
# ======================================================================================

def log(msg):
    print(msg, flush=True)


# ======================================================================================
# Validation Result Collector
# ======================================================================================

class ValidationReport:

    def __init__(self):
        self.lines = []
        self.failures = []

    def info(self, name, value):
        self.lines.append(f"[INFO] {name}: {value}")

    def pass_check(self, name, value="PASS"):
        self.lines.append(f"[PASS] {name}: {value}")

    def fail_check(self, name, value):
        self.lines.append(f"[FAIL] {name}: {value}")
        self.failures.append(f"{name}: {value}")

    def section(self, title):
        self.lines.append("")
        self.lines.append("=" * 100)
        self.lines.append(title)
        self.lines.append("=" * 100)

    def finalize(self):

        self.lines.append("")
        self.lines.append("=" * 100)
        self.lines.append("FAILED VALIDATIONS")
        self.lines.append("=" * 100)

        if not self.failures:
            self.lines.append("NONE")
        else:
            self.lines.extend(self.failures)

        return "\n".join(self.lines)


# ======================================================================================
# Dataset Discovery
# ======================================================================================

def discover_datasets():

    groups = {}

    for entry in os.listdir("."):

        if not os.path.isdir(entry):
            continue

        match = DATASET_PATTERN.match(entry)

        if not match:
            continue

        block_hours = int(match.group(1))
        seed = int(match.group(2))

        groups.setdefault(block_hours, []).append(
            {
                "seed": seed,
                "path": entry,
            }
        )

    for block_hours in groups:
        groups[block_hours] = sorted(
            groups[block_hours],
            key=lambda x: x["seed"]
        )

    return groups


# ======================================================================================
# Utility Functions
# ======================================================================================

def dataset_checksum(ds):

    md5 = hashlib.md5()

    df = pd.DataFrame(ds[:])

    payload = df.to_csv(
        index=False
    ).encode()

    md5.update(payload)

    return md5.hexdigest()


def ecdf(values):

    values = np.asarray(values)
    values = values[~np.isnan(values)]

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def write_text(path, text):

    with open(path, "w") as f:
        f.write(text)


# ======================================================================================
# Loading
# ======================================================================================

def load_train_dataset(dataset_dir):

    train_dir = os.path.join(dataset_dir, "train_chunks")

    chunk_dirs = sorted(
        [
            os.path.join(train_dir, x)
            for x in os.listdir(train_dir)
            if x.startswith("train_chunk_")
        ]
    )

    ds_list = []

    for chunk_dir in chunk_dirs:
        ds_list.append(
            datasets.Dataset.load_from_disk(chunk_dir)
        )

    return datasets.concatenate_datasets(ds_list)


def load_dataset_group(dataset_dir):

    train_ds = load_train_dataset(dataset_dir)

    val_ds = datasets.Dataset.load_from_disk(
        os.path.join(dataset_dir, "val-blocks")
    )

    test_ds = datasets.Dataset.load_from_disk(
        os.path.join(dataset_dir, "test-blocks")
    )

    storm_ds = datasets.Dataset.load_from_disk(
        os.path.join(dataset_dir, "test-storm")
    )

    return train_ds, val_ds, test_ds, storm_ds


# ======================================================================================
# Plotting
# ======================================================================================

def create_cdf_plots(output_dir, cache):

    for variable in KEY_FEATURES:

        key = f"{variable}_train"

        if key not in cache:
            continue

        plt.figure(figsize=(8, 5))

        for split in ["train", "val", "test"]:

            x, y = ecdf(cache[f"{variable}_{split}"])

            plt.plot(
                x,
                y,
                label=split,
                linewidth=2
            )

        plt.xlabel(variable)
        plt.ylabel("CDF")
        plt.title(variable)

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"cdf_{variable}.png"
            ),
            dpi=200
        )

        plt.close()


def create_year_coverage_plot(output_dir, cache):

    plt.figure(figsize=(10, 5))

    for split in ["train", "val", "test"]:

        counts = (
            pd.Series(cache[f"{split}_year"])
            .value_counts()
            .sort_index()
        )

        counts = counts / counts.sum()

        plt.plot(
            counts.index,
            counts.values,
            marker="o",
            label=split
        )

    plt.xlabel("Year")
    plt.ylabel("Fraction")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "year_coverage.png"
        ),
        dpi=200
    )

    plt.close()


def create_timeline_plot(output_dir, block_df, seed):

    mapping = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    y = [mapping[x] for x in block_df["split"]]

    plt.figure(figsize=(14, 3))

    plt.scatter(
        block_df["start"],
        y,
        s=10
    )

    plt.yticks(
        [0, 1, 2],
        ["Train", "Val", "Test"]
    )

    plt.title(
        f"Seed {seed}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "timeline.png"
        ),
        dpi=200
    )

    plt.close()


# ======================================================================================
# Cache Builder
# ======================================================================================

def build_cache(train_ds, val_ds, test_ds):

    cache = {}

    for split, ds in [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]:

        times = pd.to_datetime(
            ds["DateTimeFormatted"]
        )

        cache[f"{split}_times"] = times.values.astype(
            "datetime64[m]"
        )

        cache[f"{split}_year"] = times.year.values

        for variable in KEY_FEATURES:

            if variable in ds.column_names:

                cache[
                    f"{variable}_{split}"
                ] = np.asarray(
                    ds[variable]
                )

    return cache


# ======================================================================================
# Individual Dataset Validation
# ======================================================================================

def validate_single_dataset(
    dataset_dir,
    seed,
    block_hours,
    output_dir,
):

    report = ValidationReport()

    report.section(
        f"DATASET VALIDATION - SEED {seed}"
    )

    train_ds, val_ds, test_ds, storm_ds = \
        load_dataset_group(dataset_dir)

    cache = build_cache(
        train_ds,
        val_ds,
        test_ds
    )

    train_times = set(cache["train_times"])
    val_times = set(cache["val_times"])
    test_times = set(cache["test_times"])

    storm_times = set(
        pd.to_datetime(
            storm_ds["DateTimeFormatted"]
        ).values.astype("datetime64[m]")
    )

    report.section("DATASET COUNTS")

    report.info(
        "Train Samples",
        f"{len(train_ds):,}"
    )

    report.info(
        "Validation Samples",
        f"{len(val_ds):,}"
    )

    report.info(
        "Test Samples",
        f"{len(test_ds):,}"
    )

    report.info(
        "Storm Samples",
        f"{len(storm_ds):,}"
    )

    report.section("OVERLAP VALIDATION")

    pairs = [
        ("Train-Val", train_times, val_times),
        ("Train-Test", train_times, test_times),
        ("Val-Test", val_times, test_times),
        ("Train-Storm", train_times, storm_times),
        ("Val-Storm", val_times, storm_times),
        ("Test-Storm", test_times, storm_times),
    ]

    for name, a, b in pairs:

        overlap = len(a.intersection(b))

        if overlap == 0:
            report.pass_check(name)
        else:
            report.fail_check(
                name,
                f"{overlap} overlapping timestamps"
            )

    report.section("PARTITION COMPLETENESS")

    union_count = len(
        train_times
        | val_times
        | test_times
        | storm_times
    )

    expected_count = (
        len(train_times)
        + len(val_times)
        + len(test_times)
        + len(storm_times)
    )

    if union_count == expected_count:
        report.pass_check(
            "complete_partition"
        )
    else:
        report.fail_check(
            "complete_partition",
            f"union={union_count}, expected={expected_count}"
        )

    report.section("STORM WINDOW VALIDATION")

    storm_dt = pd.to_datetime(
        storm_ds["DateTimeFormatted"]
    )

    if len(storm_dt) == 0:
        report.fail_check(
            "storm_dataset",
            "empty"
        )
    else:

        if storm_dt.min() >= STORM_START:
            report.pass_check(
                "storm_start_boundary"
            )
        else:
            report.fail_check(
                "storm_start_boundary",
                str(storm_dt.min())
            )

        if storm_dt.max() < STORM_END:
            report.pass_check(
                "storm_end_boundary"
            )
        else:
            report.fail_check(
                "storm_end_boundary",
                str(storm_dt.max())
            )

    report.section("FEATURE QUALITY")

    for feature in KEY_FEATURES:

        for split_name, ds in [
            ("train", train_ds),
            ("val", val_ds),
            ("test", test_ds),
            ("storm", storm_ds),
        ]:

            if feature not in ds.column_names:
                continue

            vals = pd.to_numeric(
                pd.Series(ds[feature]),
                errors="coerce"
            ).values

            nan_count = np.isnan(vals).sum()

            inf_count = np.isinf(vals).sum()

            if nan_count == 0:
                report.pass_check(
                    f"{split_name}_{feature}_nan"
                )
            else:
                report.fail_check(
                    f"{split_name}_{feature}_nan",
                    int(nan_count)
                )

            if inf_count == 0:
                report.pass_check(
                    f"{split_name}_{feature}_inf"
                )
            else:
                report.fail_check(
                    f"{split_name}_{feature}_inf",
                    int(inf_count)
                )

    report.section("BLOCK ASSIGNMENT VALIDATION")

    block_file = None

    for f in os.listdir(dataset_dir):

        if (
            f.startswith(
                f"block_assignment_{block_hours}h_s{seed}"
            )
            and f.endswith(".csv")
        ):
            block_file = os.path.join(
                dataset_dir,
                f
            )

    if block_file is None:

        report.fail_check(
            "block_assignment_file",
            "missing"
        )

    else:

        report.pass_check(
            "block_assignment_file"
        )

        block_df = pd.read_csv(block_file)

        expected_samples = (
            len(train_ds)
            + len(val_ds)
            + len(test_ds)
        )

        recorded_samples = (
            block_df["n_samples"].sum()
        )

        if recorded_samples == expected_samples:
            report.pass_check(
                "block_sample_counts"
            )
        else:
            report.fail_check(
                "block_sample_counts",
                f"recorded={recorded_samples}, expected={expected_samples}"
            )

        block_df["start"] = pd.to_datetime(
            block_df["start"]
        )

        block_df["end"] = pd.to_datetime(
            block_df["end"]
        )

        expected = pd.Timedelta(
            hours=block_hours
        )

        durations = (
            block_df["end"]
            - block_df["start"]
        )

        if (durations == expected).all():

            report.pass_check(
                "block_duration"
            )

        else:

            report.fail_check(
                "block_duration",
                "unexpected duration"
            )

        if block_df["block_id"].is_unique:
            report.pass_check(
                "unique_block_ids"
            )
        else:
            report.fail_check(
                "unique_block_ids",
                "duplicate block ids"
            )

        counts = (
            block_df["split"]
            .value_counts(normalize=True)
            .to_dict()
        )

        report.info(
            "Train Block Fraction",
            counts.get("train", 0)
        )

        report.info(
            "Validation Block Fraction",
            counts.get("val", 0)
        )

        report.info(
            "Test Block Fraction",
            counts.get("test", 0)
        )

        train_frac = counts.get("train", 0)
        val_frac = counts.get("val", 0)
        test_frac = counts.get("test", 0)

        tolerance = 0.02

        if abs(train_frac - EXPECTED_TRAIN_FRAC) <= tolerance:
            report.pass_check("train_fraction")
        else:
            report.fail_check(
                "train_fraction",
                train_frac
            )

        if abs(val_frac - EXPECTED_VAL_FRAC) <= tolerance:
            report.pass_check("val_fraction")
        else:
            report.fail_check(
                "val_fraction",
                val_frac
            )

        if abs(test_frac - EXPECTED_TEST_FRAC) <= tolerance:
            report.pass_check("test_fraction")
        else:
            report.fail_check(
                "test_fraction",
                test_frac
            )

        create_timeline_plot(
            output_dir,
            block_df,
            seed
        )

    report.section("DISTRIBUTION ANALYSIS")

    for variable in KEY_FEATURES:

        key = f"{variable}_train"

        if key not in cache:
            continue

        train_vals = cache[
            f"{variable}_train"
        ]

        val_vals = cache[
            f"{variable}_val"
        ]

        test_vals = cache[
            f"{variable}_test"
        ]

        report.info(
            f"{variable}_KS_train_val",
            ks_2samp(
                train_vals,
                val_vals
            ).statistic
        )

        report.info(
            f"{variable}_KS_train_test",
            ks_2samp(
                train_vals,
                test_vals
            ).statistic
        )

    create_cdf_plots(
        output_dir,
        cache
    )

    create_year_coverage_plot(
        output_dir,
        cache
    )

    write_text(
        os.path.join(
            output_dir,
            "validation_report.txt"
        ),
        report.finalize()
    )

    return {
        "seed": seed,
        "storm_checksum": dataset_checksum(storm_ds),
        "storm_count": len(storm_ds),
        "storm_times": storm_times,
    }


# ======================================================================================
# Cross-Seed Validation
# ======================================================================================

def validate_cross_seed(
    block_hours,
    dataset_infos,
):
    if not dataset_infos:
        raise RuntimeError(
            f"No datasets found for {block_hours}h"
        )
        
    output_dir = os.path.join(
        VALIDATION_ROOT,
        f"{block_hours}h"
    )

    report = ValidationReport()

    report.section(
        f"CROSS-SEED VALIDATION ({block_hours}h)"
    )

    report.section(
        "STORM CONSISTENCY"
    )
    
    if len(dataset_infos) == 1:
        report.info(
            "cross_seed_validation",
            "skipped (single seed)"
        )

        write_text(
            os.path.join(
                output_dir,
                "summary.txt"
            ),
            report.finalize()
        )

        return
    baseline = dataset_infos[0]

    for info in dataset_infos[1:]:

        if (
            info["storm_times"]
            == baseline["storm_times"]
        ):
            report.pass_check(
                f"storm_times_seed_{info['seed']}"
            )
        else:
            report.fail_check(
                f"storm_times_seed_{info['seed']}",
                "mismatch"
            )

        if (
            info["storm_checksum"]
            == baseline["storm_checksum"]
        ):
            report.pass_check(
                f"storm_checksum_seed_{info['seed']}"
            )
        else:
            report.fail_check(
                f"storm_checksum_seed_{info['seed']}",
                "mismatch"
            )

        if (
            info["storm_count"]
            == baseline["storm_count"]
        ):
            report.pass_check(
                f"storm_count_seed_{info['seed']}"
            )
        else:
            report.fail_check(
                f"storm_count_seed_{info['seed']}",
                "mismatch"
            )
    report.section(
        "BLOCK ASSIGNMENT DIVERSITY"
    )
    
    assignments = {}

    for info in dataset_infos:

        seed = info["seed"]

        dataset_dir = next(
            d["path"]
            for d in groups[block_hours]
            if d["seed"] == seed
        )

        file_name = next(
            x
            for x in os.listdir(dataset_dir)
            if x.startswith(
                f"block_assignment_{block_hours}h_s{seed}"
            )
        )

        df = pd.read_csv(
            os.path.join(
                dataset_dir,
                file_name
            )
        )

        assignments[seed] = (
            df.sort_values("block_id")
            .set_index("block_id")["split"]
        )

    for a, b in combinations(
        assignments.keys(),
        2
    ):

        diff = (
            assignments[a]
            != assignments[b]
        ).mean()

        report.info(
            f"seed_{a}_vs_seed_{b}_different_fraction",
            f"{100.0 * diff:.2f}%"
        )

    write_text(
        os.path.join(
            output_dir,
            "summary.txt"
        ),
        report.finalize()
    )


# ======================================================================================
# Main
# ======================================================================================

def main():

    global groups

    groups = discover_datasets()

    if not groups:
        raise RuntimeError(
            "No block-split datasets discovered."
        )

    for block_hours in sorted(groups):

        log(
            f"\n========== "
            f"{block_hours}h "
            f"=========="
        )

        block_output = os.path.join(
            VALIDATION_ROOT,
            f"{block_hours}h"
        )

        os.makedirs(
            block_output,
            exist_ok=True
        )

        cross_seed_info = []

        for item in groups[block_hours]:
            seed = item["seed"]
            dataset_dir = item["path"]

            seed_out = os.path.join(
                block_output,
                f"seed_{seed}"
            )

            os.makedirs(
                seed_out,
                exist_ok=True
            )

            result = validate_single_dataset(
                dataset_dir,
                seed,
                block_hours,
                seed_out,
            )

            cross_seed_info.append(
                result
            )

        validate_cross_seed(
            block_hours,
            cross_seed_info
        )

    log("\nValidation complete.")


if __name__ == "__main__":
    main()