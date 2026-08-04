import os
import hashlib
from itertools import combinations

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


BASE_DIR = "."
DATASET_PREFIX = "processed_dataset_blocksplit_s"
SEEDS = [0, 1, 2]

VALIDATION_DIR = "dataset_validation"
OLD_STORM_DATASET = "processed_dataset_01_31_storm/test-storm"

os.makedirs(VALIDATION_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def ecdf(values):
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def checksum_column(ds, column):
    arr = np.asarray(ds[column])
    return hashlib.md5(arr.tobytes()).hexdigest()


def load_train_dataset(dataset_dir):
    train_dir = os.path.join(dataset_dir, "train_chunks")

    chunk_dirs = sorted(
        [
            os.path.join(train_dir, d)
            for d in os.listdir(train_dir)
            if d.startswith("train_chunk_")
        ]
    )

    log(f"[INFO] Loading {len(chunk_dirs)} train chunks")

    ds_list = []

    for chunk_dir in chunk_dirs:
        log(f"[INFO] Loading {os.path.basename(chunk_dir)}")
        ds_list.append(datasets.Dataset.load_from_disk(chunk_dir))

    log("[INFO] Concatenating train chunks")

    return datasets.concatenate_datasets(ds_list)


def load_seed(seed):

    dataset_dir = f"{DATASET_PREFIX}{seed}"

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


def build_cache(seed, train_ds, val_ds, test_ds):

    seed_dir = os.path.join(
        VALIDATION_DIR,
        f"seed_{seed}"
    )

    os.makedirs(seed_dir, exist_ok=True)

    cache_file = os.path.join(
        seed_dir,
        "cached_stats.npz"
    )

    if os.path.exists(cache_file):

        log(f"[INFO] Loading cache for seed {seed}")

        data = np.load(
            cache_file,
            allow_pickle=True
        )

        return {k: data[k] for k in data.files}

    log(f"[INFO] Building cache for seed {seed}")

    cache = {}

    cache["train_times"] = pd.to_datetime(
        train_ds["DateTimeFormatted"]
    ).values.astype("datetime64[m]")

    cache["val_times"] = pd.to_datetime(
        val_ds["DateTimeFormatted"]
    ).values.astype("datetime64[m]")

    cache["test_times"] = pd.to_datetime(
        test_ds["DateTimeFormatted"]
    ).values.astype("datetime64[m]")

    cache["train_year"] = pd.to_datetime(
        train_ds["DateTimeFormatted"]
    ).year.values

    cache["val_year"] = pd.to_datetime(
        val_ds["DateTimeFormatted"]
    ).year.values

    cache["test_year"] = pd.to_datetime(
        test_ds["DateTimeFormatted"]
    ).year.values

    variables = [
        "Te1",
        "GMLT",
        "Kp_index",
        "f107_index_0",
    ]

    for var in variables:

        if var in train_ds.column_names:

            cache[f"{var}_train"] = np.asarray(train_ds[var])
            cache[f"{var}_val"] = np.asarray(val_ds[var])
            cache[f"{var}_test"] = np.asarray(test_ds[var])

    np.savez_compressed(cache_file, **cache)

    return cache


def min_gap_minutes(reference_times, query_times):

    reference_times = np.sort(reference_times)

    idx = np.searchsorted(
        reference_times,
        query_times
    )

    gaps = np.full(
        len(query_times),
        np.inf
    )

    left = idx > 0
    right = idx < len(reference_times)

    if np.any(left):
        gap_left = np.abs(
            query_times[left]
            - reference_times[idx[left] - 1]
        ).astype(int)

        gaps[left] = np.minimum(
            gaps[left],
            gap_left
        )

    if np.any(right):
        gap_right = np.abs(
            reference_times[idx[right]]
            - query_times[right]
        ).astype(int)

        gaps[right] = np.minimum(
            gaps[right],
            gap_right
        )

    return gaps


def write_report(seed, text):

    outdir = os.path.join(
        VALIDATION_DIR,
        f"seed_{seed}"
    )

    os.makedirs(outdir, exist_ok=True)

    with open(
        os.path.join(
            outdir,
            f"seed_{seed}_report.txt"
        ),
        "w"
    ) as f:

        f.write(text)


def create_cdf_plots(seed, cache):

    outdir = os.path.join(
        VALIDATION_DIR,
        f"seed_{seed}"
    )

    variables = [
        "Te1",
        "Kp_index",
        "f107_index_0",
        "GMLT",
    ]

    for variable in variables:

        train_key = f"{variable}_train"

        if train_key not in cache:
            continue

        log(f"[INFO] Seed {seed}: CDF {variable}")

        plt.figure(figsize=(8, 5))

        for split in [
            "train",
            "val",
            "test",
        ]:

            vals = cache[
                f"{variable}_{split}"
            ]

            x, y = ecdf(vals)

            plt.plot(
                x,
                y,
                label=split.capitalize(),
                linewidth=2
            )

        plt.xlabel(variable)
        plt.ylabel("CDF")
        plt.title(
            f"{variable} Distribution Comparison"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                outdir,
                f"cdf_{variable}.png"
            ),
            dpi=200,
        )

        plt.close()


def create_year_coverage_plot(seed, cache):

    outdir = os.path.join(
        VALIDATION_DIR,
        f"seed_{seed}"
    )

    plt.figure(figsize=(10, 5))

    for label in [
        "train",
        "val",
        "test",
    ]:

        years = cache[f"{label}_year"]

        counts = pd.Series(
            years
        ).value_counts().sort_index()

        counts = counts / counts.sum()

        plt.plot(
            counts.index,
            counts.values,
            marker="o",
            label=label.capitalize()
        )

    plt.xlabel("Year")
    plt.ylabel("Fraction")
    plt.title("Year Coverage")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            outdir,
            "year_coverage.png"
        ),
        dpi=200,
    )

    plt.close()


def create_timeline_plot(seed):

    df = pd.read_csv(
        os.path.join(
            f"{DATASET_PREFIX}{seed}",
            f"block_assignment_s{seed}.csv"
        )
    )

    df["start"] = pd.to_datetime(
        df["start"]
    )

    mapping = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    y = [
        mapping[x]
        for x in df["split"]
    ]

    plt.figure(figsize=(14, 3))

    plt.scatter(
        df["start"],
        y,
        s=12
    )

    plt.yticks(
        [0, 1, 2],
        ["Train", "Val", "Test"]
    )

    plt.title(
        f"Block Timeline Seed {seed}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VALIDATION_DIR,
            f"seed_{seed}",
            "timeline.png"
        ),
        dpi=200
    )

    plt.close()


def validate_seed(seed):

    log(f"\n========== SEED {seed} ==========")

    train_ds, val_ds, test_ds, storm_ds = load_seed(seed)

    cache = build_cache(
        seed,
        train_ds,
        val_ds,
        test_ds,
    )

    report = []

    report.append(
        f"Seed {seed} Validation Report\n"
    )

    report.append(
        "=" * 80 + "\n"
    )

    report.append(
        f"Train samples: {len(train_ds):,}\n"
    )

    report.append(
        f"Validation samples: {len(val_ds):,}\n"
    )

    report.append(
        f"Test samples: {len(test_ds):,}\n"
    )

    report.append(
        f"Storm samples: {len(storm_ds):,}\n\n"
    )

    combinations_to_check = [
        ("train_times", "val_times"),
        ("train_times", "test_times"),
        ("val_times", "test_times"),
    ]

    report.append(
        "Check 3 - Overlap\n"
    )

    for a, b in combinations_to_check:

        overlap = len(
            np.intersect1d(
                cache[a],
                cache[b]
            )
        )

        report.append(
            f"{a} vs {b}: {overlap}\n"
        )

    report.append("\n")

    val_gaps = min_gap_minutes(
        cache["train_times"],
        cache["val_times"]
    )

    test_gaps = min_gap_minutes(
        cache["train_times"],
        cache["test_times"]
    )

    report.append(
        "Check 4 - 6 Hour Buffer\n"
    )

    report.append(
        f"Validation minimum gap: {val_gaps.min()} minutes\n"
    )

    report.append(
        f"Test minimum gap: {test_gaps.min()} minutes\n\n"
    )

    report.append(
        "Check 5 - Storm Dataset\n"
    )

    if os.path.exists(OLD_STORM_DATASET):

        old_ds = datasets.Dataset.load_from_disk(
            OLD_STORM_DATASET
        )

        identical = (
            len(old_ds) == len(storm_ds)
            and checksum_column(old_ds, "Te1")
            == checksum_column(storm_ds, "Te1")
        )

        report.append(
            f"Storm identical: {identical}\n\n"
        )

    variables = [
        "Te1",
        "Kp_index",
        "f107_index_0",
        "GMLT",
    ]

    report.append(
        "Check 9 - Distribution Statistics\n\n"
    )

    for variable in variables:

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

        report.append(
            f"Variable: {variable}\n"
        )

        for name, vals in [
            ("Train", train_vals),
            ("Validation", val_vals),
            ("Test", test_vals),
        ]:

            vals = vals[
                ~np.isnan(vals)
            ]

            report.append(
                f"{name}: "
                f"mean={vals.mean():.4f}, "
                f"median={np.median(vals):.4f}, "
                f"std={vals.std():.4f}, "
                f"min={vals.min():.4f}, "
                f"max={vals.max():.4f}\n"
            )

        report.append(
            f"KS(train,val)="
            f"{ks_2samp(train_vals, val_vals).statistic:.6f}\n"
        )

        report.append(
            f"KS(train,test)="
            f"{ks_2samp(train_vals, test_vals).statistic:.6f}\n"
        )

        report.append(
            f"KS(val,test)="
            f"{ks_2samp(val_vals, test_vals).statistic:.6f}\n\n"
        )

    if "GMLT_train" in cache:

        report.append(
            "MLT Occupancy\n"
        )

        bins = np.arange(
            0,
            25,
            1
        )

        for split in [
            "train",
            "val",
            "test",
        ]:

            hist, _ = np.histogram(
                cache[f"GMLT_{split}"],
                bins=bins
            )

            hist = hist / hist.sum()

            report.append(
                f"{split}: "
                + ", ".join(
                    f"{x:.4f}"
                    for x in hist
                )
                + "\n"
            )

    write_report(
        seed,
        "".join(report)
    )

    create_cdf_plots(
        seed,
        cache
    )

    create_timeline_plot(seed)

    create_year_coverage_plot(
        seed,
        cache
    )

    return {
        "seed": seed,
        "cache": cache
    }


def write_cross_seed_report():

    lines = []

    lines.append(
        "Cross Seed Comparison\n"
    )

    lines.append(
        "=" * 80 + "\n\n"
    )

    assignments = {}

    for seed in SEEDS:

        df = pd.read_csv(
            os.path.join(
                f"{DATASET_PREFIX}{seed}",
                f"block_assignment_s{seed}.csv"
            )
        )

        assignments[seed] = (
            df.sort_values("block_id")
            .set_index("block_id")["split"]
        )

    for a, b in combinations(
        SEEDS,
        2
    ):

        difference = (
            assignments[a]
            != assignments[b]
        ).mean()

        lines.append(
            f"Seed {a} vs Seed {b}: "
            f"{100*difference:.2f}% different blocks\n"
        )

    with open(
        os.path.join(
            VALIDATION_DIR,
            "summary_all_seeds.txt"
        ),
        "w"
    ) as f:

        f.write("".join(lines))


def main():

    log(
        "\n========== DATASET VALIDATION ==========\n"
    )

    for seed in SEEDS:
        validate_seed(seed)

    write_cross_seed_report()

    log(
        "\nValidation complete."
    )


if __name__ == "__main__":
    main()