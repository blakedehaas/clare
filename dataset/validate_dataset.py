import json
import os

import datasets
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_MINUTES = int(os.environ.get("BLOCK_MINUTES", 212))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 0))
T0 = pd.Timestamp("1990-01-01")
STORM_START = pd.Timestamp("1991-01-31")
STORM_END = pd.Timestamp("1991-02-07")
EMBARGO_END = STORM_END + pd.Timedelta(hours=72)
DATASET_DIR = os.path.join(
    SCRIPT_DIR,
    f"processed_dataset_blocksplit_{BLOCK_MINUTES}m_s{SPLIT_SEED}",
)


def load_train():
    root = os.path.join(DATASET_DIR, "train_chunks")
    names = [name for name in os.listdir(root) if name.startswith("train_chunk_")]
    names.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if not names:
        raise RuntimeError(f"No training chunks found in {root}")
    return datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(os.path.join(root, name)) for name in names]
    )


def timestamps(ds):
    values = pd.to_datetime(ds["DateTimeFormatted"], errors="coerce")
    if values.isna().any():
        raise AssertionError("Invalid DateTimeFormatted values found.")
    return pd.DatetimeIndex(values)


def block_ids(index):
    return ((index - T0) // pd.Timedelta(minutes=BLOCK_MINUTES)).astype(int)


def assert_no_missing_or_nonfinite(ds, name, batch_size=50_000):
    for batch in ds.iter(batch_size=batch_size):
        for column, values in batch.items():
            if column == "DateTimeFormatted":
                if pd.to_datetime(values, errors="coerce").isna().any():
                    raise AssertionError(f"{name}: invalid timestamps in {column}")
                continue
            array = np.asarray(values)
            if np.issubdtype(array.dtype, np.number):
                if not np.isfinite(array.astype(np.float64, copy=False)).all():
                    raise AssertionError(f"{name}: non-finite values in {column}")
            elif pd.isna(array).any():
                raise AssertionError(f"{name}: missing values in {column}")


def assert_range(ds, column, minimum=None, maximum=None, maximum_exclusive=False):
    values = np.asarray(ds[column], dtype=np.float64)
    if minimum is not None and values.min() < minimum:
        raise AssertionError(f"{column} minimum {values.min()} < {minimum}")
    if maximum is not None:
        invalid = values.max() >= maximum if maximum_exclusive else values.max() > maximum
        if invalid:
            comparator = "<" if maximum_exclusive else "<="
            raise AssertionError(f"{column} maximum {values.max()} violates {comparator} {maximum}")


def main():
    required = [
        os.path.join(DATASET_DIR, "train_chunks"),
        os.path.join(DATASET_DIR, "val-blocks"),
        os.path.join(DATASET_DIR, "test-storm"),
        os.path.join(DATASET_DIR, f"block_assignment_{BLOCK_MINUTES}m_s{SPLIT_SEED}.csv"),
        os.path.join(DATASET_DIR, f"split_summary_{BLOCK_MINUTES}m_s{SPLIT_SEED}.csv"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing dataset artifacts:\n" + "\n".join(missing))

    train = load_train()
    val = datasets.Dataset.load_from_disk(os.path.join(DATASET_DIR, "val-blocks"))
    storm = datasets.Dataset.load_from_disk(os.path.join(DATASET_DIR, "test-storm"))
    splits = {"train": train, "val": val, "test-storm": storm}

    columns = set(train.column_names)
    for name, ds in splits.items():
        if set(ds.column_names) != columns:
            raise AssertionError(f"{name} column schema differs from training.")
        assert_no_missing_or_nonfinite(ds, name)
        assert_range(ds, "Altitude", 1000.0, 8000.0)
        assert_range(ds, "ILAT", maximum=90.0)
        assert_range(ds, "Kp_index", 0.0, 90.0)
        assert_range(ds, "Te1", 0.0, 15000.0, maximum_exclusive=True)

    train_time = timestamps(train)
    val_time = timestamps(val)
    storm_time = timestamps(storm)

    if len(np.intersect1d(train_time.values, val_time.values)):
        raise AssertionError("Train and validation share timestamps.")
    if ((train_time >= STORM_START) & (train_time < EMBARGO_END)).any():
        raise AssertionError("Training contains storm or embargo samples.")
    if ((val_time >= STORM_START) & (val_time < EMBARGO_END)).any():
        raise AssertionError("Validation contains storm or embargo samples.")
    if not ((storm_time >= STORM_START) & (storm_time < STORM_END)).all():
        raise AssertionError("Held-out storm contains rows outside the configured storm interval.")

    assignment_path = os.path.join(
        DATASET_DIR,
        f"block_assignment_{BLOCK_MINUTES}m_s{SPLIT_SEED}.csv",
    )
    assignment = pd.read_csv(assignment_path, parse_dates=["start", "end"])
    if set(assignment["split"]) != {"train", "val"}:
        raise AssertionError("Block assignment must contain only train and val labels.")
    if assignment["block_id"].duplicated().any():
        raise AssertionError("Duplicate block IDs in block assignment.")
    duration = (assignment["end"] - assignment["start"]).dt.total_seconds() / 60.0
    if not np.allclose(duration, BLOCK_MINUTES):
        raise AssertionError("Block assignment contains an invalid block duration.")

    train_blocks = set(block_ids(train_time))
    val_blocks = set(block_ids(val_time))
    if train_blocks & val_blocks:
        raise AssertionError("Train and validation share block IDs.")
    assigned_train = set(assignment.loc[assignment["split"] == "train", "block_id"].astype(int))
    assigned_val = set(assignment.loc[assignment["split"] == "val", "block_id"].astype(int))
    if train_blocks != assigned_train or val_blocks != assigned_val:
        raise AssertionError("Saved rows do not match the recorded block assignment.")

    observed_counts = pd.Series(np.concatenate([block_ids(train_time), block_ids(val_time)])).value_counts()
    recorded_counts = assignment.set_index("block_id")["n_samples"]
    if not observed_counts.sort_index().equals(recorded_counts.sort_index()):
        raise AssertionError("Recorded per-block sample counts do not match saved rows.")

    summary_path = os.path.join(
        DATASET_DIR,
        f"split_summary_{BLOCK_MINUTES}m_s{SPLIT_SEED}.csv",
    )
    summary = pd.read_csv(summary_path)
    summary_by_split = summary.set_index("split")
    for name, ds in splits.items():
        if int(summary_by_split.loc[name, "n_samples"]) != len(ds):
            raise AssertionError(f"Split summary row count mismatch for {name}.")

    development_rows = len(train) + len(val)
    train_fraction = len(train) / development_rows
    if abs(train_fraction - 0.9) > 0.001:
        raise AssertionError(f"Training fraction {train_fraction:.6f} is not approximately 0.9.")

    report = {
        "status": "PASS",
        "dataset_dir": os.path.basename(DATASET_DIR),
        "block_minutes": BLOCK_MINUTES,
        "split_seed": SPLIT_SEED,
        "splits": {
            "train": {"n_samples": len(train), "n_blocks": len(train_blocks)},
            "val": {"n_samples": len(val), "n_blocks": len(val_blocks)},
            "test-storm": {
                "n_samples": len(storm),
                "n_blocks": int(pd.Series(block_ids(storm_time)).nunique()),
            },
        },
        "storm_start": STORM_START.isoformat(),
        "storm_end_exclusive": STORM_END.isoformat(),
        "post_storm_embargo_end_exclusive": EMBARGO_END.isoformat(),
    }

    output_dir = os.path.join(SCRIPT_DIR, "dataset_validation")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "validation_report.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Validation report: {output_path}")


if __name__ == "__main__":
    main()
