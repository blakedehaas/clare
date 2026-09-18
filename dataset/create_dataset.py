import glob
import os
import shutil

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_MINUTES = int(os.environ.get("BLOCK_MINUTES", 212))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 0))
TRAIN_FRACTION = 0.9
STORM_START = pd.Timestamp("1991-01-31")
STORM_END = pd.Timestamp("1991-02-07")
HISTORY_WINDOW = pd.Timedelta(hours=72)
T0 = pd.Timestamp("1990-01-01")

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    f"processed_dataset_blocksplit_{BLOCK_MINUTES}m_s{SPLIT_SEED}",
)
AKEBONO_PATH = os.path.join(SCRIPT_DIR, "input_dataset", "Akebono_combined.tsv")
KP_PATH = os.path.join(SCRIPT_DIR, "input_dataset", "omni_kp_index.lst")
AL_SYMH_GLOB = os.path.join(SCRIPT_DIR, "input_dataset", "omni_al_index_symh", "*.lst")
F107_GLOB = os.path.join(SCRIPT_DIR, "input_dataset", "omni_f107", "*.lst")


def require_inputs():
    if not os.path.isfile(AKEBONO_PATH):
        raise FileNotFoundError(AKEBONO_PATH)
    if not os.path.isfile(KP_PATH):
        raise FileNotFoundError(KP_PATH)

    al_symh_files = sorted(glob.glob(AL_SYMH_GLOB))
    f107_files = sorted(glob.glob(F107_GLOB))
    if not al_symh_files:
        raise FileNotFoundError(AL_SYMH_GLOB)
    if not f107_files:
        raise FileNotFoundError(F107_GLOB)
    return al_symh_files, f107_files


def report_rows(before, frame, label):
    print(f"{label}: removed {before - len(frame):,}; remaining {len(frame):,}")
    return len(frame)


def load_akebono():
    chunks = []
    total = 0
    for chunk in tqdm(
        pd.read_csv(AKEBONO_PATH, sep="\t", chunksize=500_000),
        desc="Loading Akebono",
    ):
        chunk["DateFormatted"] = pd.to_datetime(chunk["DateFormatted"], errors="coerce")
        total += len(chunk)
        chunks.append(chunk)

    frame = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"Initial rows: {total:,}")

    before = total
    frame = frame[~(frame[["XXLAT", "XXLON"]] == 999).any(axis=1)].copy()
    before = report_rows(before, frame, "Invalid XXLAT/XXLON")

    frame = frame[frame["ILAT"] <= 90].copy()
    before = report_rows(before, frame, "ILAT > 90")

    frame = frame[(frame["Altitude"] >= 1000) & (frame["Altitude"] <= 8000)].copy()
    before = report_rows(before, frame, "Altitude outside 1000-8000 km")

    frame = frame.dropna(subset=["DateFormatted"])
    frame = frame[frame["DateFormatted"] >= T0].copy()
    report_rows(before, frame, "Invalid/pre-1990 dates")

    frame["DateTimeFormatted"] = pd.to_datetime(
        frame["DateFormatted"].dt.strftime("%Y-%m-%d")
        + " "
        + frame["TimeFormatted"].astype(str),
        errors="coerce",
    ).dt.floor("min")
    frame = frame.dropna(subset=["DateTimeFormatted"])
    frame = frame.drop(
        columns=["DateFormatted", "TimeFormatted", "Date", "Time"],
        errors="ignore",
    )
    frame = frame.set_index("DateTimeFormatted").sort_index()
    return frame


def load_al_symh(files):
    frames = []
    columns = ["Year", "Day", "Hour", "Minute", "AL_index", "SYM_H"]
    for path in tqdm(files, desc="Loading AL/SYM-H"):
        frame = pd.read_csv(path, sep=r"\s+", names=columns)
        frame["DateTime"] = (
            pd.to_datetime(frame["Year"] * 1000 + frame["Day"], format="%Y%j")
            + pd.to_timedelta(frame["Hour"], unit="h")
            + pd.to_timedelta(frame["Minute"], unit="m")
        )
        frames.append(frame[["DateTime", "AL_index", "SYM_H"]])

    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates("DateTime").set_index("DateTime").sort_index()


def load_f107(files):
    frames = []
    columns = ["Year", "Day", "Hour", "f107_index"]
    for path in tqdm(files, desc="Loading F10.7"):
        frame = pd.read_csv(path, sep=r"\s+", names=columns)
        frame["DateTime"] = (
            pd.to_datetime(frame["Year"] * 1000 + frame["Day"], format="%Y%j")
            + pd.to_timedelta(frame["Hour"], unit="h")
        )
        frames.append(frame[["DateTime", "f107_index"]])

    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates("DateTime").set_index("DateTime").sort_index()


def load_kp():
    frame = pd.read_csv(KP_PATH, sep=r"\s+", names=["Year", "DOY", "Hour", "Kp_index"])
    frame["DateTime"] = (
        pd.to_datetime(frame["Year"] * 1000 + frame["DOY"], format="%Y%j")
        + pd.to_timedelta(frame["Hour"], unit="h")
    )
    return frame[["DateTime", "Kp_index"]].drop_duplicates("DateTime").set_index("DateTime").sort_index()


def add_history_features(frame, al_symh, f107, kp):
    index = frame.index

    al_offsets = pd.timedelta_range("0m", "5h", freq="10min")
    al_times = pd.DatetimeIndex((index.values[:, None] - al_offsets.values).ravel())
    al_values = al_symh["AL_index"].reindex(al_times).to_numpy().reshape(len(index), -1)

    symh_offsets = pd.timedelta_range("0m", "3d", freq="30min")
    symh_times = pd.DatetimeIndex((index.values[:, None] - symh_offsets.values).ravel())
    symh_values = al_symh["SYM_H"].reindex(symh_times).to_numpy().reshape(len(index), -1)

    f107_offsets = pd.timedelta_range("0h", "72h", freq="24h")
    f107_times = pd.DatetimeIndex((index.values[:, None] - f107_offsets.values).ravel()).floor("h")
    f107_values = f107["f107_index"].reindex(f107_times).to_numpy().reshape(len(index), -1)

    history = {
        **{f"AL_index_{i}": al_values[:, i] for i in range(al_values.shape[1])},
        **{f"SYM_H_{i}": symh_values[:, i] for i in range(symh_values.shape[1])},
        **{f"f107_index_{i}": f107_values[:, i] for i in range(f107_values.shape[1])},
    }
    frame = pd.concat([frame, pd.DataFrame(history, index=index)], axis=1)
    frame = frame.drop(columns=["AL_index", "SYM_H"], errors="ignore")
    frame["Kp_index"] = kp["Kp_index"].reindex(index.floor("h")).to_numpy()
    return frame


def block_ids(index):
    return ((index - T0) // pd.Timedelta(minutes=BLOCK_MINUTES)).astype(int)


def save_dataset(frame, directory):
    path = os.path.join(OUTPUT_DIR, directory)
    dataset = datasets.Dataset(pa.Table.from_pandas(frame))
    dataset.save_to_disk(path)
    print(f"Saved {len(frame):,} rows to {path}")


def main():
    print(f"BLOCK_MINUTES={BLOCK_MINUTES}, SPLIT_SEED={SPLIT_SEED}")
    al_symh_files, f107_files = require_inputs()

    frame = load_akebono()
    frame = add_history_features(
        frame,
        load_al_symh(al_symh_files),
        load_f107(f107_files),
        load_kp(),
    )

    nan_rows = frame.isna().any(axis=1)
    print(f"Rows containing NaN removed: {int(nan_rows.sum()):,}")
    frame = frame.loc[~nan_rows].copy()
    if frame.empty:
        raise RuntimeError("No rows remain after filtering.")

    full_n_samples = len(frame)
    full_n_blocks = int(pd.Series(block_ids(frame.index)).nunique())
    full_start = frame.index.min()
    full_end = frame.index.max()

    embargo_end = STORM_END + HISTORY_WINDOW
    storm_mask = (frame.index >= STORM_START) & (frame.index < STORM_END)
    embargo_mask = (frame.index >= STORM_END) & (frame.index < embargo_end)

    storm_df = frame.loc[storm_mask].copy()
    embargo_df = frame.loc[embargo_mask].copy()
    development_df = frame.loc[~(storm_mask | embargo_mask)].copy()
    del frame

    development_block_ids = block_ids(development_df.index)
    block_counts = pd.Series(development_block_ids).value_counts().sort_index()
    occupied_blocks = block_counts.index.to_numpy()
    if len(occupied_blocks) < 2:
        raise RuntimeError("At least two occupied development blocks are required.")

    shuffled = np.random.default_rng(SPLIT_SEED).permutation(occupied_blocks)
    shuffled_counts = block_counts.reindex(shuffled).to_numpy()
    cumulative = np.cumsum(shuffled_counts)
    target = TRAIN_FRACTION * len(development_df)
    candidates = np.arange(1, len(shuffled))
    best_k = int(candidates[np.argmin(np.abs(cumulative[candidates - 1] - target))])

    train_blocks = set(shuffled[:best_k])
    val_blocks = set(shuffled[best_k:])
    assignment = pd.Series(development_block_ids, index=development_df.index).map(
        lambda value: "train" if value in train_blocks else "val"
    )
    train_df = development_df.loc[assignment == "train"].copy()
    val_df = development_df.loc[assignment == "val"].copy()

    if train_blocks & val_blocks:
        raise AssertionError("Train/validation block overlap detected.")
    for name, split in (("train", train_df), ("val", val_df), ("test-storm", storm_df)):
        if split.isna().any().any():
            raise AssertionError(f"{name} contains NaN values.")
    for name, split in (("train", train_df), ("val", val_df)):
        if ((split.index >= STORM_START) & (split.index < embargo_end)).any():
            raise AssertionError(f"{name} contains storm or embargo rows.")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    assignment_rows = []
    for block_id in occupied_blocks:
        assignment_rows.append(
            {
                "block_id": int(block_id),
                "start": T0 + pd.Timedelta(minutes=int(block_id) * BLOCK_MINUTES),
                "end": T0 + pd.Timedelta(minutes=(int(block_id) + 1) * BLOCK_MINUTES),
                "split": "train" if block_id in train_blocks else "val",
                "n_samples": int(block_counts.loc[block_id]),
            }
        )
    pd.DataFrame(assignment_rows).to_csv(
        os.path.join(OUTPUT_DIR, f"block_assignment_{BLOCK_MINUTES}m_s{SPLIT_SEED}.csv"),
        index=False,
    )

    def summary_row(name, split, n_blocks):
        return {
            "split": name,
            "n_samples": int(len(split)),
            "n_blocks": int(n_blocks),
            "time_start": split.index.min() if len(split) else pd.NaT,
            "time_end": split.index.max() if len(split) else pd.NaT,
        }

    summary = pd.DataFrame(
        [
            {
                "split": "full-filtered",
                "n_samples": full_n_samples,
                "n_blocks": full_n_blocks,
                "time_start": full_start,
                "time_end": full_end,
            },
            summary_row("train", train_df, len(train_blocks)),
            summary_row("val", val_df, len(val_blocks)),
            summary_row("test-storm", storm_df, pd.Series(block_ids(storm_df.index)).nunique()),
            summary_row("post-storm-embargo", embargo_df, pd.Series(block_ids(embargo_df.index)).nunique()),
        ]
    )
    summary["block_minutes"] = BLOCK_MINUTES
    summary["split_seed"] = SPLIT_SEED
    summary["storm_start"] = STORM_START
    summary["storm_end_exclusive"] = STORM_END
    summary["post_storm_embargo_end_exclusive"] = embargo_end
    summary.to_csv(
        os.path.join(OUTPUT_DIR, f"split_summary_{BLOCK_MINUTES}m_s{SPLIT_SEED}.csv"),
        index=False,
    )

    train_pct = 100.0 * len(train_df) / len(development_df)
    print(f"Train: {len(train_df):,} rows, {len(train_blocks):,} blocks ({train_pct:.4f}%)")
    print(f"Validation: {len(val_df):,} rows, {len(val_blocks):,} blocks ({100.0 - train_pct:.4f}%)")
    print(f"Held-out storm: {len(storm_df):,} rows")
    print(f"Post-storm embargo discarded: {len(embargo_df):,} rows")

    save_dataset(storm_df, "test-storm")
    save_dataset(val_df, "val-blocks")

    chunk_size = 250_000
    for i, start in enumerate(range(0, len(train_df), chunk_size), start=1):
        save_dataset(train_df.iloc[start : start + chunk_size], f"train_chunks/train_chunk_{i}")


if __name__ == "__main__":
    main()
