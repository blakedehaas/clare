"""Generate the shared synthetic time-altitude and solar-history dataset used by heatmap visualizations."""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BASE_DATA_DIR = PROJECT_ROOT / "dataset" / "input_dataset"

KP_FILE_PATH = BASE_DATA_DIR / "omni_kp_index.lst"
OMNI_AL_SYMH_GLOB = str(BASE_DATA_DIR / "omni_al_index_symh" / "*.lst")
F107_GLOB = str(BASE_DATA_DIR / "omni_f107" / "*.lst")

OUTPUT_PATH = SCRIPT_DIR / "synthetic_output_dataset.parquet"
TEMP_OUTPUT_PATH = SCRIPT_DIR / "synthetic_output_dataset.parquet.tmp"

TIME_START = "1991-01-28 00:00:00"
TIME_END = "1991-02-10 00:00:00"
TIME_INCREMENT = "10min"

ALTITUDE_START_KM = 1000
ALTITUDE_END_KM = 8000
ALTITUDE_INCREMENT_KM = 10

TIMESTAMPS_PER_BATCH = 500

SOLAR_COLUMNS = [
    *[f"AL_index_{i}" for i in range(31)],
    *[f"SYM_H_{i}" for i in range(145)],
    *[f"f107_index_{i}" for i in range(4)],
    "Kp_index",
]


def load_solar_indices():
    al_symh_frames = []
    for path in tqdm(sorted(glob.glob(OMNI_AL_SYMH_GLOB)), desc="Loading AL/SYM-H"):
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            names=["Year", "Day", "Hour", "Minute", "AL_index", "SYM_H"],
        )
        frame["DateTime"] = (
            pd.to_datetime(frame["Year"] * 1000 + frame["Day"], format="%Y%j")
            + pd.to_timedelta(frame["Hour"], unit="h")
            + pd.to_timedelta(frame["Minute"], unit="m")
        )
        al_symh_frames.append(frame[["DateTime", "AL_index", "SYM_H"]])

    if not al_symh_frames:
        raise FileNotFoundError(OMNI_AL_SYMH_GLOB)

    al_symh = (
        pd.concat(al_symh_frames, ignore_index=True)
        .drop_duplicates("DateTime")
        .set_index("DateTime")
        .sort_index()
    )

    f107_frames = []
    for path in tqdm(sorted(glob.glob(F107_GLOB)), desc="Loading F10.7"):
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            names=["Year", "Day", "Hour", "f107_index"],
        )
        frame["DateTime"] = (
            pd.to_datetime(frame["Year"] * 1000 + frame["Day"], format="%Y%j")
            + pd.to_timedelta(frame["Hour"], unit="h")
        )
        f107_frames.append(frame[["DateTime", "f107_index"]])

    if not f107_frames:
        raise FileNotFoundError(F107_GLOB)

    f107 = (
        pd.concat(f107_frames, ignore_index=True)
        .drop_duplicates("DateTime")
        .set_index("DateTime")
        .sort_index()
    )

    if not KP_FILE_PATH.is_file():
        raise FileNotFoundError(KP_FILE_PATH)

    kp = pd.read_csv(
        KP_FILE_PATH,
        sep=r"\s+",
        names=["Year", "DOY", "Hour", "Kp_index"],
    )
    kp["DateTime"] = (
        pd.to_datetime(kp["Year"] * 1000 + kp["DOY"], format="%Y%j")
        + pd.to_timedelta(kp["Hour"], unit="h")
    )
    kp = (
        kp[["DateTime", "Kp_index"]]
        .drop_duplicates("DateTime")
        .set_index("DateTime")
        .sort_index()
    )

    return al_symh, f107, kp


def create_temporal_solar_features(target_times, al_symh, f107, kp):
    """Match the temporal-index alignment used by dataset/create_dataset.py."""
    al_offsets = pd.timedelta_range("0m", "5h", freq="10min")
    al_times = pd.DatetimeIndex(
        (target_times.values[:, None] - al_offsets.values).ravel()
    )
    al_values = (
        al_symh["AL_index"]
        .reindex(al_times)
        .to_numpy()
        .reshape(len(target_times), -1)
    )

    symh_offsets = pd.timedelta_range("0m", "3d", freq="30min")
    symh_times = pd.DatetimeIndex(
        (target_times.values[:, None] - symh_offsets.values).ravel()
    )
    symh_values = (
        al_symh["SYM_H"]
        .reindex(symh_times)
        .to_numpy()
        .reshape(len(target_times), -1)
    )

    f107_offsets = pd.timedelta_range("0h", "72h", freq="24h")
    f107_times = pd.DatetimeIndex(
        (target_times.values[:, None] - f107_offsets.values).ravel()
    ).floor("h")
    f107_values = (
        f107["f107_index"]
        .reindex(f107_times)
        .to_numpy()
        .reshape(len(target_times), -1)
    )

    features = pd.DataFrame(
        {
            **{f"AL_index_{i}": al_values[:, i] for i in range(al_values.shape[1])},
            **{f"SYM_H_{i}": symh_values[:, i] for i in range(symh_values.shape[1])},
            **{f"f107_index_{i}": f107_values[:, i] for i in range(f107_values.shape[1])},
        },
        index=target_times,
    )
    features["Kp_index"] = (
        kp["Kp_index"]
        .reindex(target_times.floor("h"))
        .to_numpy()
    )
    return features


def main():
    al_symh, f107, kp = load_solar_indices()

    timestamps = pd.date_range(
        TIME_START,
        TIME_END,
        freq=TIME_INCREMENT,
        inclusive="left",
    )
    altitudes = np.arange(
        ALTITUDE_START_KM,
        ALTITUDE_END_KM + 1,
        ALTITUDE_INCREMENT_KM,
        dtype=np.float32,
    )

    if TEMP_OUTPUT_PATH.exists():
        TEMP_OUTPUT_PATH.unlink()

    writer = None
    try:
        for start in tqdm(
            range(0, len(timestamps), TIMESTAMPS_PER_BATCH),
            desc="Generating synthetic dataset",
        ):
            batch_times = timestamps[start : start + TIMESTAMPS_PER_BATCH]
            solar = create_temporal_solar_features(
                batch_times,
                al_symh,
                f107,
                kp,
            )

            batch = pd.DataFrame(
                {
                    "DateTimeFormatted": solar.index.repeat(len(altitudes)),
                    "Altitude": np.tile(altitudes, len(solar)),
                }
            )
            batch = batch.merge(
                solar,
                left_on="DateTimeFormatted",
                right_index=True,
                how="left",
            )

            required = ["DateTimeFormatted", "Altitude"] + SOLAR_COLUMNS
            batch = batch[required]

            nan_rows = batch[SOLAR_COLUMNS].isna().any(axis=1)
            if nan_rows.any():
                raise ValueError(
                    f"Synthetic feature generation produced "
                    f"{int(nan_rows.sum()):,} rows with missing solar-history values."
                )

            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(TEMP_OUTPUT_PATH, table.schema)
            writer.write_table(table)

        if writer is None:
            raise RuntimeError("No synthetic rows were generated.")

        writer.close()
        writer = None
        TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)

    except Exception:
        if writer is not None:
            writer.close()
        if TEMP_OUTPUT_PATH.exists():
            TEMP_OUTPUT_PATH.unlink()
        raise

    print(f"Saved synthetic heatmap input dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
