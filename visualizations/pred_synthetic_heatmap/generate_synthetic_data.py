import os
import sys
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import INPUT_COLUMNS, PROJECT_ROOT

# --- Configuration ---
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'input_dataset')

KP_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_kp_index.lst')
OMNI_AL_SYMH_PATH = os.path.join(BASE_DATA_DIR, 'omni_al_index_symh', '*.lst')
F107_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_f107', '*.lst')

OUTPUT_FILENAME = "synthetic_output_dataset.parquet"

TIME_START = "1991-01-31 00:00:00"
TIME_END = "1991-02-07 00:00:00"
TIME_INCREMENT = "10min"

ALTITUDE_START_KM = 1000
ALTITUDE_END_KM = 8000
ALTITUDE_INCREMENT_KM = 10

BATCH_SIZE = 500
BUFFER_DAYS = 4
RE = 6378

FIXED_VALUES = {
    'ILAT': 45.0,
    'GMLT': 0.00,
    'XXLAT': 0.0,
    'XXLON': 0.0,
    'GCLAT': 0.0,
    'GCLON': 0.0
}


def load_solar_indices(omni_al_symh_path, f107_file_path, kp_file_path):
    """Reads all solar index data files (.lst) and returns cleaned DataFrames."""
    print("Processing SYM-H and AL index data...")
    al_symh_files = glob.glob(omni_al_symh_path)
    if not al_symh_files:
        raise FileNotFoundError(f"No files found at: {omni_al_symh_path}")

    df_list = []
    for file in tqdm(al_symh_files, desc="Loading OMNI AL/SYM-H data"):
        try:
            columns = ['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H']
            df_temp = pd.read_csv(file, sep=r'\s+', names=columns)
            df_temp['DateTime'] = pd.to_datetime(df_temp['Year'] * 1000 + df_temp['Day'], format='%Y%j') \
                                + pd.to_timedelta(df_temp['Hour'], unit='h') \
                                + pd.to_timedelta(df_temp['Minute'], unit='m')
            df_list.append(df_temp[['DateTime', 'AL_index', 'SYM_H']])
        except Exception as e:
            print(f"Warning: Error reading file {file}: {str(e)}")
            continue
    omni_df = pd.concat(df_list, ignore_index=True)
    omni_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
    omni_df.set_index('DateTime', inplace=True)
    omni_df.sort_index(inplace=True)

    print("Processing F10.7 solar flux index data...")
    f107_files = glob.glob(f107_file_path)
    if not f107_files:
        raise FileNotFoundError(f"No files found at: {f107_file_path}")

    f107_list = []
    for file in tqdm(f107_files, desc="Loading F10.7 data"):
        try:
            columns = ['Year', 'Day', 'Hour', 'f107_index']
            f107_df = pd.read_csv(file, sep=r'\s+', names=columns)
            f107_df['DateTime'] = pd.to_datetime(f107_df['Year'] * 1000 + f107_df['Day'], format='%Y%j') \
                                  + pd.to_timedelta(f107_df['Hour'], unit='h')
            f107_list.append(f107_df[['DateTime', 'f107_index']])
        except Exception as e:
            print(f"Warning: Error reading file {file}: {str(e)}")
            continue
    f107_df = pd.concat(f107_list, ignore_index=True)
    f107_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
    f107_df.set_index('DateTime', inplace=True)
    f107_df.sort_index(inplace=True)

    print("Processing Kp index data...")
    if not os.path.exists(kp_file_path):
        raise FileNotFoundError(f"KP Index file not found at: {kp_file_path}")

    kp_df = pd.read_csv(kp_file_path, sep=r'\s+', names=['Year', 'DOY', 'Hour', 'Kp_index'])
    kp_df['DateTime'] = pd.to_datetime(kp_df['Year'] * 1000 + kp_df['DOY'], format='%Y%j') \
                        + pd.to_timedelta(kp_df['Hour'], unit='h')
    kp_df = kp_df[['DateTime', 'Kp_index']]
    kp_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
    kp_df.set_index('DateTime', inplace=True)
    kp_df.sort_index(inplace=True)

    return omni_df, f107_df, kp_df


def create_temporal_solar_features(target_times, omni_df, f107_df, kp_df):
    """Generates a DataFrame with temporally expanded solar features."""
    all_features = []

    print("Expanding temporal features for AL index...")
    al_time_range = pd.timedelta_range(start='0m', end='5h', freq='10min')
    al_timestamps = target_times.values[:, None] - al_time_range.values
    al_values = omni_df['AL_index'].reindex(al_timestamps.ravel()).values.reshape(len(target_times), -1)
    al_cols = {f'AL_index_{i}': al_values[:, i] for i in range(al_values.shape[1])}
    all_features.append(pd.DataFrame(al_cols, index=target_times))

    print("Expanding temporal features for SYM-H index...")
    sym_h_time_range = pd.timedelta_range(start='0m', end='3d', freq='30min')
    sym_h_timestamps = target_times.values[:, None] - sym_h_time_range.values
    sym_h_values = omni_df['SYM_H'].reindex(sym_h_timestamps.ravel()).values.reshape(len(target_times), -1)
    sym_h_cols = {f'SYM_H_{i}': sym_h_values[:, i] for i in range(sym_h_values.shape[1])}
    all_features.append(pd.DataFrame(sym_h_cols, index=target_times))

    print("Expanding temporal features for F10.7 index...")
    f107_time_range = pd.timedelta_range(start='0h', end='72h', freq='24h')
    f107_timestamps = target_times.values[:, None] - f107_time_range.values
    f107_timestamps = pd.DatetimeIndex(f107_timestamps.ravel()).round('h')
    f107_values = f107_df['f107_index'].reindex(f107_timestamps, method='nearest').values.reshape(len(target_times), -1)
    f107_cols = {f'f107_index_{i}': f107_values[:, i] for i in range(f107_values.shape[1])}
    all_features.append(pd.DataFrame(f107_cols, index=target_times))

    print("Adding instantaneous Kp index...")
    rounded_index = target_times.round('H')
    kp_values = kp_df['Kp_index'].reindex(rounded_index, method='nearest').values
    all_features.append(pd.DataFrame({'Kp_index': kp_values}, index=target_times))

    print("Concatenating all temporal features...")
    final_df = pd.concat(all_features, axis=1)
    final_df.replace([999.9, 99999.99], np.nan, inplace=True)

    return final_df


def calculate_glat(altitude_km):
    x = np.sqrt((altitude_km + RE) / (2 * RE))
    x = np.clip(x, -1.0, 1.0)
    theta_radian = np.arcsin(x)
    return 90.0 - np.rad2deg(theta_radian)


def create_batch(solar_features_subset, altitude_range):
    batch_df = pd.DataFrame({
        'DateTimeFormatted': solar_features_subset['DateTimeFormatted'].repeat(len(altitude_range)),
        'Altitude': np.tile(altitude_range, len(solar_features_subset))
    })

    batch_df = pd.merge(batch_df, solar_features_subset, on='DateTimeFormatted', how='left')

    batch_df['GLAT'] = calculate_glat(batch_df['Altitude'].values)
    for col, val in FIXED_VALUES.items():
        batch_df[col] = val

    all_cols = ['DateTimeFormatted'] + INPUT_COLUMNS
    final_cols_order = [col for col in all_cols if col in batch_df.columns]

    return batch_df[final_cols_order]


def main():
    print("--- Synthetic Data Generation with Temporal Features ---")

    try:
        omni_df, f107_df, kp_df = load_solar_indices(OMNI_AL_SYMH_PATH, F107_FILE_PATH, KP_FILE_PATH)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        print("Please ensure the 'input_dataset' directory and its contents are in the correct location.")
        sys.exit(1)

    buffer = pd.Timedelta(days=BUFFER_DAYS)
    buffered_start_time = pd.to_datetime(TIME_START) - buffer

    time_range_for_features = pd.date_range(start=buffered_start_time, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')
    time_range_final = pd.date_range(start=TIME_START, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')

    altitude_range = np.arange(ALTITUDE_START_KM, ALTITUDE_END_KM + 1, ALTITUDE_INCREMENT_KM, dtype=np.float32)

    print(f"\nGenerating data for {len(time_range_final)} final time steps and {len(altitude_range)} altitude levels.")

    solar_features_df = create_temporal_solar_features(time_range_for_features, omni_df, f107_df, kp_df)

    print(f"Filtering out the buffer period to start from {TIME_START}.")
    solar_features_df = solar_features_df[solar_features_df.index >= TIME_START].copy()
    solar_features_df.reset_index(inplace=True)
    solar_features_df.rename(columns={'index': 'DateTimeFormatted'}, inplace=True)

    print(f"\nConstructing and saving data in batches of {BATCH_SIZE} timestamps...")

    first_batch_df = create_batch(solar_features_df.iloc[0:1], altitude_range)
    schema = pa.Table.from_pandas(first_batch_df).schema

    with pq.ParquetWriter(OUTPUT_FILENAME, schema) as writer:
        table = pa.Table.from_pandas(first_batch_df, schema=schema)
        writer.write_table(table)

        for i in tqdm(range(1, len(solar_features_df), BATCH_SIZE), desc="Writing Batches"):
            batch_solar_features = solar_features_df.iloc[i : i + BATCH_SIZE]
            if batch_solar_features.empty:
                continue

            batch_df = create_batch(batch_solar_features, altitude_range)
            table = pa.Table.from_pandas(batch_df, schema=schema)
            writer.write_table(table)

    print(f"\n--- Synthetic data generation complete. Saved to '{OUTPUT_FILENAME}' ---")


if __name__ == "__main__":
    main()
