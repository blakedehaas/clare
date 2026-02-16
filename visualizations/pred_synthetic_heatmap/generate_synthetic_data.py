import os
import sys
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --- Configuration ---
# Define the base path to the data directory relative to the script's location
# Goes up two levels (from pred_synthetic_heatmap -> visualizations -> clare)
# and then down into the dataset/input_dataset directory.
BASE_DATA_DIR = os.path.join('..', '..', 'dataset', 'input_dataset')

# Define paths to the raw solar index data files using the base path
KP_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_kp_index.lst')
OMNI_AL_SYMH_PATH = os.path.join(BASE_DATA_DIR, 'omni_al_index_symh', '*.lst')
F107_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_f107', '*.lst')


# Output filename for the new synthetic dataset
OUTPUT_FILENAME = "synthetic_output_dataset.parquet"

# --- Parameters for the synthetic data grid ---
TIME_START = "1991-01-31 00:00:00"
TIME_END = "1991-02-07 00:00:00"
TIME_INCREMENT = "10min"

ALTITUDE_START_KM = 1000
ALTITUDE_END_KM = 8000
ALTITUDE_INCREMENT_KM = 10

# BATCH_SIZE controls how many timestamps are processed at once.
# Adjust based on your available RAM.
BATCH_SIZE = 500

# Buffer to ensure there is enough historical data for the first timestamp's features.
# Must be greater than the longest lookback period (3 days).
BUFFER_DAYS = 4

# Earth's radius in kilometers
RE = 6378

# Constant values for synthetic data points
FIXED_VALUES = {
    'ILAT': 45.0,
    'GMLT': 0.00,
    'XXLAT': 0.0,
    'XXLON': 0.0,
    'GCLAT': 0.0,
    'GCLON': 0.0
}

# --- Define input columns ---
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']


def load_solar_indices(omni_al_symh_path, f107_file_path, kp_file_path):
    """
    Reads all solar index data files (.lst) and returns cleaned DataFrames.
    """
    # --- 1. SYM-H and AL Index Data ---
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

    # --- 2. F10.7 Solar Flux Index ---
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

    # --- 3. Kp Index Data ---
    print("Processing Kp index data...")
    if not os.path.exists(kp_file_path):
        raise FileNotFoundError(f"KP Index file not found at: {kp_file_path}")
        
    try:
        kp_df = pd.read_csv(kp_file_path, sep=r'\s+', names=['Year', 'DOY', 'Hour', 'Kp_index'])
        kp_df['DateTime'] = pd.to_datetime(kp_df['Year'] * 1000 + kp_df['DOY'], format='%Y%j') \
                            + pd.to_timedelta(kp_df['Hour'], unit='h')
        kp_df = kp_df[['DateTime', 'Kp_index']]
        kp_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
        kp_df.set_index('DateTime', inplace=True)
        kp_df.sort_index(inplace=True)
    except Exception as e:
        raise IOError(f"Error reading kp index file {kp_file_path}: {str(e)}")

    return omni_df, f107_df, kp_df

def create_temporal_solar_features(target_times, omni_df, f107_df, kp_df):
    """
    Generates a DataFrame with temporally expanded solar features for a given
    DatetimeIndex.
    """
    all_features = []
    
    # --- 1. AL Index (10 min for 5 hours) ---
    print("Expanding temporal features for AL index...")
    al_time_range = pd.timedelta_range(start='0m', end='5h', freq='10min')
    al_timestamps = target_times.values[:, None] - al_time_range.values
    al_values = omni_df['AL_index'].reindex(al_timestamps.ravel()).values.reshape(len(target_times), -1)
    al_cols = {f'AL_index_{i}': al_values[:, i] for i in range(al_values.shape[1])}
    all_features.append(pd.DataFrame(al_cols, index=target_times))

    # --- 2. SYM-H (30 min for 3 days) ---
    print("Expanding temporal features for SYM-H index...")
    sym_h_time_range = pd.timedelta_range(start='0m', end='3d', freq='30min')
    sym_h_timestamps = target_times.values[:, None] - sym_h_time_range.values
    sym_h_values = omni_df['SYM_H'].reindex(sym_h_timestamps.ravel()).values.reshape(len(target_times), -1)
    sym_h_cols = {f'SYM_H_{i}': sym_h_values[:, i] for i in range(sym_h_values.shape[1])}
    all_features.append(pd.DataFrame(sym_h_cols, index=target_times))
    
    # --- 3. F10.7 Index(1 day for past 3 days) ---
    print("Expanding temporal features for F10.7 index...")
    # This range gives 4 points for days 0, -1, -2, and -3, matching the original code
    f107_time_range = pd.timedelta_range(start='0h', end='72h', freq='24h') 
    f107_timestamps = target_times.values[:, None] - f107_time_range.values
    f107_timestamps = pd.DatetimeIndex(f107_timestamps.ravel()).round('h')
    f107_values = f107_df['f107_index'].reindex(f107_timestamps, method='nearest').values.reshape(len(target_times), -1)
    f107_cols = {f'f107_index_{i}': f107_values[:, i] for i in range(f107_values.shape[1])}
    all_features.append(pd.DataFrame(f107_cols, index=target_times))

    # --- 4. Kp Index (Instantaneous) ---
    print("Adding instantaneous Kp index...")
    rounded_index = target_times.round('H')
    kp_values = kp_df['Kp_index'].reindex(rounded_index, method='nearest').values
    all_features.append(pd.DataFrame({'Kp_index': kp_values}, index=target_times))

    # --- Combine all features ---
    print("Concatenating all temporal features...")
    final_df = pd.concat(all_features, axis=1)
    
    # Replace placeholder values common in this data
    final_df.replace([999.9, 99999.99], np.nan, inplace=True)
    
    return final_df

def calculate_glat(altitude_km):
    """Calculates GLAT based on the provided formula. Vectorized for performance."""
    x = np.sqrt((altitude_km + RE) / (2 * RE))
    x = np.clip(x, -1.0, 1.0)
    theta_radian = np.arcsin(x)
    return 90.0 - np.rad2deg(theta_radian)

def create_batch(solar_features_subset, altitude_range):
    """
    Creates a DataFrame for a batch of timestamps combined with all altitudes.
    """
    # Create the grid of (timestamp x altitude) using a cross merge
    batch_df = pd.DataFrame({
        'DateTimeFormatted': solar_features_subset['DateTimeFormatted'].repeat(len(altitude_range)),
        'Altitude': np.tile(altitude_range, len(solar_features_subset))
    })

    # Merge the solar features into the expanded grid
    batch_df = pd.merge(batch_df, solar_features_subset, on='DateTimeFormatted', how='left')

    # Add calculated and fixed value columns
    batch_df['GLAT'] = calculate_glat(batch_df['Altitude'].values)
    for col, val in FIXED_VALUES.items():
        batch_df[col] = val

    # Ensure final column order matches the required schema
    all_cols = ['DateTimeFormatted'] + input_columns
    final_cols_order = [col for col in all_cols if col in batch_df.columns]
    
    return batch_df[final_cols_order]

def main():
    """
    Main function to generate and save the synthetic dataset.
    """
    print("--- Synthetic Data Generation with Temporal Features ---")

    # --- 1. Load raw solar indices data ---
    try:
        omni_df, f107_df, kp_df = load_solar_indices(OMNI_AL_SYMH_PATH, F107_FILE_PATH, KP_FILE_PATH)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        print("Please ensure the 'input_dataset' directory and its contents are in the correct location.")
        sys.exit(1)

    # --- 2. Create time grids, including a buffer for temporal calculations ---
    buffer = pd.Timedelta(days=BUFFER_DAYS)
    buffered_start_time = pd.to_datetime(TIME_START) - buffer

    # The time range for which we need to calculate features (includes the buffer)
    time_range_for_features = pd.date_range(start=buffered_start_time, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')
    
    # The final desired time range for the output file
    time_range_final = pd.date_range(start=TIME_START, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')
    
    altitude_range = np.arange(ALTITUDE_START_KM, ALTITUDE_END_KM + 1, ALTITUDE_INCREMENT_KM, dtype=np.float32)
    
    print(f"\nGenerating data for {len(time_range_final)} final time steps and {len(altitude_range)} altitude levels.")
    print(f"Using a {BUFFER_DAYS}-day buffer for accurate temporal feature calculation (starting from {buffered_start_time}).")

    # --- 3. Create temporal solar features for the extended time range ---
    solar_features_df = create_temporal_solar_features(time_range_for_features, omni_df, f107_df, kp_df)

    # --- 4. Filter out the buffer period to keep only the desired time range ---
    print(f"Filtering out the buffer period to start from {TIME_START}.")
    solar_features_df = solar_features_df[solar_features_df.index >= TIME_START].copy()
    solar_features_df.reset_index(inplace=True)
    solar_features_df.rename(columns={'index': 'DateTimeFormatted'}, inplace=True)

    # --- 5. Build and save the synthetic dataset in batches ---
    print(f"\nConstructing and saving data in batches of {BATCH_SIZE} timestamps...")
    
    # Define the schema from the first batch to ensure consistency
    first_batch_df = create_batch(solar_features_df.iloc[0:1], altitude_range)
    schema = pa.Table.from_pandas(first_batch_df).schema
    
    # Use a ParquetWriter to append batches efficiently
    with pq.ParquetWriter(OUTPUT_FILENAME, schema) as writer:
        table = pa.Table.from_pandas(first_batch_df, schema=schema)
        writer.write_table(table)
        
        # Process the remaining batches
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