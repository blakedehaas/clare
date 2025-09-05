import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from tqdm import tqdm

# --- Configuration ---
# Path to the existing dataset to borrow solar features from
EXISTING_DATASET_DIR = os.path.join("../../", "dataset", "processed_dataset_01_31_storm", "test-storm")
# Output filename for the new synthetic dataset
OUTPUT_FILENAME = "synthetic_data_for_heatmap.parquet"

# --- Parameters for the synthetic data grid ---
TIME_START = "1991-01-31 00:00:00"
TIME_END = "1991-02-07 00:00:00"
TIME_INCREMENT = "10min"

ALTITUDE_START_KM = 1000
ALTITUDE_END_KM = 8000
ALTITUDE_INCREMENT_KM = 10

# BATCH_SIZE controls how many timestamps are processed at once.
# Adjust based on your available RAM. A smaller number uses less RAM.
BATCH_SIZE = 500 

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

# --- Define input columns directly in the script ---
# (This section is unchanged, so it's collapsed for brevity)
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
SOLAR_FEATURE_COLS = [col for col in input_columns if any(sub in col for sub in ['AL_index', 'SYM_H', 'f107_index', 'Kp_index'])]


def calculate_glat(altitude_km):
    """Calculates GLAT based on the provided formula. Vectorized for performance."""
    x = np.sqrt((altitude_km + RE) / (2 * RE))
    x = np.clip(x, -1.0, 1.0)
    theta_radian = np.arcsin(x)
    return 90.0 - np.rad2deg(theta_radian)

def main():
    """
    Main function to generate and save the synthetic dataset using a memory-efficient
    batch processing approach.
    """
    print("--- Optimized Synthetic Data Generation ---")

    # --- 1. Load existing data and prepare for efficient lookup ---
    print(f"Loading existing dataset from: {EXISTING_DATASET_DIR}")
    if not os.path.exists(EXISTING_DATASET_DIR):
        print(f"Error: Directory not found at '{EXISTING_DATASET_DIR}'. Exiting.")
        sys.exit(1)
        
    original_dataset = load_from_disk(EXISTING_DATASET_DIR)
    original_df = original_dataset.to_pandas()
    original_df['DateTimeFormatted'] = pd.to_datetime(original_df['DateTimeFormatted'])
    # Sort for merge_asof, which is crucial for performance
    original_df = original_df.sort_values('DateTimeFormatted').reset_index(drop=True)
    print("Existing dataset loaded and sorted.")

    # --- 2. Create the time and altitude grids ---
    time_range = pd.date_range(start=TIME_START, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')
    altitude_range = np.arange(ALTITUDE_START_KM, ALTITUDE_END_KM + 1, ALTITUDE_INCREMENT_KM, dtype=np.float32)
    print(f"Generating data for {len(time_range)} time steps and {len(altitude_range)} altitude levels.")

    # --- 3. Pre-calculate solar features for all synthetic timestamps ---
    # This is MUCH faster than finding the closest time in a loop.
    print("Mapping solar features to synthetic timestamps...")
    synthetic_times_df = pd.DataFrame({'DateTimeFormatted': time_range})
    # Use merge_asof to find the nearest row in original_df for each synthetic timestamp
    solar_features_df = pd.merge_asof(
        synthetic_times_df,
        original_df[['DateTimeFormatted'] + SOLAR_FEATURE_COLS],
        on='DateTimeFormatted',
        direction='nearest'
    )

    # --- 4. Build and save the synthetic dataset in batches ---
    print(f"Constructing and saving data in batches of {BATCH_SIZE} timestamps...")
    
    # Define the schema for the Parquet file from the first batch
    # This ensures consistency across all written chunks.
    first_batch_df = create_batch(solar_features_df.iloc[0:1], altitude_range)
    schema = pa.Table.from_pandas(first_batch_df).schema
    
    # Use a ParquetWriter to efficiently append batches to a single file
    with pq.ParquetWriter(OUTPUT_FILENAME, schema) as writer:
        # Process the first batch (which we already created)
        table = pa.Table.from_pandas(first_batch_df, schema=schema)
        writer.write_table(table)
        
        # Process the remaining batches
        # We start from 1 because the 0th index was in the first batch
        range_start = 1
        for i in tqdm(range(range_start, len(solar_features_df), BATCH_SIZE), desc="Writing Batches"):
            batch_solar_features = solar_features_df.iloc[i : i + BATCH_SIZE]
            
            if batch_solar_features.empty:
                continue

            # Create the DataFrame for the current batch
            batch_df = create_batch(batch_solar_features, altitude_range)
            
            # Convert to pyarrow Table and write to disk
            table = pa.Table.from_pandas(batch_df, schema=schema)
            writer.write_table(table)

    print(f"\n--- Synthetic data generation complete. Saved to '{OUTPUT_FILENAME}' ---")

    # --- 5. Verify the structure of the saved data ---
    print("\n--- Verifying the structure of the first entry from the Parquet file ---")
    try:
        # Read only the first row from the created file for verification
        verify_df = pd.read_parquet(OUTPUT_FILENAME, engine='pyarrow').head(1)
        first_entry_string = verify_df.iloc[0].to_string()
        with open("first_entry_verification.txt", 'w') as f:
            f.write(first_entry_string)
        print("Full details of the first entry saved to first_entry_verification.txt")
    except Exception as e:
        print(f"\nCould not read verification file: {e}")


def create_batch(solar_features_subset, altitude_range):
    """
    Creates a DataFrame for a batch of timestamps combined with all altitudes.
    """
    # Create the grid of (timestamp x altitude) using a cross merge (cartesian product)
    # This is a highly efficient way to build the base grid for the batch.
    batch_df = pd.DataFrame({
        'DateTimeFormatted': solar_features_subset['DateTimeFormatted'].repeat(len(altitude_range)),
        'Altitude': np.tile(altitude_range, len(solar_features_subset))
    })

    # Merge the solar features into the expanded grid
    # This broadcasts the features for each timestamp across all its altitudes.
    batch_df = pd.merge(batch_df, solar_features_subset, on='DateTimeFormatted', how='left')

    # Add calculated and fixed value columns using vectorized operations
    batch_df['GLAT'] = calculate_glat(batch_df['Altitude'].values)
    for col, val in FIXED_VALUES.items():
        batch_df[col] = val

    # Ensure final column order matches the required schema
    all_cols = input_columns + ['DateTimeFormatted']
    # Filter to only include columns that actually exist in our dataframe
    final_cols_order = [col for col in all_cols if col in batch_df.columns]
    
    return batch_df[final_cols_order]


if __name__ == "__main__":
    # You may need to install pyarrow: pip install pyarrow
    main()