import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
import calmap
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datasets
import math
from datasets import Dataset
import pyarrow as pa

def print_rows_removed(before_count, after_df, step_description, column_to_check=None):
    after_count = len(after_df)
    rows_removed = before_count - after_count
    print(f"{step_description}:")
    print(f"Rows removed: {rows_removed}")
    print(f"Rows remaining: {after_count}")
    
    if column_to_check:
        min_value = after_df[column_to_check].min()
        max_value = after_df[column_to_check].max()
        print(f"New range of '{column_to_check}': {min_value} to {max_value}")
    
    print("\n")
    return after_count

# Define updated relative paths
akebono_file_path = '../../data/Akebono_combined.tsv'
omni_al_symh_path = '../../data/omni_al_index_symh/*.lst'
f107_file_path = '../../data/omni_f107/*.lst'
train_output_path = '../../data/akebono_solar_combined_v6_chu_train'
val_output_path = '../../data/akebono_solar_combined_v6_chu_val'
test_output_path = '../../data/akebono_solar_combined_v6_chu_test'

# Read the Akebono data in chunks to optimize memory usage
chunk_size = 500000  # Adjust this based on available memory
chunks = []
initial_row_count = 0

print("Reading Akebono data in chunks...")
for chunk in tqdm(pd.read_csv(akebono_file_path, sep='\t', chunksize=chunk_size), desc="Loading Akebono data"):
    # Convert DateFormatted to datetime
    chunk['DateFormatted'] = pd.to_datetime(chunk['DateFormatted'], errors='coerce')
    initial_row_count += len(chunk)
    chunks.append(chunk)

# Concatenate all chunks
df = pd.concat(chunks, ignore_index=True)
del chunks  # Free up memory

print(f"\nInitial number of rows: {initial_row_count}\n")

# Data Cleaning Steps with Progress Bars
print("Starting data cleaning steps...")

# Remove rows with '999' values in XXLAT and XXLON
columns_with_999 = ['XXLAT', 'XXLON']
mask_999 = (df[columns_with_999] == 999).any(axis=1)
filtered_df = df[~mask_999]

initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After removing rows with '999' values", column_to_check='GLAT')

# Remove rows with ILAT > 90
filtered_df = filtered_df[filtered_df['ILAT'] <= 90]

initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After removing rows with ILAT > 90", column_to_check='ILAT')

# Filter Altitude between 1000km and 8000km
filtered_df = filtered_df[(filtered_df['Altitude'] >= 1000) & (filtered_df['Altitude'] <= 8000)]

initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After filtering Altitude between 1000km and 8000km", column_to_check='Altitude')

# Convert DateFormatted to datetime, remove NaT values, and filter rows before 1990-01-01
filtered_df['DateFormatted'] = pd.to_datetime(filtered_df['DateFormatted'], errors='coerce')
filtered_df = filtered_df.dropna(subset=['DateFormatted'])
filtered_df = filtered_df[filtered_df['DateFormatted'] >= '1990-01-01']

initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After filtering rows before January 1st, 1990", column_to_check='DateFormatted')

# Combine date and time into a single datetime column
filtered_df['DateTimeFormatted'] = pd.to_datetime(
    filtered_df['DateFormatted'].dt.strftime('%Y-%m-%d') + ' ' + filtered_df['TimeFormatted'].astype(str),
    errors='coerce'
)

# Round datetime to nearest minute
filtered_df['DateTimeFormatted'] = filtered_df['DateTimeFormatted'].dt.floor('min')  # Changed 'T' to 'min'

# Drop redundant date and time columns
filtered_df = filtered_df.drop(columns=['DateFormatted', 'TimeFormatted', 'Date', 'Time'], errors='ignore')

# Ensure 'AL_index' and 'SYM_H' columns exist
for col in ['AL_index', 'SYM_H']:
    if col not in filtered_df.columns:
        filtered_df[col] = np.nan

# Reset the index and set 'DateTimeFormatted' as the index
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.set_index('DateTimeFormatted', inplace=True)

# -----------------------------------
# SYM-H and AL Index Data
# -----------------------------------
print("Processing SYM-H and AL index data...")

# Read all .lst files from the directory
file_list = glob.glob(omni_al_symh_path)
df_list = []

for file in tqdm(file_list, desc="Loading omni_al_index_symh data"):
    # Define column names
    columns = ['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H']
    
    # Read the OMNI data file with updated sep
    df = pd.read_csv(file, sep='\s+', names=columns, engine='python')
    
    # Create the 'DateTime' column
    df['DateTime'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j') \
                     + pd.to_timedelta(df['Hour'], unit='h') \
                     + pd.to_timedelta(df['Minute'], unit='m')
    
    # Keep only the necessary columns
    df = df[['DateTime', 'AL_index', 'SYM_H']]
    
    df_list.append(df)

# Concatenate and process the OMNI data
omni_df = pd.concat(df_list, ignore_index=True)
del df_list  # Free up memory

omni_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
omni_df.set_index('DateTime', inplace=True)
omni_df.sort_index(inplace=True)

# Ensure filtered_df is sorted by index
filtered_df.sort_index(inplace=True)

# Temporal feature expansion
print("Expanding temporal features for SYM-H and AL index...")

# Define time ranges with 'min' instead of 'T'
al_time_range = pd.timedelta_range(start='0m', end='5h', freq='10min')       # Changed '10T' to '10min'
sym_h_time_range = pd.timedelta_range(start='0m', end='3d', freq='30min')  # Changed '30T' to '30min'

dt_index = filtered_df.index

# Generate timestamps for AL_index
al_timestamps = dt_index.values[:, None] - al_time_range.values
al_timestamps = pd.DatetimeIndex(al_timestamps.ravel())

# Generate timestamps for SYM_H
sym_h_timestamps = dt_index.values[:, None] - sym_h_time_range.values
sym_h_timestamps = pd.DatetimeIndex(sym_h_timestamps.ravel())

# Reindex OMNI data to align with the timestamps
al_values = omni_df['AL_index'].reindex(al_timestamps).values.reshape(len(dt_index), -1)
sym_h_values = omni_df['SYM_H'].reindex(sym_h_timestamps).values.reshape(len(dt_index), -1)

# Collect new columns with progress bars
print("Creating AL_index temporal features...")
al_columns = {}
for i in tqdm(range(al_values.shape[1]), desc="AL_index columns"):
    al_columns[f'AL_index_{i}'] = al_values[:, i]

print("Creating SYM_H temporal features...")
sym_h_columns = {}
for i in tqdm(range(sym_h_values.shape[1]), desc="SYM_H columns"):
    sym_h_columns[f'SYM_H_{i}'] = sym_h_values[:, i]

# Concatenate all new columns at once to prevent fragmentation
print("Concatenating new temporal features to the DataFrame...")
filtered_df = pd.concat([
    filtered_df, 
    pd.DataFrame(al_columns, index=filtered_df.index),
    pd.DataFrame(sym_h_columns, index=filtered_df.index)
], axis=1)

# Remove the original columns if they exist
filtered_df.drop(columns=['AL_index', 'SYM_H'], inplace=True, errors='ignore')

# Reset DataFrame fragmentation
filtered_df = filtered_df.copy()

# -----------------------------------
# F10.7 Solar Flux Index
# -----------------------------------
print("Processing F10.7 solar flux index data...")

# Read all .lst files from the directory
f107_file_list = glob.glob(f107_file_path)
f107_list = []

for file in tqdm(f107_file_list, desc="Loading f107 data"):
    columns = ['Year', 'Day', 'Hour', 'f107_index']
    f107_df = pd.read_csv(file, sep='\s+', names=columns, engine='python')
    f107_df['DateTime'] = pd.to_datetime(f107_df['Year'] * 1000 + f107_df['Day'], format='%Y%j') \
                          + pd.to_timedelta(f107_df['Hour'], unit='h')
    f107_df = f107_df[['DateTime', 'f107_index']]
    f107_list.append(f107_df)

# Concatenate and process F10.7 data
f107_df_combined = pd.concat(f107_list, ignore_index=True)
del f107_list  # Free up memory

f107_df_combined.drop_duplicates(subset='DateTime', keep='first', inplace=True)
f107_df_combined.set_index('DateTime', inplace=True)
f107_df_combined.sort_index(inplace=True)

# Temporal feature expansion for F10.7
print("Expanding temporal features for F10.7 index...")

f107_time_range = pd.timedelta_range(start='0h', end='72h', freq='24h')  # 3 days with daily frequency
f107_timestamps = dt_index.values[:, None] - f107_time_range.values
f107_timestamps = pd.DatetimeIndex(f107_timestamps.ravel()).round('h')  # Round to the nearest hour

# Reindex F10.7 data
f107_values = f107_df_combined['f107_index'].reindex(f107_timestamps).values.reshape(len(dt_index), -1)

# Collect new columns with progress bar
print("Creating f107_index temporal features...")
f107_columns = {}
for i in tqdm(range(f107_values.shape[1]), desc="f107_index columns"):
    f107_columns[f'f107_index_{i}'] = f107_values[:, i]

# Concatenate new columns
print("Concatenating f107_index temporal features to the DataFrame...")
filtered_df = pd.concat([filtered_df, pd.DataFrame(f107_columns, index=filtered_df.index)], axis=1)

# Reset DataFrame fragmentation
filtered_df = filtered_df.copy()

# -----------------------------------
# Data Cleaning: Replace Invalid Values and Optimize Data Types
# -----------------------------------
print("Cleaning data and optimizing data types...")

# Replace invalid values with 0
invalid_values = [999.9, 9.999, 9999.0, 9999.99, 99999.99, 9999999, 9999999.0]
filtered_df.replace(invalid_values, 0, inplace=True)

# Downcast numerical columns to optimize memory usage
print("Downcasting numerical columns to reduce memory usage...")
for col in tqdm(filtered_df.select_dtypes(include=['float', 'int']).columns, desc="Downcasting columns"):
    filtered_df[col] = pd.to_numeric(filtered_df[col], downcast='float')

# -----------------------------------
# Splitting and Saving the Dataset
# -----------------------------------
print("Splitting the dataset into training, validation, and test sets...")

# Define the validation period for the solar storm (June 2 - June 7, 1991)
validation_start = '1991-06-02'
validation_end = '1991-06-08'  # Exclusive end date to include June 7

# Create a boolean mask for the validation period
val_mask = (filtered_df.index >= validation_start) & (filtered_df.index < validation_end)

# Extract the validation set
val = filtered_df.loc[val_mask]

# Remove the validation data from the main dataset
filtered_df = filtered_df.loc[~val_mask]

# Now split the remaining data into training and test sets
train, test = train_test_split(filtered_df, test_size=100000, random_state=42)
del filtered_df  # Free up memory

# Save train in chunks
num_chunks = math.ceil(len(train) / 100000)
print("Saving training data in chunks...")

os.makedirs(train_output_path, exist_ok=True)

for i in tqdm(range(num_chunks), desc="Saving training chunks"):
    start_idx = i * 100000
    end_idx = min((i + 1) * 100000, len(train))
    chunk = train.iloc[start_idx:end_idx]
    
    # Convert the chunk to a PyArrow table
    table = pa.Table.from_pandas(chunk)
    
    # Create a dataset from the PyArrow table
    dataset = datasets.Dataset(table)
    
    # Save the dataset as a separate shard
    dataset.save_to_disk(
        os.path.join(train_output_path, f'chunk_{i:03d}'),
        num_shards=1
    )

print(f"Train dataset saved in {num_chunks} separate chunks.")

# Save validation and test datasets to disk
print("Saving validation and test datasets...")
os.makedirs(val_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)

# Save validation set
val_dataset = Dataset(pa.Table.from_pandas(val))
val_dataset.save_to_disk(val_output_path)

# Save test set
test_dataset = Dataset(pa.Table.from_pandas(test))
test_dataset.save_to_disk(test_output_path)

print("Validation and test datasets converted to Hugging Face datasets.")
