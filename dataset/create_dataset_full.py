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

akebono_file_path = '../../data/Akebono_combined.tsv'
df = pd.read_csv(akebono_file_path, sep='\t')

# Convert DateFormatted to datetime
df['DateFormatted'] = pd.to_datetime(df['DateFormatted'], errors='coerce')

initial_row_count = len(df)
print(f"Initial number of rows: {initial_row_count}\n")

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
filtered_df['DateTimeFormatted'] = pd.to_datetime(filtered_df['DateFormatted'].dt.strftime('%Y-%m-%d') + ' ' + filtered_df['TimeFormatted'].astype(str), errors='coerce')

# Round datetime to nearest minute
filtered_df['DateTimeFormatted'] = filtered_df['DateTimeFormatted'].dt.floor('T')

# Drop redundant date and time columns
filtered_df = filtered_df.drop(columns=['DateFormatted', 'TimeFormatted', 'Date', 'Time'], errors='ignore')

if 'AL_index' not in filtered_df.columns:
    filtered_df['AL_index'] = np.nan
if 'SYM_H' not in filtered_df.columns:
    filtered_df['SYM_H'] = np.nan

# # -----------------------------------
# # Add magnetic field, plasma, solar indices, particle data
# # -----------------------------------

# # Parse into datetime object
# filtered_df['DateTimeFormatted'] = pd.to_datetime(filtered_df['DateTimeFormatted'], format='%Y-%m-%d %H:%M:%S')

# # Read the single OMNI data file
# omni_file_path = '../../data/omni_magnetic_plasma_solar/omni2_K1CK9l7koS.lst'

# # Define column names
# columns = [
#     'Year', 'Day', 'Hour', 
#     'Mag_Scalar_B', 'Mag_Vector_B', 'Mag_B_Lat_GSE', 'Mag_B_Long_GSE',
#     'Mag_BX_GSE', 'Mag_BY_GSE', 'Mag_BZ_GSE', 'Mag_BY_GSM', 'Mag_BZ_GSM', 
#     'Mag_RMS_Mag', 'Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE',
#     'Plasma_SW_Temp', 'Plasma_SW_Density', 'Plasma_SW_Speed',
#     'Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 
#     'Plasma_Sigma_T', 'Plasma_Sigma_N', 'Plasma_Sigma_V', 
#     'Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio',
#     'Solar_Kp', 'Solar_R_Sunspot', 'Solar_Dst', 'Solar_Ap', 
#     'Solar_AE', 'Solar_AL', 'Solar_AU', 'Solar_PC',
#     'Solar_Lyman_Alpha',
#     'Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 
#     'Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 
#     'Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV',
#     'Particle_Flux_Flag'
# ]

# # Read the OMNI data file
# omni_df = pd.read_csv(omni_file_path, delim_whitespace=True, names=columns)

# # Create the 'DateTime' column from Year, Day of Year, Hour, and Minute
# omni_df['DateTime'] = pd.to_datetime(omni_df['Year'] * 1000 + omni_df['Day'], format='%Y%j') \
#                  + pd.to_timedelta(omni_df['Hour'], unit='h')

# # Keep only the necessary columns
# omni_df = omni_df[['DateTime', 'Mag_Scalar_B', 'Mag_Vector_B', 'Mag_B_Lat_GSE', 'Mag_B_Long_GSE',
#          'Mag_BX_GSE', 'Mag_BY_GSE', 'Mag_BZ_GSE', 'Mag_BY_GSM', 'Mag_BZ_GSM', 
#          'Mag_RMS_Mag', 'Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE',
#          'Plasma_SW_Temp', 'Plasma_SW_Density', 'Plasma_SW_Speed',
#          'Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 
#          'Plasma_Sigma_T', 'Plasma_Sigma_N', 'Plasma_Sigma_V', 
#          'Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio',
#          'Solar_Kp', 'Solar_R_Sunspot', 'Solar_Dst', 'Solar_Ap', 
#          'Solar_AE', 'Solar_AL', 'Solar_AU', 'Solar_PC',
#          'Solar_Lyman_Alpha',
#          'Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 
#          'Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 
#          'Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV',
#          'Particle_Flux_Flag']]

# # Reset index to make 'DateTime' a column
# omni_df.reset_index(drop=True, inplace=True)
# omni_df = omni_df.drop_duplicates(subset='DateTime', keep='first')
# omni_df.set_index('DateTime', inplace=True)
# filtered_df['DateTimeFormatted_copy'] = filtered_df['DateTimeFormatted']
# filtered_df['DateTimeFormatted_copy'] = filtered_df['DateTimeFormatted_copy'].dt.round('H')
# filtered_df.set_index('DateTimeFormatted_copy', inplace=True)
# filtered_df.sort_index(inplace=True)

# omni_df.sort_index(inplace=True)

# # Instantaneous
# # for col in ['Mag_Scalar_B', 'Mag_Vector_B', 'Mag_B_Lat_GSE', 'Mag_B_Long_GSE',
# #             'Mag_BX_GSE', 'Mag_BY_GSE', 'Mag_BZ_GSE', 'Mag_BY_GSM', 'Mag_BZ_GSM', 
# #             'Mag_RMS_Mag', 'Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE',
# #             'Plasma_SW_Temp', 'Plasma_SW_Density', 'Plasma_SW_Speed',
# #             'Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 
# #             'Plasma_Sigma_T', 'Plasma_Sigma_N', 'Plasma_Sigma_V', 
# #             'Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio',
# #             'Solar_Kp', 'Solar_R_Sunspot', 'Solar_Dst', 'Solar_Ap', 
# #             'Solar_AE', 'Solar_AL', 'Solar_AU', 'Solar_PC',
# #             'Solar_Lyman_Alpha',
# #             'Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 
# #             'Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 
# #             'Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV', 'Particle_Flux_Flag']:
# #     if col not in filtered_df.columns:
# #         filtered_df[col] = pd.Series(np.nan, index=filtered_df.index)
# #     filtered_df[col] = filtered_df[col].fillna(omni_df[col])

# # Temporal
# low_res_time_range = pd.timedelta_range(start='0D', end='3D', freq='1H')
# dt_index = pd.DatetimeIndex(filtered_df.index)
# low_res_timestamps = dt_index.values[:, None] - low_res_time_range.values

# new_columns = {}
# for col in tqdm(['Mag_Scalar_B', 'Mag_Vector_B', 'Mag_B_Lat_GSE', 'Mag_B_Long_GSE',
#             'Mag_BX_GSE', 'Mag_BY_GSE', 'Mag_BZ_GSE', 'Mag_BY_GSM', 'Mag_BZ_GSM', 
#             'Mag_RMS_Mag', 'Mag_RMS_Vector', 'Mag_RMS_BX_GSE', 'Mag_RMS_BY_GSE', 'Mag_RMS_BZ_GSE',
#             'Plasma_SW_Temp', 'Plasma_SW_Density', 'Plasma_SW_Speed',
#             'Plasma_SW_Flow_Long', 'Plasma_SW_Flow_Lat', 'Plasma_Alpha_Prot_Ratio', 
#             'Plasma_Sigma_T', 'Plasma_Sigma_N', 'Plasma_Sigma_V', 
#             'Plasma_Sigma_Phi_V', 'Plasma_Sigma_Theta_V', 'Plasma_Sigma_Ratio',
#             'Solar_Kp', 'Solar_R_Sunspot', 'Solar_Dst', 'Solar_Ap', 
#             'Solar_AE', 'Solar_AL', 'Solar_AU', 'Solar_PC',
#             'Solar_Lyman_Alpha',
#             'Particle_Proton_Flux_1MeV', 'Particle_Proton_Flux_2MeV', 
#             'Particle_Proton_Flux_4MeV', 'Particle_Proton_Flux_10MeV', 
#             'Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV', 'Particle_Flux_Flag'], desc="Expanding low res columns temporally"):
#     values = omni_df[col].reindex(pd.DatetimeIndex(low_res_timestamps.ravel())).values.reshape(low_res_timestamps.shape)
#     for i in range(values.shape[1]):
#         new_columns[f'{col}_{i}'] = values[:, i]
#     missing_values = np.isnan(values).any(axis=1)
#     if missing_values.any():
#         print("Some values are missing:")
#         print(f"Missing {col}: {missing_values.sum()} rows")

# # Add all new columns at once
# filtered_df = pd.concat([filtered_df, pd.DataFrame(new_columns, index=filtered_df.index)], axis=1)

# Reset the index if you want to keep 'DateTimeFormatted' as a column
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.set_index('DateTimeFormatted', inplace=True)

# -----------------------------------
# 1 Min data
# -----------------------------------

# # More 1 min data
# # Read all .lst files from the directory
# file_list = glob.glob('../../data/omni_1min/*.txt')

# df_list = []

# for file in file_list:
#     # Define column names
#     columns = ['Year', 'Day', 'Hour', 'Minute',
#                'BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE',
#                'AE_index', 'AU_index', 'SYM_D', 'ASY_D', 'ASY_H', 'PCN_index']
    
#     # Read the OMNI data file
#     df = pd.read_csv(file, delim_whitespace=True, names=columns)
    
#     # Create the 'DateTime' column from Year, Day of Year, Hour, and Minute
#     df['DateTime'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j') \
#                      + pd.to_timedelta(df['Hour'], unit='h') \
#                      + pd.to_timedelta(df['Minute'], unit='m')
    
#     # Keep only the necessary columns
#     df = df[['DateTime', 'BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE',
#                'AE_index', 'AU_index', 'SYM_D', 'ASY_D', 'ASY_H', 'PCN_index']]
    
#     df_list.append(df)
# # Concatenate all OMNI DataFrames into one
# omni_df = pd.concat(df_list, ignore_index=True)

# # Reset index to make 'DateTime' a column
# omni_df.reset_index(drop=True, inplace=True)
# omni_df = omni_df.drop_duplicates(subset='DateTime', keep='first')
# omni_df.set_index('DateTime', inplace=True)
# filtered_df.sort_index(inplace=True)
# omni_df.sort_index(inplace=True)

# # Instantaneous
# # for col in ['BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE', 'AE_index', 'AU_index', 'SYM_D',
# #        'ASY_D', 'ASY_H', 'PCN_index']:
# #     if col not in filtered_df.columns:
# #         filtered_df[col] = pd.Series(np.nan, index=filtered_df.index)
# #     filtered_df[col] = filtered_df[col].fillna(omni_df[col])
# #     print(f"{col}: {filtered_df[col].min()} to {filtered_df[col].max()}")


# # Temporal
# high_res_time_range = pd.timedelta_range(start='0D', end='3D', freq='30min')
# dt_index = pd.DatetimeIndex(filtered_df.index)
# high_res_timestamps = dt_index.values[:, None] - high_res_time_range.values

# new_columns = {}
# for col in tqdm(['BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE', 'AE_index', 'AU_index', 'SYM_D', 'ASY_D', 'ASY_H', 'PCN_index'], desc="Expanding high res columns temporally"):
#     values = omni_df[col].reindex(pd.DatetimeIndex(high_res_timestamps.ravel())).values.reshape(high_res_timestamps.shape)
#     for i in range(values.shape[1]):
#         new_columns[f'{col}_{i}'] = values[:, i]
#     missing_values = np.isnan(values).any(axis=1)
#     if missing_values.any():
#         print("Some values are missing:")
#         print(f"Missing {col}: {missing_values.sum()} rows")

# # Add all new columns at once
# filtered_df = pd.concat([filtered_df, pd.DataFrame(new_columns, index=filtered_df.index)], axis=1)

# SYMH
# Read all .lst files from the directory
file_list = glob.glob('../../data/omni_al_index_symh/*.lst')

df_list = []

for file in file_list:
    # Define column names
    columns = ['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H']
    
    # Read the OMNI data file
    df = pd.read_csv(file, delim_whitespace=True, names=columns)
    
    # Create the 'DateTime' column from Year, Day of Year, Hour, and Minute
    df['DateTime'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j') \
                     + pd.to_timedelta(df['Hour'], unit='h') \
                     + pd.to_timedelta(df['Minute'], unit='m')
    
    # Keep only the necessary columns
    df = df[['DateTime', 'AL_index', 'SYM_H']]
    
    df_list.append(df)

omni_df = pd.concat(df_list, ignore_index=True)
omni_df.reset_index(drop=True, inplace=True)
omni_df = omni_df.drop_duplicates(subset='DateTime', keep='first')
omni_df.set_index('DateTime', inplace=True)
filtered_df.sort_index(inplace=True)
omni_df.sort_index(inplace=True)
aligned_omni = omni_df.reindex(filtered_df.index)

# Instantaneous
# for col in ['AL_index', 'SYM_H']:
#     filtered_df[col] = filtered_df[col].fillna(aligned_omni[col])


# Temporal
al_time_range = pd.timedelta_range(start='0S', end='5H', freq='10T')
sym_h_time_range = pd.timedelta_range(start='0D', end='3D', freq='30min')
dt_index = pd.DatetimeIndex(filtered_df.index)
al_timestamps = dt_index.values[:, None] - al_time_range.values
sym_h_timestamps = dt_index.values[:, None] - sym_h_time_range.values
al_values = omni_df['AL_index'].reindex(pd.DatetimeIndex(al_timestamps.ravel())).values.reshape(al_timestamps.shape)
sym_h_values = omni_df['SYM_H'].reindex(pd.DatetimeIndex(sym_h_timestamps.ravel())).values.reshape(sym_h_timestamps.shape)

# Expand AL_index values into separate columns
for i in tqdm(range(al_values.shape[1]), desc="AL_index columns"):
    filtered_df[f'AL_index_{i}'] = al_values[:, i]

# Expand SYM_H values into separate columns
for i in tqdm(range(sym_h_values.shape[1]), desc="SYM_H columns"):
    filtered_df[f'SYM_H_{i}'] = sym_h_values[:, i]

# Remove the original list columns
filtered_df = filtered_df.drop(columns=['AL_index', 'SYM_H'])

missing_al = np.isnan(al_values).any(axis=1)
missing_sym_h = np.isnan(sym_h_values).any(axis=1)

if missing_al.any() or missing_sym_h.any():
    print("Some values are missing:")
    print(f"Missing AL index: {missing_al.sum()} rows")
    print(f"Missing SYM/H: {missing_sym_h.sum()} rows")
    import IPython; IPython.embed()

# -----------------------------------
# F 107 Index
# -----------------------------------
f107_file_list = glob.glob('../../data/omni_f107/*.lst')
f107_list = []

for file in f107_file_list:
    columns = ['Year', 'Day', 'Hour', 'f107_index']
    f107_df = pd.read_csv(file, delim_whitespace=True, names=columns)
    f107_df['DateTime'] = pd.to_datetime(f107_df['Year'] * 1000 + f107_df['Day'], format='%Y%j') \
                          + pd.to_timedelta(f107_df['Hour'], unit='h')
    f107_df = f107_df[['DateTime', 'f107_index']]
    f107_list.append(f107_df)

f107_df_combined = pd.concat(f107_list, ignore_index=True)

f107_df_combined = f107_df_combined.drop_duplicates(subset='DateTime', keep='first')
filtered_df.sort_index(inplace=True)
f107_df_combined.sort_values('DateTime', inplace=True)
f107_df_combined.set_index('DateTime', inplace=True)

# Instantaneous
# filtered_df = pd.merge_asof(
#     filtered_df.reset_index(),
#     f107_df_combined,
#     left_on='DateTimeFormatted',
#     right_on='DateTime',
#     direction='nearest',
#     tolerance=pd.Timedelta('1H')
# )

# Temporal
f107_time_range = pd.timedelta_range(start='0s', end='3d', freq='1d')
f107_timestamps = filtered_df.index.values[:, None] - f107_time_range.values
f107_timestamps_rounded = pd.DatetimeIndex(f107_timestamps.ravel()).round('h')
f107_values = f107_df_combined['f107_index'].reindex(f107_timestamps_rounded).values.reshape(f107_timestamps.shape)

# Expand f107_index values into separate columns
for i in range(f107_values.shape[1]):
    filtered_df[f'f107_index_{i}'] = f107_values[:, i]

if np.isnan(f107_values).any():
    print("Warning: Some f107 index values are missing.") 
    import IPython; IPython.embed()


# Replace all invalid values with 0
invalid_values = [999.9, 9.999, 9999.0, 9999.99, 99999.99, 9999999, 9999999.0]
filtered_df = filtered_df.replace(invalid_values, 0)

# Save dataset
train_val, test = train_test_split(filtered_df, test_size=100000, random_state=42)
train, val = train_test_split(train_val, test_size=100000, random_state=42)

# Save train in chunks cos its too big
num_chunks = math.ceil(len(train) / 100000)
for i in tqdm(range(num_chunks)):
    start_idx = i * 100000
    end_idx = min((i + 1) * 100000, len(train))
    chunk = train.iloc[start_idx:end_idx]
    
    # Convert the chunk to a PyArrow table
    table = pa.Table.from_pandas(chunk)
    
    # Create a dataset from the PyArrow table
    dataset = datasets.Dataset(table)
    
    # Save the dataset as a separate shard
    dataset.save_to_disk(
        f'../../data/akebono_solar_combined_v6_chu_train/chunk_{i:03d}',
        num_shards=1
    )
print(f"Train dataset saved in {num_chunks} separate chunks.")

# Convert val and test DataFrames to Hugging Face datasets
val_dataset = datasets.Dataset.from_pandas(val)
test_dataset = datasets.Dataset.from_pandas(test)

# Save validation and test datasets to disk
val_dataset.save_to_disk('../../data/akebono_solar_combined_v6_chu_val')
test_dataset.save_to_disk('../../data/akebono_solar_combined_v6_chu_test')

print("Validation and test datasets converted to Hugging Face datasets.")
