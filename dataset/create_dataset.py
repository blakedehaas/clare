import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import calmap
from scipy import stats
from sklearn.linear_model import LinearRegression

def print_rows_removed(before_count, after_df, step_description, column_to_check=None):
    after_count = len(after_df)
    rows_removed = before_count - after_count
    print(f"{step_description}:")
    print(f"Rows removed: {rows_removed}")
    print(f"Rows remaining: {after_count}")
    
    # If a column is provided, print the new range of that column
    if column_to_check:
        min_value = after_df[column_to_check].min()
        max_value = after_df[column_to_check].max()
        print(f"New range of '{column_to_check}': {min_value} to {max_value}")
    
    print("\n")
    return after_count

# # Load the dataset
# akebono_file_path = '../data/Akebono_combined.tsv'
# df = pd.read_csv(akebono_file_path, sep='\t')

# # Convert DateFormatted to datetime if it's not already
# df['DateFormatted'] = pd.to_datetime(df['DateFormatted'], errors='coerce')

# # Initial row count
# initial_row_count = len(df)
# print(f"Initial number of rows: {initial_row_count}\n")

# # -----------------------------------
# # 1. Drop Unnecessary Columns during training
# # -----------------------------------
# # This step can be done when training the model, specify output and input columns
# # -----------------------------------
# # 2. Remove Rows with '999' in Specified Columns
# # -----------------------------------
# columns_with_999 = ['XXLAT', 'XXLON']
# mask_999 = (df[columns_with_999] == 999).any(axis=1)
# filtered_df = df[~mask_999]

# # Check rows removed after step 2 (checking 'GLAT' and 'GMLT' range)
# initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After removing rows with '999' values", column_to_check='GLAT')

# # -----------------------------------
# # 3. Remove Rows Greater Than '90' in 'ILAT'
# # -----------------------------------
# filtered_df = filtered_df[filtered_df['ILAT'] <= 90]

# # Check rows removed after step 3 (checking 'ILAT' range)
# initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After removing rows with ILAT > 90", column_to_check='ILAT')

# # -----------------------------------
# # 4. Filter Data Between 1000km and 8000km Altitude
# # -----------------------------------
# filtered_df = filtered_df[(filtered_df['Altitude'] >= 1000) & (filtered_df['Altitude'] <= 8000)]

# # Check rows removed after step 4 (checking 'Altitude' range)
# initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After filtering Altitude between 1000km and 8000km", column_to_check='Altitude')

# # -----------------------------------
# # 5. Filter Out Rows Before January 1st, 1990
# # -----------------------------------
# filtered_df['DateFormatted'] = pd.to_datetime(filtered_df['DateFormatted'], errors='coerce')
# filtered_df = filtered_df.dropna(subset=['DateFormatted'])
# filtered_df = filtered_df[filtered_df['DateFormatted'] >= '1990-01-01']

# # Check rows removed after step 5 (checking 'DateFormatted' range)
# initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After filtering rows before January 1st, 1990", column_to_check='DateFormatted')

# # -----------------------------------
# # 6. Combine 'DateFormatted' and 'TimeFormatted' into 'DateTimeFormatted'
# # -----------------------------------
# # Create the 'DateTimeFormatted' column by combining 'DateFormatted' and 'TimeFormatted' columns
# filtered_df['DateTimeFormatted'] = pd.to_datetime(filtered_df['DateFormatted'].dt.strftime('%Y-%m-%d') + ' ' + filtered_df['TimeFormatted'].astype(str), errors='coerce')

# # Round the timestamps in filtered_df to the nearest minute to avoid precision issues
# filtered_df['DateTimeFormatted'] = filtered_df['DateTimeFormatted'].dt.floor('T')

# # Drop the old 'DateFormatted' and 'TimeFormatted' columns
# filtered_df = filtered_df.drop(columns=['DateFormatted', 'TimeFormatted', 'Date', 'Time'], errors='ignore')

# # Check the first few rows after combining
# print("After creating DateTimeFormatted column:")
# print(filtered_df[['DateTimeFormatted']].head())

# # -----------------------------------
# # 7. Initialize 'AL_index' and 'SYM_H' 
# # -----------------------------------
# if 'AL_index' not in filtered_df.columns:
#     filtered_df['AL_index'] = np.nan
# if 'SYM_H' not in filtered_df.columns:
#     filtered_df['SYM_H'] = np.nan

# # -----------------------------------
# # Final check and summary
# # -----------------------------------
# final_row_count = len(filtered_df)
# print(f"Final number of rows: {final_row_count}")
# print(filtered_df.head())


# Load the dataset
akebono_file_path = '1.tsv'
filtered_df = pd.read_csv(akebono_file_path, sep='\t')

# # Parse into datetime object
filtered_df['DateTimeFormatted'] = pd.to_datetime(filtered_df['DateTimeFormatted'], format='%Y-%m-%d %H:%M:%S')

# # Read the single OMNI data file
# omni_file_path = '../data/omni_magnetic_plasma_solar/omni2_K1CK9l7koS.lst'

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
# filtered_df['DateTimeFormatted'] = filtered_df['DateTimeFormatted'].dt.round('H')
# filtered_df.set_index('DateTimeFormatted', inplace=True)
# filtered_df.sort_index(inplace=True)
# omni_df.sort_index(inplace=True)
# for col in omni_df.columns:
#     if col not in ['DateTime', 'Particle_Flux_Flag']:
#         omni_df[col] = omni_df[col].fillna(omni_df[col])
#         # Check for exact fill value patterns and create existence column
#         fill_values = ['999.9', '9.999', '9999.0', '99999.99','99999.99', '9999999', '9999999.0']
#         omni_df[f'exists_{col}'] = (~omni_df[col].astype(str).isin(fill_values)).astype(int)


# # Now, fill NaN values in 'filtered_df' with corresponding values from 'aligned_omni'
# for col in ['Mag_Scalar_B', 'Mag_Vector_B', 'Mag_B_Lat_GSE', 'Mag_B_Long_GSE',
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
#             'Particle_Proton_Flux_30MeV', 'Particle_Proton_Flux_60MeV', 'Particle_Flux_Flag',
#             'exists_Mag_Scalar_B', 'exists_Mag_Vector_B', 'exists_Mag_B_Lat_GSE', 'exists_Mag_B_Long_GSE',
#             'exists_Mag_BX_GSE', 'exists_Mag_BY_GSE', 'exists_Mag_BZ_GSE', 'exists_Mag_BY_GSM', 'exists_Mag_BZ_GSM', 
#             'exists_Mag_RMS_Mag', 'exists_Mag_RMS_Vector', 'exists_Mag_RMS_BX_GSE', 'exists_Mag_RMS_BY_GSE', 'exists_Mag_RMS_BZ_GSE',
#             'exists_Plasma_SW_Temp', 'exists_Plasma_SW_Density', 'exists_Plasma_SW_Speed',
#             'exists_Plasma_SW_Flow_Long', 'exists_Plasma_SW_Flow_Lat', 'exists_Plasma_Alpha_Prot_Ratio', 
#             'exists_Plasma_Sigma_T', 'exists_Plasma_Sigma_N', 'exists_Plasma_Sigma_V', 
#             'exists_Plasma_Sigma_Phi_V', 'exists_Plasma_Sigma_Theta_V', 'exists_Plasma_Sigma_Ratio',
#             'exists_Solar_Kp', 'exists_Solar_R_Sunspot', 'exists_Solar_Dst', 'exists_Solar_Ap', 
#             'exists_Solar_AE', 'exists_Solar_AL', 'exists_Solar_AU', 'exists_Solar_PC',
#             'exists_Solar_Lyman_Alpha',
#             'exists_Particle_Proton_Flux_1MeV', 'exists_Particle_Proton_Flux_2MeV', 
#             'exists_Particle_Proton_Flux_4MeV', 'exists_Particle_Proton_Flux_10MeV', 
#             'exists_Particle_Proton_Flux_30MeV', 'exists_Particle_Proton_Flux_60MeV']: # 83
#     if col not in filtered_df.columns:
#         filtered_df[col] = pd.Series(np.nan, index=filtered_df.index)
#     filtered_df[col] = filtered_df[col].fillna(omni_df[col])

# # Reset the index if you want to keep 'DateTimeFormatted' as a column
# filtered_df.reset_index(inplace=True)

# # Now, filtered_df has the 'AL_index' and 'SYM_H' columns filled in where data is available
# print(filtered_df.iloc[0].to_dict())




# Read all .lst files from the directory
file_list = glob.glob('../data/omni_1min/*.txt')

df_list = []

for file in file_list:
    # Define column names
    columns = ['Year', 'Day', 'Hour', 'Minute',
               'BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE',
               'AE_index', 'AU_index', 'SYM_D', 'ASY_D', 'ASY_H', 'PCN_index']
    
    # Read the OMNI data file
    df = pd.read_csv(file, delim_whitespace=True, names=columns)
    
    # Create the 'DateTime' column from Year, Day of Year, Hour, and Minute
    df['DateTime'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j') \
                     + pd.to_timedelta(df['Hour'], unit='h') \
                     + pd.to_timedelta(df['Minute'], unit='m')
    
    # Keep only the necessary columns
    df = df[['DateTime', 'BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE',
               'AE_index', 'AU_index', 'SYM_D', 'ASY_D', 'ASY_H', 'PCN_index']]
    
    df_list.append(df)
# Concatenate all OMNI DataFrames into one
omni_df = pd.concat(df_list, ignore_index=True)

# Reset index to make 'DateTime' a column
omni_df.reset_index(drop=True, inplace=True)
omni_df = omni_df.drop_duplicates(subset='DateTime', keep='first')
omni_df.set_index('DateTime', inplace=True)
filtered_df.set_index('DateTimeFormatted', inplace=True)
filtered_df.sort_index(inplace=True)
omni_df.sort_index(inplace=True)
for col in omni_df.columns:
    if col not in ['DateTime']:
        omni_df[col] = omni_df[col].fillna(omni_df[col])
        # Check for exact fill value patterns and create existence column
        fill_values = ['999.9', '9.999', '9999.0', '9999.99', '99999.99','99999.99', '9999999', '9999999.0']
        omni_df[f'exists_{col}'] = (~omni_df[col].astype(str).isin(fill_values)).astype(int)
        

# Now, fill NaN values in 'filtered_df' with corresponding values from 'aligned_omni'
for col in ['BSN_X_GSE', 'BSN_Y_GSE', 'BSN_Z_GSE', 'AE_index', 'AU_index', 'SYM_D',
       'ASY_D', 'ASY_H', 'PCN_index', 'exists_BSN_X_GSE', 'exists_BSN_Y_GSE',
       'exists_BSN_Z_GSE', 'exists_AE_index', 'exists_AU_index',
       'exists_SYM_D', 'exists_ASY_D', 'exists_ASY_H', 'exists_PCN_index']:
    if col not in filtered_df.columns:
        filtered_df[col] = pd.Series(np.nan, index=filtered_df.index)
    filtered_df[col] = filtered_df[col].fillna(omni_df[col])
    print(f"{col}: {filtered_df[col].min()} to {filtered_df[col].max()}")

# Reset the index if you want to keep 'DateTimeFormatted' as a column
filtered_df.reset_index(inplace=True)