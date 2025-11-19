# --- GENERAL IMPORTS ---
import os
import sys
import glob
import json
from typing import Dict, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- VISUALIZATION IMPORTS ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving files on servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- SPACEPY IMPORT ---
try:
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
except ImportError:
    print("Error: The 'spacepy' library is required.", flush=True)
    sys.exit(1)

# --- MODEL IMPORTS & PATH CONFIGURATION ---
# Set up paths to import model definition and constants from the project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
sys.path.append(PROJECT_ROOT)
import models.feed_forward as model_definition
import constants

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Path Configuration ---
# Define paths for input data, model checkpoints, and normalization statistics
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'input_dataset')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
KP_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_kp_index.lst')
OMNI_AL_SYMH_PATH = os.path.join(BASE_DATA_DIR, 'omni_al_index_symh', '*.lst')
F107_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_f107', '*.lst')
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "checkpoint.pth")
STATS_PATH = os.path.join(CHECKPOINTS_DIR, "norm_stats.json")

# --- Data Generation Config ---
# Set parameters for generating the synthetic dataset time and space grid using a simple dipole approximation
TIME_START = "1991-01-28 00:00:00"
TIME_END = "1991-02-10 00:00:00"
TIME_INCREMENT = "10min"
ALTITUDE_START_KM = 1000
ALTITUDE_END_KM = 8000
ALTITUDE_INCREMENT_KM = 10
FIXED_MAGNETIC_COORDS = {'L_SHELL': 4.0, 'GMLT': 0.0}
L_shell_val = FIXED_MAGNETIC_COORDS['L_SHELL']
GMLT_val = FIXED_MAGNETIC_COORDS['GMLT']
TIMESTAMPS_PER_BATCH = 500
BATCH_SIZE_PREDICT = 8192

# --- Physics & Model Constants ---
# Define physical constants and model-specific parameters for calculations
RE_KM = 6371.0
DEGREES_PER_HOUR_MLT = 15.0
TEMP_BIN_WIDTH_K = 100
TEMP_BIN_CENTER_OFFSET_K = 50
ILAT_DEG = np.rad2deg(np.arccos(1.0 / np.sqrt(L_shell_val)))

# --- Data Handling Config ---
# Configure thresholds for data quality checks
MISSING_DATA_THRESHOLD_PERCENT = 5.0

# --- Visualization Config ---
# Define settings for the output plots and data columns
COMBINED_PLOT_FILENAME = f"L{L_shell_val}_GMLT_{GMLT_val}_combined_synthetic_visualization.png"
PRED_TEMP_COLUMN = 'Te1_pred'
CONFIDENCE_COLUMN = 'confidence'
SOLAR_FEATURE_COLUMNS = ['Kp_index', 'SYM_H_0', 'f107_index_0', 'AL_index_0']
KP_INDEX_SCALE_FACTOR = 10.0
CONFIDENCE_VMAX_PERCENT = 50.0

# --- Model Input Config ---
# List all feature columns required by the model and define network architecture
input_columns = ['Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index']
input_size = len(input_columns)
hidden_size = 2048
output_size = 150

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def check_file_exists(filepath, description):
    """Verifies that a required file exists and exits if not."""
    # Exit with an error if the specified file path does not exist
    if not os.path.exists(filepath):
        print(f"Error: {description} not found at '{filepath}'", flush=True)
        sys.exit(1)

def load_normalization_stats(stats_path):
    """Loads the mean and standard deviation statistics for data normalization from a JSON file."""
    # Check for the stats file and load the JSON content
    check_file_exists(stats_path, "Normalization stats file")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    # Return mean and standard deviation values
    return stats['mean'], stats['std']

def load_solar_indices(omni_al_symh_path, f107_file_path, kp_file_path):
    """Loads and preprocesses all required solar and geomagnetic index data from OMNI files."""
    print("--- Loading and Processing Solar Index Data ---", flush=True)
    # Verify Kp index file exists before proceeding
    check_file_exists(kp_file_path, "Kp index file")
    
    # Load and process AL and SYM-H index files
    al_symh_files = glob.glob(omni_al_symh_path)
    if not al_symh_files: sys.exit(f"Error: No AL/SYM-H files found matching pattern '{omni_al_symh_path}'")
    df_list = [pd.read_csv(f, sep=r'\s+', names=['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H'], usecols=[0,1,2,3,4,5], na_values=[9999, 99999, 999.9, 99999.99]) for f in tqdm(al_symh_files, desc="Reading AL/SYM-H files")]
    omni_df = pd.concat(df_list, ignore_index=True)
    omni_df['DateTime'] = pd.to_datetime(omni_df['Year'].astype(str) + omni_df['Day'].astype(str).str.zfill(3), format='%Y%j') + pd.to_timedelta(omni_df['Hour'], unit='h') + pd.to_timedelta(omni_df['Minute'], unit='m')
    omni_df = omni_df[['DateTime', 'AL_index', 'SYM_H']].drop_duplicates('DateTime').set_index('DateTime').sort_index()
    
    # Load and process F10.7 index files
    f107_files = glob.glob(f107_file_path)
    if not f107_files: sys.exit(f"Error: No F10.7 files found matching pattern '{f107_file_path}'")
    f107_list = [pd.read_csv(f, sep=r'\s+', names=['Year', 'Day', 'Hour', 'f107_index'], na_values=[999.9]) for f in f107_files]
    f107_df = pd.concat(f107_list, ignore_index=True)
    f107_df['DateTime'] = pd.to_datetime(f107_df['Year'].astype(str) + f107_df['Day'].astype(str).str.zfill(3), format='%Y%j') + pd.to_timedelta(f107_df['Hour'], unit='h')
    f107_df = f107_df[['DateTime', 'f107_index']].drop_duplicates('DateTime').set_index('DateTime').sort_index()
    
    # Load and process Kp index files
    kp_df = pd.read_csv(kp_file_path, sep=r'\s+', names=['Year', 'DOY', 'Hour', 'Kp_index'], na_values=[99.9])
    kp_df.dropna(subset=['Year', 'DOY'], inplace=True)
    kp_df['DateTime'] = pd.to_datetime(kp_df['Year'].astype(int).astype(str) + kp_df['DOY'].astype(int).astype(str).str.zfill(3), format='%Y%j') + pd.to_timedelta(kp_df['Hour'], unit='h')
    kp_df = kp_df[['DateTime', 'Kp_index']].drop_duplicates('DateTime').set_index('DateTime').sort_index()
    print("Solar index data loaded successfully.", flush=True)
    
    # Return the processed dataframes
    return omni_df, f107_df, kp_df

def create_temporal_solar_features(target_times: pd.DatetimeIndex, omni_df: pd.DataFrame, f107_df: pd.DataFrame, kp_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame of time-lagged solar features for each target timestamp."""
    all_features = {}
    # Generate time-lagged AL index features by reindexing
    al_time_lags = pd.timedelta_range(start='0m', end='5h', freq='10min'); past_al_timestamps = target_times.values[:, None] - al_time_lags.values
    flat_al_values = omni_df['AL_index'].reindex(past_al_timestamps.ravel()).values; lagged_al_values = flat_al_values.reshape(past_al_timestamps.shape)
    for i in range(lagged_al_values.shape[1]): all_features[f'AL_index_{i}'] = lagged_al_values[:, i]
    
    # Generate time-lagged SYM-H index features
    sym_h_time_lags = pd.timedelta_range(start='0m', end='3d', freq='30min'); past_sym_h_timestamps = target_times.values[:, None] - sym_h_time_lags.values
    flat_sym_h_values = omni_df['SYM_H'].reindex(past_sym_h_timestamps.ravel()).values; lagged_sym_h_values = flat_sym_h_values.reshape(past_sym_h_timestamps.shape)
    for i in range(lagged_sym_h_values.shape[1]): all_features[f'SYM_H_{i}'] = lagged_sym_h_values[:, i]
    
    # Generate time-lagged F10.7 index features with nearest-neighbor lookup
    f107_time_lags = pd.timedelta_range(start='0h', end='72h', freq='24h'); past_f107_timestamps = target_times.values[:, None] - f107_time_lags.values
    rounded_f107_timestamps = pd.DatetimeIndex(past_f107_timestamps.ravel()).round('h'); flat_f107_values = f107_df['f107_index'].reindex(rounded_f107_timestamps, method='nearest').values
    lagged_f107_values = flat_f107_values.reshape(past_f107_timestamps.shape)
    for i in range(lagged_f107_values.shape[1]): all_features[f'f107_index_{i}'] = lagged_f107_values[:, i]
    
    # Add the nearest Kp index value
    all_features['Kp_index'] = kp_df['Kp_index'].reindex(target_times.round('h'), method='nearest').values
    
    # Return a DataFrame containing all generated features
    return pd.DataFrame(all_features, index=target_times)

def calculate_spacecraft_coords(df: pd.DataFrame, L_shell: float, mlt_hours: float) -> pd.DataFrame:
    """Calculates altitude-dependent spacecraft coordinates (XXLAT, XXLON, GLAT)."""
    # Prepare time and position inputs for spacepy
    times_iso = df['DateTimeFormatted'].dt.strftime('%Y-%m-%dT%H:%M:%S').to_list()
    times = Ticktock(times_iso, 'ISO')
    radius_re = (df['Altitude'].values + RE_KM) / RE_KM
    
    # Calculate spacecraft position in Solar Magnetic (SM) coordinates
    mlt_rad = np.deg2rad(mlt_hours * DEGREES_PER_HOUR_MLT)
    mag_lat_rad = np.arccos(np.sqrt(np.clip(radius_re / L_shell, 0, 1)))
    x_sm = radius_re * np.cos(mag_lat_rad) * np.cos(mlt_rad)
    y_sm = radius_re * np.cos(mag_lat_rad) * np.sin(mlt_rad)
    z_sm = radius_re * np.sin(mag_lat_rad)
    
    # Convert coordinates from SM to Geographic (GEO) and Geomagnetic (MAG)
    sc_pos_sm = coord.Coords(np.column_stack([x_sm, y_sm, z_sm]), 'SM', 'car', ticks=times)
    sc_pos_geo = sc_pos_sm.convert('GEO', 'sph')
    sc_pos_mag = sc_pos_geo.convert('MAG', 'sph')
    
    # Return a DataFrame with the calculated coordinates
    return pd.DataFrame({'XXLAT': sc_pos_geo.lati, 'XXLON': sc_pos_geo.long, 'GLAT': sc_pos_mag.lati}, index=df.index)

def calculate_footprint_coords(timestamps: pd.DatetimeIndex, L_shell: float, mlt_hours: float) -> pd.DataFrame:
    """Calculates altitude-independent footprint coordinates (GCLAT, GCLON) for unique timestamps."""
    # Prepare time inputs for spacepy
    times_iso = timestamps.strftime('%Y-%m-%dT%H:%M:%S').to_list()
    times = Ticktock(times_iso, 'ISO')
    
    # Calculate the magnetic field line footprint in SM coordinates
    mlt_rad = np.deg2rad(mlt_hours * DEGREES_PER_HOUR_MLT)
    ilat_rad = np.arccos(1.0 / np.sqrt(L_shell))
    x_sm_fp = np.cos(ilat_rad) * np.cos(mlt_rad)
    y_sm_fp = np.cos(ilat_rad) * np.sin(mlt_rad)
    z_sm_fp = np.sin(ilat_rad)
    
    # Create an array of identical vectors, one for each unique timestamp
    fp_vectors = np.tile([x_sm_fp, y_sm_fp, z_sm_fp], (len(timestamps), 1))
    
    # Convert footprint coordinates from SM to GEO
    fp_pos_sm = coord.Coords(fp_vectors, 'SM', 'car', ticks=times)
    fp_pos_geo = fp_pos_sm.convert('GEO', 'sph')
    
    # Return a DataFrame with footprint coordinates
    return pd.DataFrame({'DateTimeFormatted': timestamps, 'GCLAT': fp_pos_geo.lati, 'GCLON': fp_pos_geo.long})

def handle_missing_data(df: pd.DataFrame, required_cols: List[str], threshold: float) -> pd.DataFrame:
    """Checks for, warns about, and handles missing data in the dataframe."""
    # Identify columns with any missing values
    nan_counts = df[required_cols].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    
    # Return the original dataframe if no NaNs are found
    if cols_with_nan.empty: return df
    
    # Report on columns that have missing data
    print("\n--- WARNING: Missing data detected before processing ---", flush=True)
    total_rows = len(df); rows_with_nan = df[cols_with_nan.index].isnull().any(axis=1)
    num_rows_to_drop = rows_with_nan.sum()
    for col, count in cols_with_nan.items():
        percentage = (count / total_rows) * 100
        print(f"  - Column '{col}' has {count} missing values ({percentage:.2f}%).", flush=True)
        # Exit if the percentage of missing data exceeds the defined threshold
        if percentage > threshold: sys.exit(f"  - ERROR: Missing data in '{col}' ({percentage:.2f}%) exceeds threshold of {threshold}%.")
    
    # Report the number of rows being dropped and the first occurrence of missing data
    print(f"Dropping {num_rows_to_drop} rows ({num_rows_to_drop/total_rows*100:.2f}%) due to missing values.", flush=True)
    first_missing_time = df.loc[rows_with_nan, 'DateTimeFormatted'].min()
    print(f"First occurrence of missing data is at: {first_missing_time}", flush=True)
    
    # Drop rows with any NaN values in the specified columns and return the cleaned dataframe
    return df.dropna(subset=cols_with_nan.index).reset_index(drop=True)

def preprocess_data(df: pd.DataFrame, means: Dict, stds: Dict) -> pd.DataFrame:
    """Applies preprocessing steps like normalization to the input DataFrame."""
    df_processed = df.copy()
    # Apply custom normalization functions defined in the constants file
    if hasattr(constants, 'NORMALIZATIONS'):
        for col, norm_func in constants.NORMALIZATIONS.items():
            if col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').apply(norm_func)
    
    # Apply standard z-score normalization to grouped solar index features
    index_groups = {'AL_index': [c for c in input_columns if c.startswith('AL_index')], 'SYM_H': [c for c in input_columns if c.startswith('SYM_H')], 'f107_index': [c for c in input_columns if c.startswith('f107_index')]}
    for group, cols in index_groups.items():
        if group in means and group in stds and stds.get(group) != 0: df_processed[cols] = (df_processed[cols] - means[group]) / stds[group]
    
    # Return the fully preprocessed dataframe
    return df_processed

def get_predictions_and_confidence(model: torch.nn.Module, df_processed: pd.DataFrame, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Uses the trained model to make predictions on the processed data."""
    # Convert dataframe to a PyTorch tensor and create a DataLoader
    input_tensor = torch.tensor(df_processed[input_columns].values.astype(np.float32)); dataset = TensorDataset(input_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_PREDICT, shuffle=False)
    all_predicted_temps, all_confidences = [], []
    
    # Process the data in batches without calculating gradients
    with torch.no_grad():
        for batch_tensor, in data_loader:
            # Get model output (logits) and determine the predicted class
            logits = model(batch_tensor.to(device)); predicted_classes = torch.argmax(logits, dim=1)
            
            # Convert predicted class index back to a temperature value
            predicted_temps_k = (predicted_classes.cpu().numpy() * TEMP_BIN_WIDTH_K + TEMP_BIN_CENTER_OFFSET_K)
            all_predicted_temps.extend(predicted_temps_k)
            
            # Calculate prediction confidence from softmax probabilities
            softmax_probs = torch.softmax(logits, dim=1); confidence_batch, _ = torch.max(softmax_probs, dim=1)
            all_confidences.extend((confidence_batch * 100).cpu().numpy())
            
    # Return arrays of all predicted temperatures and confidences
    return np.array(all_predicted_temps), np.array(all_confidences)

def process_single_timestamp_batch(timestamps: pd.DatetimeIndex, model: torch.nn.Module, device: torch.device, all_solar_indices: Tuple, altitude_range: np.ndarray, magnetic_coords: Dict, means: Dict, stds: Dict) -> pd.DataFrame:
    """Processes a single batch of timestamps to generate a full set of predictions."""
    # Unpack the solar index dataframes
    omni_df, f107_df, kp_df = all_solar_indices
    
    # Create the base dataframe and merge in solar features for the batch
    solar_features_batch = create_temporal_solar_features(timestamps, omni_df, f107_df, kp_df)
    batch_df = pd.DataFrame({'DateTimeFormatted': solar_features_batch.index.repeat(len(altitude_range)), 'Altitude': np.tile(altitude_range, len(solar_features_batch))})
    batch_df = pd.merge(batch_df, solar_features_batch, left_on='DateTimeFormatted', right_index=True, how='left')

    # Add fixed magnetic coordinates
    batch_df['GMLT'], batch_df['ILAT'] = magnetic_coords['GMLT'], ILAT_DEG
    
    # Calculate and add altitude-dependent coordinates
    sc_coords_df = calculate_spacecraft_coords(batch_df, magnetic_coords['L_SHELL'], magnetic_coords['GMLT'])
    batch_df = pd.concat([batch_df, sc_coords_df], axis=1)

    # Calculate footprint coordinates for unique times and merge back
    unique_timestamps = batch_df['DateTimeFormatted'].unique()
    footprint_df = calculate_footprint_coords(pd.DatetimeIndex(unique_timestamps), magnetic_coords['L_SHELL'], magnetic_coords['GMLT'])
    batch_df = pd.merge(batch_df, footprint_df, on='DateTimeFormatted', how='left')
    
    # Check for missing data on the complete dataframe
    batch_df = handle_missing_data(batch_df, input_columns, MISSING_DATA_THRESHOLD_PERCENT)
    if batch_df.empty:
        print("Warning: Batch is empty after removing rows with missing data. Skipping.", flush=True)
        return pd.DataFrame()

    # Proceed with preprocessing and prediction
    df_processed = preprocess_data(batch_df, means, stds)
    predicted_temps, confidences = get_predictions_and_confidence(model, df_processed, device)
    batch_df[PRED_TEMP_COLUMN], batch_df[CONFIDENCE_COLUMN] = predicted_temps, confidences
    
    # Return the final dataframe with predictions
    return batch_df

def plot_combined_visualization(df: pd.DataFrame, output_filename: str):
    """Creates and saves a multi-panel plot of predictions and solar indices."""
    print("--- Preparing data for visualization ---", flush=True)
    # Pivot data for heatmap plotting and select solar features for line plots
    available_features = [col for col in SOLAR_FEATURE_COLUMNS if col in df.columns]
    temp_heatmap = df.pivot(index='Altitude', columns='DateTimeFormatted', values=PRED_TEMP_COLUMN)
    conf_heatmap = df.pivot(index='Altitude', columns='DateTimeFormatted', values=CONFIDENCE_COLUMN)
    line_plot_data = df[['DateTimeFormatted'] + available_features].drop_duplicates().set_index('DateTimeFormatted')
    
    print("--- Generating combined visualization plot ---", flush=True)
    # Set up subplots with shared x-axis and appropriate height ratios
    num_subplots = 2 + len(available_features); height_ratios = [4, 4] + [1] * len(available_features)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(20, 6 + 2 * num_subplots), sharex=True, gridspec_kw={'height_ratios': height_ratios})
    if num_subplots == 1: axes = [axes]
    fig.suptitle(f'Model Predictions and Solar Wind Indices (L-shell = {L_shell_val}, GMLT = {GMLT_val})', fontsize=20)
    
    # Plot predicted temperature as a heatmap
    ax_temp, ax_conf = axes[0], axes[1]
    im_temp = ax_temp.pcolormesh(temp_heatmap.columns, temp_heatmap.index, temp_heatmap.values, shading='auto', cmap='turbo', vmin=0)
    ax_temp.set_ylabel('Altitude [km]', fontsize=14); ax_temp.set_title('Predicted Electron Temperature', fontsize=16, pad=10)
    
    # Plot model confidence as a heatmap
    im_conf = ax_conf.pcolormesh(conf_heatmap.columns, conf_heatmap.index, conf_heatmap.values, shading='auto', cmap='magma', vmin=0, vmax=CONFIDENCE_VMAX_PERCENT)
    ax_conf.set_ylabel('Altitude [km]', fontsize=14); ax_conf.set_title('Peak Model Prediction Confidence', fontsize=16, pad=10)
    
    # Plot solar index features as line plots
    plot_params = {'Kp_index': {'label': 'Kp Index', 'color': 'red'}, 'SYM_H_0': {'label': 'SYM-H (nT)', 'color': 'blue'}, 'f107_index_0': {'label': 'F10.7 Index', 'color': 'green'}, 'AL_index_0': {'label': 'AL Index (nT)', 'color': 'purple'}}
    for i, feature in enumerate(available_features):
        ax_line = axes[2 + i]; params = plot_params.get(feature, {'label': feature, 'color': 'black'})
        data = line_plot_data[feature] / KP_INDEX_SCALE_FACTOR if feature == 'Kp_index' else line_plot_data[feature]
        ax_line.plot(line_plot_data.index, data, color=params['color'], linewidth=1.5)
        ax_line.set_ylabel(params['label'], fontsize=12); ax_line.grid(True, linestyle='--', alpha=0.6); ax_line.set_xlim(line_plot_data.index.min(), line_plot_data.index.max())
    
    # Configure the shared x-axis (date formatting)
    ax_bottom = axes[-1]
    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1)); ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_bottom.set_xlabel(f'Date in {temp_heatmap.columns.min().year}', fontsize=14); plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right"); fig.align_ylabels(axes)
    
    # Add colorbars for the heatmaps
    fig.subplots_adjust(right=0.92, top=0.94)
    pos_temp = ax_temp.get_position(); cax_temp = fig.add_axes([pos_temp.x1 + 0.01, pos_temp.y0, 0.015, pos_temp.height])
    fig.colorbar(im_temp, cax=cax_temp).set_label('Temperature [K]', fontsize=12)
    pos_conf = ax_conf.get_position(); cax_conf = fig.add_axes([pos_conf.x1 + 0.01, pos_conf.y0, 0.015, pos_conf.height])
    cbar_conf = fig.colorbar(im_conf, cax=cax_conf); cbar_conf.set_label('Confidence (%)', fontsize=12)
    
    # Customize confidence colorbar ticks to show a '+' for the max value
    ticks = cbar_conf.get_ticks(); tick_labels = [f'{int(t)}' for t in ticks]
    if tick_labels: tick_labels[-1] = f'{tick_labels[-1]}+'
    cbar_conf.set_ticks(ticks); cbar_conf.set_ticklabels(tick_labels)
    
    # Save the finalized plot to a file
    print(f"Saving combined plot to: {output_filename}", flush=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight'); plt.close(fig)

# ==============================================================================
# --- 3. MAIN EXECUTION LOGIC ---
# ==============================================================================
def main():
    """Main function to orchestrate the data loading, prediction, and visualization pipeline."""
    print("--- Starting Combined Data Generation and Visualization ---", flush=True)
    print(f"Using L-shell = {L_shell_val}, ILAT = ~{ILAT_DEG:.2f} degrees.", flush=True)
    
    # Set up the computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Check for required model and stats files
    check_file_exists(CHECKPOINT_PATH, "Model checkpoint"); check_file_exists(STATS_PATH, "Normalization stats file")
    
    # Load all necessary solar and geomagnetic index data
    omni_df, f107_df, kp_df = load_solar_indices(OMNI_AL_SYMH_PATH, F107_FILE_PATH, KP_FILE_PATH)
    
    # Load the pre-trained neural network model and set to evaluation mode
    model = model_definition.FeedForwardNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device); model.eval()
    
    # Load the normalization statistics
    means, stds = load_normalization_stats(STATS_PATH)
    
    # Create the time and altitude grids for data generation
    time_range = pd.date_range(start=TIME_START, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')
    altitude_range = np.arange(ALTITUDE_START_KM, ALTITUDE_END_KM + 1, ALTITUDE_INCREMENT_KM, dtype=np.float32)
    
    # Split the time range into manageable batches to control memory usage
    timestamp_batches = [time_range[i:i + TIMESTAMPS_PER_BATCH] for i in range(0, len(time_range), TIMESTAMPS_PER_BATCH)]
    all_solar_indices = (omni_df, f107_df, kp_df)
    
    print(f"\n--- Generating data for {len(timestamp_batches)} batches ---", flush=True)
    all_dfs = []
    # Process each batch of timestamps to generate predictions
    for batch_timestamps in tqdm(timestamp_batches, desc="Generating Data"):
        if batch_timestamps.empty: continue
        batch_df = process_single_timestamp_batch(
            timestamps=batch_timestamps, model=model, device=device,
            all_solar_indices=all_solar_indices, altitude_range=altitude_range,
            magnetic_coords=FIXED_MAGNETIC_COORDS, means=means, stds=stds
        )
        # Append the processed batch dataframe to a list
        if not batch_df.empty: all_dfs.append(batch_df)
        
    # Exit if no data could be generated, likely due to missing solar index data
    if not all_dfs: sys.exit("\nError: No data was generated. Check for missing solar index data for the specified time range.")
    
    # Concatenate all batch dataframes into a single dataframe for plotting
    print("\n--- Data generation complete. Concatenating and visualizing... ---", flush=True)
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Final DataFrame created with shape: {final_df.shape}", flush=True)
    
    # Generate and save the final combined visualization plot
    plot_combined_visualization(final_df, COMBINED_PLOT_FILENAME)
    print(f"\n--- Pipeline complete. Visualization saved to '{COMBINED_PLOT_FILENAME}' ---", flush=True)

if __name__ == "__main__":
    # Execute the main function when the script is run
    main()