# --- GENERAL IMPORTS ---
import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- VISUALIZATION IMPORTS ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- APEXPY IMPORT ---
try:
    from apexpy import Apex
except ImportError:
    print("Error: The 'apexpy' library is required. Install via 'pip install apexpy'", flush=True)
    sys.exit(1)

# --- MODEL IMPORTS & PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
sys.path.append(PROJECT_ROOT)
import models.feed_forward as model_definition
import constants

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Path Configuration ---
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'input_dataset')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
KP_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_kp_index.lst')
OMNI_AL_SYMH_PATH = os.path.join(BASE_DATA_DIR, 'omni_al_index_symh', '*.lst')
F107_FILE_PATH = os.path.join(BASE_DATA_DIR, 'omni_f107', '*.lst')
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "checkpoint.pth")
STATS_PATH = os.path.join(CHECKPOINTS_DIR, "norm_stats.json")

# --- Data Generation Grid Config ---
TIME_START = "1991-01-28 00:00:00"
TIME_END = "1991-02-10 00:00:00"
TIME_INCREMENT = "10min"
ALTITUDE_START_KM = 1000
ALTITUDE_END_KM = 8000
ALTITUDE_INCREMENT_KM = 10

TIMESTAMPS_PER_BATCH = 500
BATCH_SIZE_PREDICT = 8192

# --- FIXED GEOGRAPHIC ANCHOR (Millstone Hill) ---
FIXED_GCLAT = 43.0  # Millstone Hill Latitude
FIXED_GCLON = 289.0 # Millstone Hill Longitude (East)
FOOTPRINT_ALT_KM = 300.0 # Altitude of the magnetic footprint matching Akebono orbits

# --- Physics & Model Constants ---
RE_KM = 6371.0
TEMP_BIN_WIDTH_K = 100
TEMP_BIN_CENTER_OFFSET_K = 50

# --- Visualization Config ---
COMBINED_PLOT_FILENAME = "MillstoneHill_synthetic_visualization.png"
PRED_TEMP_COLUMN = 'Te1_pred'
CONFIDENCE_COLUMN = 'confidence'
SOLAR_FEATURE_COLUMNS = ['Kp_index', 'SYM_H_0', 'f107_index_0', 'AL_index_0']
KP_INDEX_SCALE_FACTOR = 10.0
CONFIDENCE_VMAX_PERCENT = 50.0

# --- Model Input Config ---
input_columns = [
    'Altitude', 'GCLAT', 'GCLON', 'ILAT', 'GLAT', 'GMLT', 'XXLAT', 'XXLON', 
    'AL_index_0', 'AL_index_1', 'AL_index_2', 'AL_index_3', 'AL_index_4', 'AL_index_5', 'AL_index_6', 'AL_index_7', 'AL_index_8', 'AL_index_9', 'AL_index_10', 'AL_index_11', 'AL_index_12', 'AL_index_13', 'AL_index_14', 'AL_index_15', 'AL_index_16', 'AL_index_17', 'AL_index_18', 'AL_index_19', 'AL_index_20', 'AL_index_21', 'AL_index_22', 'AL_index_23', 'AL_index_24', 'AL_index_25', 'AL_index_26', 'AL_index_27', 'AL_index_28', 'AL_index_29', 'AL_index_30', 
    'SYM_H_0', 'SYM_H_1', 'SYM_H_2', 'SYM_H_3', 'SYM_H_4', 'SYM_H_5', 'SYM_H_6', 'SYM_H_7', 'SYM_H_8', 'SYM_H_9', 'SYM_H_10', 'SYM_H_11', 'SYM_H_12', 'SYM_H_13', 'SYM_H_14', 'SYM_H_15', 'SYM_H_16', 'SYM_H_17', 'SYM_H_18', 'SYM_H_19', 'SYM_H_20', 'SYM_H_21', 'SYM_H_22', 'SYM_H_23', 'SYM_H_24', 'SYM_H_25', 'SYM_H_26', 'SYM_H_27', 'SYM_H_28', 'SYM_H_29', 'SYM_H_30', 'SYM_H_31', 'SYM_H_32', 'SYM_H_33', 'SYM_H_34', 'SYM_H_35', 'SYM_H_36', 'SYM_H_37', 'SYM_H_38', 'SYM_H_39', 'SYM_H_40', 'SYM_H_41', 'SYM_H_42', 'SYM_H_43', 'SYM_H_44', 'SYM_H_45', 'SYM_H_46', 'SYM_H_47', 'SYM_H_48', 'SYM_H_49', 'SYM_H_50', 'SYM_H_51', 'SYM_H_52', 'SYM_H_53', 'SYM_H_54', 'SYM_H_55', 'SYM_H_56', 'SYM_H_57', 'SYM_H_58', 'SYM_H_59', 'SYM_H_60', 'SYM_H_61', 'SYM_H_62', 'SYM_H_63', 'SYM_H_64', 'SYM_H_65', 'SYM_H_66', 'SYM_H_67', 'SYM_H_68', 'SYM_H_69', 'SYM_H_70', 'SYM_H_71', 'SYM_H_72', 'SYM_H_73', 'SYM_H_74', 'SYM_H_75', 'SYM_H_76', 'SYM_H_77', 'SYM_H_78', 'SYM_H_79', 'SYM_H_80', 'SYM_H_81', 'SYM_H_82', 'SYM_H_83', 'SYM_H_84', 'SYM_H_85', 'SYM_H_86', 'SYM_H_87', 'SYM_H_88', 'SYM_H_89', 'SYM_H_90', 'SYM_H_91', 'SYM_H_92', 'SYM_H_93', 'SYM_H_94', 'SYM_H_95', 'SYM_H_96', 'SYM_H_97', 'SYM_H_98', 'SYM_H_99', 'SYM_H_100', 'SYM_H_101', 'SYM_H_102', 'SYM_H_103', 'SYM_H_104', 'SYM_H_105', 'SYM_H_106', 'SYM_H_107', 'SYM_H_108', 'SYM_H_109', 'SYM_H_110', 'SYM_H_111', 'SYM_H_112', 'SYM_H_113', 'SYM_H_114', 'SYM_H_115', 'SYM_H_116', 'SYM_H_117', 'SYM_H_118', 'SYM_H_119', 'SYM_H_120', 'SYM_H_121', 'SYM_H_122', 'SYM_H_123', 'SYM_H_124', 'SYM_H_125', 'SYM_H_126', 'SYM_H_127', 'SYM_H_128', 'SYM_H_129', 'SYM_H_130', 'SYM_H_131', 'SYM_H_132', 'SYM_H_133', 'SYM_H_134', 'SYM_H_135', 'SYM_H_136', 'SYM_H_137', 'SYM_H_138', 'SYM_H_139', 'SYM_H_140', 'SYM_H_141', 'SYM_H_142', 'SYM_H_143', 'SYM_H_144', 
    'f107_index_0', 'f107_index_1', 'f107_index_2', 'f107_index_3', 'Kp_index'
]
input_size = len(input_columns)
hidden_size = 2048
output_size = 150

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def check_file_exists(filepath, description):
    if not os.path.exists(filepath):
        print(f"Error: {description} not found at '{filepath}'", flush=True)
        sys.exit(1)

def load_normalization_stats(stats_path):
    check_file_exists(stats_path, "Normalization stats file")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']

def load_solar_indices(omni_al_symh_path, f107_file_path, kp_file_path):
    print("--- Loading and Processing Solar Index Data ---", flush=True)
    check_file_exists(kp_file_path, "Kp index file")
    
    # --- AL and SYM-H ---
    al_symh_files = glob.glob(omni_al_symh_path)
    if not al_symh_files: sys.exit(f"Error: No AL/SYM-H files found")
    df_list = []
    for f in tqdm(al_symh_files, desc="Reading AL/SYM-H files"):
        df_chunk = pd.read_csv(f, sep=r'\s+', names=['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H'], 
                               usecols=[0,1,2,3,4,5], na_values=[9999, 99999, 999.9, 99999.99])
        df_chunk.dropna(subset=['Year', 'Day', 'Hour', 'Minute'], inplace=True) # Drop rows with missing time
        df_list.append(df_chunk)
    
    omni_df = pd.concat(df_list, ignore_index=True)
    omni_df['DateTime'] = pd.to_datetime(omni_df['Year'].astype(int).astype(str) + 
                                         omni_df['Day'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                          pd.to_timedelta(omni_df['Hour'], unit='h') + \
                          pd.to_timedelta(omni_df['Minute'], unit='m')
    omni_df = omni_df[['DateTime', 'AL_index', 'SYM_H']].drop_duplicates('DateTime').set_index('DateTime').sort_index()
    
    # --- F10.7 ---
    f107_files = glob.glob(f107_file_path)
    f107_list = []
    for f in f107_files:
        df_chunk = pd.read_csv(f, sep=r'\s+', names=['Year', 'Day', 'Hour', 'f107_index'], na_values=[99.9, 999.9])
        df_chunk.dropna(subset=['Year', 'Day', 'Hour'], inplace=True)
        f107_list.append(df_chunk)
    
    f107_df = pd.concat(f107_list, ignore_index=True)
    f107_df['DateTime'] = pd.to_datetime(f107_df['Year'].astype(int).astype(str) + 
                                         f107_df['Day'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                          pd.to_timedelta(f107_df['Hour'], unit='h')
    f107_df = f107_df[['DateTime', 'f107_index']].drop_duplicates('DateTime').set_index('DateTime').sort_index()
    
    # --- Kp Index ---
    kp_df = pd.read_csv(kp_file_path, sep=r'\s+', names=['Year', 'DOY', 'Hour', 'Kp_index'], na_values=[99.9])
    # Drop rows where time info is NaN before converting to int
    kp_df.dropna(subset=['Year', 'DOY', 'Hour'], inplace=True)
    
    kp_df['DateTime'] = pd.to_datetime(kp_df['Year'].astype(int).astype(str) + 
                                       kp_df['DOY'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                        pd.to_timedelta(kp_df['Hour'], unit='h')
    kp_df = kp_df[['DateTime', 'Kp_index']].drop_duplicates('DateTime').set_index('DateTime').sort_index()
    
    print("Solar index data loaded successfully.", flush=True)
    return omni_df, f107_df, kp_df

def create_temporal_solar_features(target_times, omni_df, f107_df, kp_df):
    all_features = {}
    al_time_lags = pd.timedelta_range(start='0m', end='5h', freq='10min'); past_al_ts = target_times.values[:, None] - al_time_lags.values
    flat_al = omni_df['AL_index'].reindex(past_al_ts.ravel()).values; lagged_al = flat_al.reshape(past_al_ts.shape)
    for i in range(lagged_al.shape[1]): all_features[f'AL_index_{i}'] = lagged_al[:, i]
    
    sym_h_time_lags = pd.timedelta_range(start='0m', end='3d', freq='30min'); past_sym_h_ts = target_times.values[:, None] - sym_h_time_lags.values
    flat_sym_h = omni_df['SYM_H'].reindex(past_sym_h_ts.ravel()).values; lagged_sym_h = flat_sym_h.reshape(past_sym_h_ts.shape)
    for i in range(lagged_sym_h.shape[1]): all_features[f'SYM_H_{i}'] = lagged_sym_h[:, i]
    
    f107_time_lags = pd.timedelta_range(start='0h', end='72h', freq='24h'); past_f107_ts = target_times.values[:, None] - f107_time_lags.values
    rounded_f107 = pd.DatetimeIndex(past_f107_ts.ravel()).round('h'); flat_f107 = f107_df['f107_index'].reindex(rounded_f107, method='nearest').values
    lagged_f107 = flat_f107.reshape(past_f107_ts.shape)
    for i in range(lagged_f107.shape[1]): all_features[f'f107_index_{i}'] = lagged_f107[:, i]
    
    all_features['Kp_index'] = kp_df['Kp_index'].reindex(target_times.round('h'), method='nearest').values
    return pd.DataFrame(all_features, index=target_times)

def compute_apex_footprint_products(
    df: pd.DataFrame,
    xlat_fixed: float,
    xlon_fixed: float,
    footprint_alt_km: float = FOOTPRINT_ALT_KM, 
    refh_km: float = 110.0,  # must be fixed
    lon_convention: str = "east_0_360",   # "east_0_360" or "pm180"
    include_qd_glat: bool = True,
    verify_mapping: bool = True,
    verify_tol_deg: float = 1e-3,
) -> pd.DataFrame:
    """
    Apex mapping for MI-coupling / auroral precipitation workflows.

    Fixed inputs:
      - xlat_fixed, xlon_fixed: geodetic spacecraft lat/lon (constant for all rows)
      - df['Altitude']: spacecraft altitude [km]
      - df['DateTimeFormatted']: timestamps (timezone-aware OK; interpreted as UTC)

    Outputs per row:
      - XXLAT, XXLON: fixed geodetic inputs
      - ILAT: invariant latitude = modified apex latitude (alat) at spacecraft point
      - ALON: modified apex longitude (alon) at spacecraft point (field-line label)
      - GMLT: magnetic local time [h] computed from ALON and UTC time (Apex definition)
      - GCLAT, GCLON: geodetic footprint at `footprint_alt_km` on the same apex field line
                      (computed by converting (ILAT, ALON) apex->geo at that altitude)
      - GLAT (optional): QD latitude at spacecraft point (diagnostic "geomagnetic lat")
      - VERIFY_OK (optional): whether apex->geo footprint maps back to same (ILAT, ALON)

    Notes:
      - This uses an APEX-consistent footprint (hold (alat, alon) constant), which is
        appropriate when field-line mapping fidelity is critical.
      - `refh_km` defines the coordinate system (typically 110 km). Do not confuse it
        with `footprint_alt_km`.
    """
    # --------------------------
    # Input validation
    # --------------------------
    required = {"DateTimeFormatted", "Altitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    if lon_convention not in ("east_0_360", "pm180"):
        raise ValueError("lon_convention must be 'east_0_360' or 'pm180'")

    # Work on a copy to avoid mutating caller's DataFrame
    df_calc = df.copy()

    # Ensure datetime column is datetime64 and timezone-naive UTC
    df_calc["DateTimeFormatted"] = pd.to_datetime(df_calc["DateTimeFormatted"], utc=True, errors="raise")
    # Remove tzinfo (ApexPy expects naive datetimes; still UTC by construction above)
    df_calc["DateTimeFormatted"] = df_calc["DateTimeFormatted"].dt.tz_convert(None)

    # Ensure Altitude is numeric
    alt = pd.to_numeric(df_calc["Altitude"], errors="raise").to_numpy(dtype=float)

    # Normalize fixed longitude to [0, 360) for internal consistency
    xlat_fixed = float(xlat_fixed)
    xlon_fixed = float(xlon_fixed)
    xlon_fixed_360 = xlon_fixed % 360.0

    def _wrap360(x):
        return np.mod(x, 360.0)

    def _wrap180(x):
        return (x + 180.0) % 360.0 - 180.0

    n = len(df_calc)
    times_py = df_calc["DateTimeFormatted"].dt.to_pydatetime()

    # Preallocate outputs
    ILAT = np.full(n, np.nan, dtype=float)
    ALON = np.full(n, np.nan, dtype=float)
    MLT  = np.full(n, np.nan, dtype=float)
    GCLAT = np.full(n, np.nan, dtype=float)
    GCLON = np.full(n, np.nan, dtype=float)
    GLAT = np.full(n, np.nan, dtype=float) if include_qd_glat else None
    VERIFY_OK = np.full(n, True, dtype=bool) if verify_mapping else None

    # --------------------------
    # Main loop: group by day
    # --------------------------
    day_groups = df_calc.groupby(df_calc["DateTimeFormatted"].dt.floor("D")).groups
    for day, idx in day_groups.items():
        A = Apex(pd.Timestamp(day).to_pydatetime(), refh=refh_km)

        ii = np.array(list(idx), dtype=int)
        k = len(ii)

        lat_i = np.full(k, xlat_fixed, dtype=float)
        lon_i = np.full(k, xlon_fixed_360, dtype=float)
        alt_i = alt[ii]
        t_i = [times_py[j] for j in ii]

        # 1) Spacecraft apex coords (field-line labels)
        alat_i, alon_i = A.geo2apex(lat_i, lon_i, alt_i)
        alat_i = np.asarray(alat_i, dtype=float)
        alon_i = _wrap360(np.asarray(alon_i, dtype=float))

        ILAT[ii] = alat_i
        ALON[ii] = alon_i

        # 2) MLT from apex longitude + time
        # ApexPy's mlon2mlt expects "magnetic longitude" + datetime
        mlt_i = np.array([A.mlon2mlt(a, tt) for a, tt in zip(alon_i, t_i)], dtype=float)
        MLT[ii] = np.mod(mlt_i, 24.0)

        # 3) Footprint at chosen altitude: (alat, alon) in apex -> GEO at height
        fp_lat, fp_lon = A.convert(alat_i, alon_i, "apex", "geo", height=footprint_alt_km)
        fp_lat = np.asarray(fp_lat, dtype=float)
        fp_lon = _wrap360(np.asarray(fp_lon, dtype=float))

        GCLAT[ii] = fp_lat
        GCLON[ii] = fp_lon

        # 4) Optional: diagnostic QD latitude at spacecraft
        if include_qd_glat:
            qdlat_i, _ = A.geo2qd(lat_i, lon_i, alt_i)
            GLAT[ii] = np.asarray(qdlat_i, dtype=float)

        # 5) Optional: verify mapping consistency (footprint maps back to same ILAT/ALON)
        if verify_mapping:
            alat_chk, alon_chk = A.geo2apex(fp_lat, fp_lon, np.full(k, footprint_alt_km, dtype=float))
            alat_chk = np.asarray(alat_chk, dtype=float)
            alon_chk = _wrap360(np.asarray(alon_chk, dtype=float))

            d_ilat = np.abs(alat_chk - alat_i)
            d_alon = np.abs(_wrap180(alon_chk - alon_i))

            VERIFY_OK[ii] = (d_ilat <= verify_tol_deg) & (d_alon <= verify_tol_deg)

    # --------------------------
    # Assemble output DataFrame
    # --------------------------
    out = pd.DataFrame(index=df.index)
    out["XXLAT"] = xlat_fixed
    out["XXLON"] = xlon_fixed if lon_convention == "pm180" else xlon_fixed_360

    out["ILAT"] = ILAT
    out["ALON"] = ALON
    
    # Note: MLT returned here is mapped to "GMLT" to align with neural network input features
    out["GMLT"] = MLT
    out["GCLAT"] = GCLAT
    out["GCLON"] = _wrap180(GCLON) if lon_convention == "pm180" else GCLON
    out["FOOTPRINT_ALT_KM"] = float(footprint_alt_km)

    if include_qd_glat:
        out["GLAT"] = GLAT
    if verify_mapping:
        out["VERIFY_OK"] = VERIFY_OK

    return out

def handle_missing_data(df, required_cols, threshold):
    nan_counts = df[required_cols].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if cols_with_nan.empty: return df
    for col, count in cols_with_nan.items():
        if (count / len(df)) * 100 > threshold: sys.exit(f"Error: Too much missing data in {col}")
    return df.dropna(subset=cols_with_nan.index).reset_index(drop=True)

def preprocess_data(df, means, stds):
    df_p = df.copy()
    if hasattr(constants, 'NORMALIZATIONS'):
        for col, norm_func in constants.NORMALIZATIONS.items():
            if col in df_p.columns: df_p[col] = pd.to_numeric(df_p[col], errors='coerce').apply(norm_func)
    groups = {'AL_index': [c for c in input_columns if 'AL_index' in c], 'SYM_H': [c for c in input_columns if 'SYM_H' in c], 'f107_index': [c for c in input_columns if 'f107_index' in c]}
    for g, cols in groups.items():
        if g in means and g in stds: df_p[cols] = (df_p[cols] - means[g]) / stds[g]
    return df_p

def get_predictions_and_confidence(model, df_p, device):
    input_tensor = torch.tensor(df_p[input_columns].values.astype(np.float32))
    loader = DataLoader(TensorDataset(input_tensor), batch_size=BATCH_SIZE_PREDICT, shuffle=False)
    temps, confs = [], []
    with torch.no_grad():
        for b_tensor, in loader:
            logits = model(b_tensor.to(device))
            p_class = torch.argmax(logits, dim=1)
            temps.extend((p_class.cpu().numpy() * TEMP_BIN_WIDTH_K + TEMP_BIN_CENTER_OFFSET_K))
            probs = torch.softmax(logits, dim=1)
            conf_val, _ = torch.max(probs, dim=1)
            confs.extend((conf_val * 100).cpu().numpy())
    return np.array(temps), np.array(confs)

def process_single_timestamp_batch(timestamps, model, device, all_solar_indices, altitude_range, means, stds):
    omni_df, f107_df, kp_df = all_solar_indices
    solar_features = create_temporal_solar_features(timestamps, omni_df, f107_df, kp_df)
    batch_df = pd.DataFrame({'DateTimeFormatted': solar_features.index.repeat(len(altitude_range)), 'Altitude': np.tile(altitude_range, len(solar_features))})
    batch_df = pd.merge(batch_df, solar_features, left_on='DateTimeFormatted', right_index=True, how='left')

    # Apply ApexPy mappings for Millstone Hill coordinates natively constructing ILAT, GLAT, GMLT, etc.
    coords_df = compute_apex_footprint_products(
        df=batch_df, 
        xlat_fixed=FIXED_GCLAT, 
        xlon_fixed=FIXED_GCLON, 
        footprint_alt_km=FOOTPRINT_ALT_KM
    )
    
    # Horizontally merge the coordinate outputs natively matched to the identical index order
    batch_df = pd.concat([batch_df, coords_df], axis=1)
    
    batch_df = handle_missing_data(batch_df, input_columns, 5.0)
    if batch_df.empty: return pd.DataFrame()

    df_p = preprocess_data(batch_df, means, stds)
    p_temp, p_conf = get_predictions_and_confidence(model, df_p, device)
    batch_df[PRED_TEMP_COLUMN], batch_df[CONFIDENCE_COLUMN] = p_temp, p_conf
    return batch_df

def plot_combined_visualization(df: pd.DataFrame, output_filename: str):
    """Creates and saves a multi-panel plot including colorbars for the heatmaps."""
    print("--- Preparing data and generating visualization ---", flush=True)
    
    # 1. Prepare Data
    available_features = [col for col in SOLAR_FEATURE_COLUMNS if col in df.columns]
    temp_heatmap = df.pivot(index='Altitude', columns='DateTimeFormatted', values=PRED_TEMP_COLUMN)
    conf_heatmap = df.pivot(index='Altitude', columns='DateTimeFormatted', values=CONFIDENCE_COLUMN)
    line_plot_data = df[['DateTimeFormatted'] + available_features].drop_duplicates().set_index('DateTimeFormatted')
    
    # 2. Setup Figure
    num_subplots = 2 + len(available_features)
    height_ratios = [4, 4] + [1] * len(available_features)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(20, 6 + 2 * num_subplots), 
                             sharex=True, gridspec_kw={'height_ratios': height_ratios})
    
    fig.suptitle('Model Predictions: Millstone Hill Anchor', fontsize=20)
    
    # 3. Plot Temperature Heatmap
    ax_temp = axes[0]
    im_temp = ax_temp.pcolormesh(temp_heatmap.columns, temp_heatmap.index, temp_heatmap.values, 
                                 shading='auto', cmap='turbo', vmin=0)
    ax_temp.set_ylabel('Altitude [km]', fontsize=14)
    ax_temp.set_title('Predicted Electron Temperature', fontsize=16, pad=10)
    
    # 4. Plot Confidence Heatmap
    ax_conf = axes[1]
    im_conf = ax_conf.pcolormesh(conf_heatmap.columns, conf_heatmap.index, conf_heatmap.values, 
                                 shading='auto', cmap='magma', vmin=0, vmax=CONFIDENCE_VMAX_PERCENT)
    ax_conf.set_ylabel('Altitude [km]', fontsize=14)
    ax_conf.set_title('Peak Model Prediction Confidence', fontsize=16, pad=10)
    
    # 5. Plot Solar Index Features
    plot_params = {
        'Kp_index': {'label': 'Kp Index', 'color': 'red'},
        'SYM_H_0': {'label': 'SYM-H (nT)', 'color': 'blue'},
        'f107_index_0': {'label': 'F10.7 Index', 'color': 'green'},
        'AL_index_0': {'label': 'AL Index (nT)', 'color': 'purple'}
    }
    for i, feature in enumerate(available_features):
        ax_line = axes[2 + i]
        params = plot_params.get(feature, {'label': feature, 'color': 'black'})
        # Scale Kp back to 0-9 range for display if it was scaled by 10
        data = line_plot_data[feature] / KP_INDEX_SCALE_FACTOR if feature == 'Kp_index' else line_plot_data[feature]
        ax_line.plot(line_plot_data.index, data, color=params['color'], linewidth=1.5)
        ax_line.set_ylabel(params['label'], fontsize=12)
        ax_line.grid(True, linestyle='--', alpha=0.6)
        ax_line.set_xlim(line_plot_data.index.min(), line_plot_data.index.max())
    
    # 6. Format X-Axis
    ax_bottom = axes[-1]
    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_bottom.set_xlabel(f'Date in {temp_heatmap.columns.min().year}', fontsize=14)
    plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right")
    
    # 7. Add Colorbars
    # Adjust main plot to make room on the right
    fig.subplots_adjust(right=0.90, top=0.94)
    
    # Temperature Colorbar
    pos_temp = ax_temp.get_position()
    cax_temp = fig.add_axes([pos_temp.x1 + 0.01, pos_temp.y0, 0.015, pos_temp.height])
    fig.colorbar(im_temp, cax=cax_temp).set_label('Temperature [K]', fontsize=12)
    
    # Confidence Colorbar
    pos_conf = ax_conf.get_position()
    cax_conf = fig.add_axes([pos_conf.x1 + 0.01, pos_conf.y0, 0.015, pos_conf.height])
    cbar_conf = fig.colorbar(im_conf, cax=cax_conf)
    cbar_conf.set_label('Confidence (%)', fontsize=12)
    
    # Customize confidence ticks to show the '+' on the max value (e.g., "50+")
    ticks = cbar_conf.get_ticks()
    tick_labels = [f'{int(t)}' for t in ticks]
    if tick_labels:
        tick_labels[-1] = f'{tick_labels[-1]}+'
    cbar_conf.set_ticks(ticks)
    cbar_conf.set_ticklabels(tick_labels)
    
    # Save the finalized plot
    print(f"Saving plot to: {output_filename}", flush=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# --- 3. MAIN EXECUTION ---
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_file_exists(CHECKPOINT_PATH, "Model checkpoint"); check_file_exists(STATS_PATH, "Stats")
    
    omni_df, f107_df, kp_df = load_solar_indices(OMNI_AL_SYMH_PATH, F107_FILE_PATH, KP_FILE_PATH)
    
    model = model_definition.FeedForwardNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    means, stds = load_normalization_stats(STATS_PATH)
    
    time_range = pd.date_range(start=TIME_START, end=TIME_END, freq=TIME_INCREMENT, inclusive='left')
    altitude_range = np.arange(ALTITUDE_START_KM, ALTITUDE_END_KM + 1, ALTITUDE_INCREMENT_KM, dtype=np.float32)
    timestamp_batches = [time_range[i:i + TIMESTAMPS_PER_BATCH] for i in range(0, len(time_range), TIMESTAMPS_PER_BATCH)]
    
    all_dfs = []
    for batch in tqdm(timestamp_batches, desc="Batches"):
        res = process_single_timestamp_batch(batch, model, device, (omni_df, f107_df, kp_df), altitude_range, means, stds)
        if not res.empty: all_dfs.append(res)
        
    if not all_dfs: sys.exit("No data generated.")
    final_df = pd.concat(all_dfs, ignore_index=True)
    plot_combined_visualization(final_df, COMBINED_PLOT_FILENAME)
    print(f"Done. Saved to {COMBINED_PLOT_FILENAME}")

if __name__ == "__main__":
    main()