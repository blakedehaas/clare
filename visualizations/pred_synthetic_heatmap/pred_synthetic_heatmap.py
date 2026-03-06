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

# --- SPACEPY IMPORT ---
try:
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
except ImportError:
    print("Error: The 'spacepy' library is required.", flush=True)
    sys.exit(1)

# --- MODEL IMPORTS & PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
sys.path.append(PROJECT_ROOT)
from config import INPUT_COLUMNS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NORMALIZATIONS, INDEX_GROUPS
from inference import load_normalization_stats
from models.feed_forward import FeedForwardNetwork

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
L_shell_val = 3.0
FIXED_GCLAT = 43.0
FIXED_GCLON = 289.0
ILAT_DEG = np.rad2deg(np.arccos(1.0 / np.sqrt(L_shell_val)))

# --- Physics & Model Constants ---
RE_KM = 6371.0
TEMP_BIN_WIDTH_K = 100
TEMP_BIN_CENTER_OFFSET_K = 50

# --- Visualization Config ---
COMBINED_PLOT_FILENAME = f"L{L_shell_val}_MillstoneHill_synthetic_visualization.png"
PRED_TEMP_COLUMN = 'Te1_pred'
CONFIDENCE_COLUMN = 'confidence'
SOLAR_FEATURE_COLUMNS = ['Kp_index', 'SYM_H_0', 'f107_index_0', 'AL_index_0']
KP_INDEX_SCALE_FACTOR = 10.0
CONFIDENCE_VMAX_PERCENT = 50.0

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def check_file_exists(filepath, description):
    if not os.path.exists(filepath):
        print(f"Error: {description} not found at '{filepath}'", flush=True)
        sys.exit(1)

def load_solar_indices(omni_al_symh_path, f107_file_path, kp_file_path):
    print("--- Loading and Processing Solar Index Data ---", flush=True)
    check_file_exists(kp_file_path, "Kp index file")

    al_symh_files = glob.glob(omni_al_symh_path)
    if not al_symh_files: sys.exit(f"Error: No AL/SYM-H files found")
    df_list = []
    for f in tqdm(al_symh_files, desc="Reading AL/SYM-H files"):
        df_chunk = pd.read_csv(f, sep=r'\s+', names=['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H'],
                               usecols=[0,1,2,3,4,5], na_values=[9999, 99999, 999.9, 99999.99])
        df_chunk.dropna(subset=['Year', 'Day', 'Hour', 'Minute'], inplace=True)
        df_list.append(df_chunk)

    omni_df = pd.concat(df_list, ignore_index=True)
    omni_df['DateTime'] = pd.to_datetime(omni_df['Year'].astype(int).astype(str) +
                                         omni_df['Day'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                          pd.to_timedelta(omni_df['Hour'], unit='h') + \
                          pd.to_timedelta(omni_df['Minute'], unit='m')
    omni_df = omni_df[['DateTime', 'AL_index', 'SYM_H']].drop_duplicates('DateTime').set_index('DateTime').sort_index()

    f107_files = glob.glob(f107_file_path)
    f107_list = []
    for f in f107_files:
        df_chunk = pd.read_csv(f, sep=r'\s+', names=['Year', 'Day', 'Hour', 'f107_index'], na_values=[999.9])
        df_chunk.dropna(subset=['Year', 'Day', 'Hour'], inplace=True)
        f107_list.append(df_chunk)

    f107_df = pd.concat(f107_list, ignore_index=True)
    f107_df['DateTime'] = pd.to_datetime(f107_df['Year'].astype(int).astype(str) +
                                         f107_df['Day'].astype(int).astype(str).str.zfill(3), format='%Y%j') + \
                          pd.to_timedelta(f107_df['Hour'], unit='h')
    f107_df = f107_df[['DateTime', 'f107_index']].drop_duplicates('DateTime').set_index('DateTime').sort_index()

    kp_df = pd.read_csv(kp_file_path, sep=r'\s+', names=['Year', 'DOY', 'Hour', 'Kp_index'], na_values=[99.9])
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

def calculate_millstone_anchored_coords(df, L_shell):
    required_cols = {'DateTimeFormatted', 'Altitude'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required column(s): {sorted(missing)}")
    if not np.issubdtype(df['DateTimeFormatted'].dtype, np.datetime64):
        raise TypeError("df['DateTimeFormatted'] must be pandas datetime64[ns] dtype.")

    FOOTPRINT_ALT_KM = 1000.0

    t_unique = df['DateTimeFormatted'].drop_duplicates()
    iso_unique = t_unique.dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    ticks_unique = Ticktock(iso_unique, 'ISO')

    foot_gdz_unique = coord.Coords(
        [[FOOTPRINT_ALT_KM, FIXED_GCLAT, FIXED_GCLON]] * len(t_unique),
        'GDZ', 'sph', ticks=ticks_unique
    )

    foot_sm_unique = foot_gdz_unique.convert('SM', 'sph')
    mlt_unique = (foot_sm_unique.long / 15.0 + 12.0) % 24.0

    mlt_map = pd.Series(mlt_unique, index=t_unique.values)
    lon_sm_map = pd.Series(foot_sm_unique.long, index=t_unique.values)

    mlt_all = df['DateTimeFormatted'].map(mlt_map).to_numpy()
    lon_sm_deg_all = df['DateTimeFormatted'].map(lon_sm_map).to_numpy()
    lon_sm_rad_all = np.deg2rad(lon_sm_deg_all)

    r_re = (df['Altitude'].to_numpy() + RE_KM) / RE_KM
    arg = np.clip(r_re / L_shell, 0.0, 1.0)
    lam_rad = np.arccos(np.sqrt(arg))

    x_sm = r_re * np.cos(lam_rad) * np.cos(lon_sm_rad_all)
    y_sm = r_re * np.cos(lam_rad) * np.sin(lon_sm_rad_all)
    z_sm = r_re * np.sin(lam_rad)

    iso_all = df['DateTimeFormatted'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    ticks_all = Ticktock(iso_all, 'ISO')

    sc_sm = coord.Coords(np.column_stack([x_sm, y_sm, z_sm]), 'SM', 'car', ticks=ticks_all)
    sc_gdz = sc_sm.convert('GDZ', 'sph')
    sc_mag = sc_sm.convert('MAG', 'sph')

    output = pd.DataFrame({
        'GMLT': mlt_all.astype(float),
        'XXLAT': sc_gdz.lati.astype(float),
        'XXLON': sc_gdz.long.astype(float),
        'GLAT': sc_mag.lati.astype(float),
        'GCLAT': np.full(len(df), FIXED_GCLAT, dtype=float),
        'GCLON': np.full(len(df), FIXED_GCLON, dtype=float),
    }, index=df.index)

    return output

def handle_missing_data(df, required_cols, threshold):
    nan_counts = df[required_cols].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if cols_with_nan.empty: return df
    for col, count in cols_with_nan.items():
        if (count / len(df)) * 100 > threshold: sys.exit(f"Error: Too much missing data in {col}")
    return df.dropna(subset=cols_with_nan.index).reset_index(drop=True)

def preprocess_data(df, means, stds):
    df_p = df.copy()
    for col, norm_func in NORMALIZATIONS.items():
        if col in df_p.columns: df_p[col] = pd.to_numeric(df_p[col], errors='coerce').apply(norm_func)
    for g, cols in INDEX_GROUPS.items():
        if g in means and g in stds: df_p[cols] = (df_p[cols] - means[g]) / stds[g]
    return df_p

def get_predictions_and_confidence(model, df_p, device):
    input_tensor = torch.tensor(df_p[INPUT_COLUMNS].values.astype(np.float32))
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

    coords_df = calculate_millstone_anchored_coords(batch_df, L_shell_val)
    batch_df = pd.concat([batch_df, coords_df], axis=1)
    batch_df['ILAT'] = ILAT_DEG

    batch_df = handle_missing_data(batch_df, INPUT_COLUMNS, 5.0)
    if batch_df.empty: return pd.DataFrame()

    df_p = preprocess_data(batch_df, means, stds)
    p_temp, p_conf = get_predictions_and_confidence(model, df_p, device)
    batch_df[PRED_TEMP_COLUMN], batch_df[CONFIDENCE_COLUMN] = p_temp, p_conf
    return batch_df

def plot_combined_visualization(df, output_filename):
    """Creates and saves a multi-panel plot including colorbars for the heatmaps."""
    print("--- Preparing data and generating visualization ---", flush=True)

    available_features = [col for col in SOLAR_FEATURE_COLUMNS if col in df.columns]
    temp_heatmap = df.pivot(index='Altitude', columns='DateTimeFormatted', values=PRED_TEMP_COLUMN)
    conf_heatmap = df.pivot(index='Altitude', columns='DateTimeFormatted', values=CONFIDENCE_COLUMN)
    line_plot_data = df[['DateTimeFormatted'] + available_features].drop_duplicates().set_index('DateTimeFormatted')

    num_subplots = 2 + len(available_features)
    height_ratios = [4, 4] + [1] * len(available_features)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(20, 6 + 2 * num_subplots),
                             sharex=True, gridspec_kw={'height_ratios': height_ratios})

    fig.suptitle(f'Model Predictions: Millstone Hill Anchor (L={L_shell_val})', fontsize=20)

    ax_temp = axes[0]
    im_temp = ax_temp.pcolormesh(temp_heatmap.columns, temp_heatmap.index, temp_heatmap.values,
                                 shading='auto', cmap='turbo', vmin=0)
    ax_temp.set_ylabel('Altitude [km]', fontsize=14)
    ax_temp.set_title('Predicted Electron Temperature', fontsize=16, pad=10)

    ax_conf = axes[1]
    im_conf = ax_conf.pcolormesh(conf_heatmap.columns, conf_heatmap.index, conf_heatmap.values,
                                 shading='auto', cmap='magma', vmin=0, vmax=CONFIDENCE_VMAX_PERCENT)
    ax_conf.set_ylabel('Altitude [km]', fontsize=14)
    ax_conf.set_title('Peak Model Prediction Confidence', fontsize=16, pad=10)

    plot_params = {
        'Kp_index': {'label': 'Kp Index', 'color': 'red'},
        'SYM_H_0': {'label': 'SYM-H (nT)', 'color': 'blue'},
        'f107_index_0': {'label': 'F10.7 Index', 'color': 'green'},
        'AL_index_0': {'label': 'AL Index (nT)', 'color': 'purple'}
    }
    for i, feature in enumerate(available_features):
        ax_line = axes[2 + i]
        params = plot_params.get(feature, {'label': feature, 'color': 'black'})
        data = line_plot_data[feature] / KP_INDEX_SCALE_FACTOR if feature == 'Kp_index' else line_plot_data[feature]
        ax_line.plot(line_plot_data.index, data, color=params['color'], linewidth=1.5)
        ax_line.set_ylabel(params['label'], fontsize=12)
        ax_line.grid(True, linestyle='--', alpha=0.6)
        ax_line.set_xlim(line_plot_data.index.min(), line_plot_data.index.max())

    ax_bottom = axes[-1]
    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_bottom.set_xlabel(f'Date in {temp_heatmap.columns.min().year}', fontsize=14)
    plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right")

    fig.subplots_adjust(right=0.90, top=0.94)

    pos_temp = ax_temp.get_position()
    cax_temp = fig.add_axes([pos_temp.x1 + 0.01, pos_temp.y0, 0.015, pos_temp.height])
    fig.colorbar(im_temp, cax=cax_temp).set_label('Temperature [K]', fontsize=12)

    pos_conf = ax_conf.get_position()
    cax_conf = fig.add_axes([pos_conf.x1 + 0.01, pos_conf.y0, 0.015, pos_conf.height])
    cbar_conf = fig.colorbar(im_conf, cax=cax_conf)
    cbar_conf.set_label('Confidence (%)', fontsize=12)

    ticks = cbar_conf.get_ticks()
    tick_labels = [f'{int(t)}' for t in ticks]
    if tick_labels:
        tick_labels[-1] = f'{tick_labels[-1]}+'
    cbar_conf.set_ticks(ticks)
    cbar_conf.set_ticklabels(tick_labels)

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

    model = FeedForwardNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
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
