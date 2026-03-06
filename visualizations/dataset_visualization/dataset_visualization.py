import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects

from datasets import load_from_disk, concatenate_datasets, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DEFAULT_DATASET_DIR

# --- Define constants ---
BASE_DATASET_DIR = DEFAULT_DATASET_DIR

TE_COLUMN = 'Te1'
TE_DISPLAY_NAME = 'Electron Temperature'
TE_YAXIS_LABEL = 'Te'

BINS = 100

ILAT_COLUMN_NAME = 'ILAT'
L_SHELL_COLUMN_NAME = 'L_SHELL'
GLAT_COLUMN_NAME = 'GLAT'
GMLT_COLUMN_NAME = 'GMLT'
ALTITUDE_COLUMN_NAME = 'Altitude'

MAX_ROWS_FOR_PANDAS_CONVERSION = None


def find_dataset_dirs(base_dir):
    """Finds all directories containing dataset_info.json recursively."""
    dataset_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'dataset_info.json' in files:
            if os.path.abspath(root) != os.path.abspath(base_dir):
                dataset_dirs.append(root)
    return dataset_dirs


def calculate_l_shell(ilat_deg):
    """Calculates L-shell from Invariant Latitude (ILAT)."""
    if not isinstance(ilat_deg, pd.Series):
        ilat_deg = pd.Series(ilat_deg, dtype=float)

    ilat_deg_float = ilat_deg.astype(float)
    ilat_rad = np.deg2rad(ilat_deg_float)
    cos_ilat_rad = np.cos(ilat_rad)
    cos_ilat_rad_sq = cos_ilat_rad**2

    with np.errstate(divide='ignore', invalid='ignore'):
        l_shell = 1.0 / cos_ilat_rad_sq

    l_shell[np.isinf(l_shell) | (cos_ilat_rad_sq < 1e-9) | ilat_deg_float.abs().ge(90)] = np.nan
    return l_shell


def load_and_combine_datasets(base_dir, columns_to_keep=None, max_rows_to_convert=None):
    """Loads dataset(s) from the specified base directory."""
    all_datasets = []
    dataset_info_path = os.path.join(base_dir, 'dataset_info.json')

    if os.path.exists(dataset_info_path):
        print(f"Found dataset info file directly in: {base_dir}")
        ds = load_from_disk(base_dir)
        if isinstance(ds, dict):
            if 'train' in ds: ds = ds['train']
            else: ds = ds[next(iter(ds.keys()))]
        all_datasets.append(ds)
        print(f"Successfully loaded dataset with {len(ds)} entries.")
    else:
        print(f"Searching for datasets recursively under {base_dir}...")
        dataset_dirs = find_dataset_dirs(base_dir)
        if not dataset_dirs:
            raise FileNotFoundError(f"No datasets found recursively under {base_dir}.")

        print(f"Found {len(dataset_dirs)} dataset directories to load.")
        for dir_path in tqdm(dataset_dirs, desc="Loading datasets"):
            try:
                ds = load_from_disk(dir_path)
                if isinstance(ds, dict):
                    if 'train' in ds: ds = ds['train']
                    else: ds = ds[next(iter(ds.keys()))]
                all_datasets.append(ds)
            except Exception as e:
                print(f"Error loading dataset from {dir_path}: {e}. Skipping.")

    if not all_datasets:
        raise ValueError("No datasets could be successfully loaded.")

    if len(all_datasets) == 1:
        combined_dataset = all_datasets[0]
    else:
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"Combined dataset contains {len(combined_dataset)} entries.")

    dataset_for_pandas = combined_dataset

    if columns_to_keep:
        actual_cols = [col for col in columns_to_keep if col in dataset_for_pandas.column_names]
        if actual_cols and set(actual_cols) != set(dataset_for_pandas.column_names):
            dataset_for_pandas = dataset_for_pandas.select_columns(actual_cols)

    if max_rows_to_convert is not None and len(dataset_for_pandas) > max_rows_to_convert:
        dataset_for_pandas = dataset_for_pandas.shuffle(seed=42).select(range(max_rows_to_convert))

    print(f"Converting {len(dataset_for_pandas)} rows to Pandas DataFrame...")
    df = dataset_for_pandas.to_pandas()
    return df


def plot_observation_heatmaps(df, te_col, param_cols, param_labels, param_ranges, bins=50):
    df_filtered_te = df[df[te_col].notna() & (df[te_col] > 0)].copy()
    if len(df_filtered_te) == 0:
        print(f"Warning: No data after filtering for positive {te_col}.")
        return

    min_te_percentile = np.percentile(df_filtered_te[te_col], 1)
    max_te_percentile = np.percentile(df_filtered_te[te_col], 99.9)
    min_te = max(0, min_te_percentile)
    max_te = max_te_percentile
    if max_te <= min_te:
        max_te = min_te * 10 if min_te > 0 else 1000

    te_bins = np.linspace(min_te, max_te, bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    plot_panel_labels = ['(a)', '(b)', '(c)']
    text_outline_effect = [path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()]
    im = None

    max_counts_overall = 0
    hist_plot_data = []

    font_size_label = 28
    font_size_tick = 18
    font_size_panel_label = 28
    font_size_suptitle = 28

    for i, (param_col, param_range) in enumerate(zip(param_cols, param_ranges)):
        current_data = {'param_col': param_col, 'param_range': param_range, 'x_data': None, 'y_data': None, 'valid': False}
        if param_col not in df_filtered_te.columns:
            hist_plot_data.append(current_data)
            continue

        current_param_min, current_param_max = param_range

        valid_param_data_mask = (
            df_filtered_te[param_col].notna() &
            np.isfinite(df_filtered_te[param_col]) &
            (df_filtered_te[param_col] >= current_param_min) &
            (df_filtered_te[param_col] <= current_param_max)
        )
        x_data_temp = df_filtered_te.loc[valid_param_data_mask, param_col]
        y_data_temp = df_filtered_te.loc[valid_param_data_mask, te_col]

        if not x_data_temp.empty and not y_data_temp.empty:
            param_bins_temp = np.linspace(current_param_min, current_param_max, bins + 1)
            counts_temp, _, _ = np.histogram2d(x_data_temp, y_data_temp, bins=[param_bins_temp, te_bins])
            max_counts_overall = max(max_counts_overall, counts_temp.max() if counts_temp.size > 0 else 0)
            current_data['x_data'] = x_data_temp
            current_data['y_data'] = y_data_temp
            current_data['valid'] = True
        hist_plot_data.append(current_data)

    if max_counts_overall == 0:
        max_counts_overall = 1

    for i, ax in enumerate(axes):
        param_label = param_labels[i]
        plot_data = hist_plot_data[i]

        ax.set_ylim(min_te, max_te)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        if i == 0:
            ax.set_ylabel(f'{TE_YAXIS_LABEL} [K]', fontsize=font_size_label)

        ax.text(0.05, 0.95, plot_panel_labels[i], transform=ax.transAxes,
                fontsize=font_size_panel_label, fontweight='bold', va='top', path_effects=text_outline_effect)

        current_plot_param_min, current_plot_param_max = plot_data['param_range']

        if not plot_data['valid']:
            ax.set_xlabel(param_label, fontsize=font_size_label)
            ax.set_xlim(current_plot_param_min, current_plot_param_max)
            ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
            continue

        x_data = plot_data['x_data']
        y_data = plot_data['y_data']

        param_bins_local = np.linspace(current_plot_param_min, current_plot_param_max, bins + 1)

        h_counts, _, _, im_current = ax.hist2d(
            x_data, y_data,
            bins=[param_bins_local, te_bins], cmap='viridis',
            norm=LogNorm(vmin=1, vmax=max_counts_overall),
            cmin=1
        )
        if np.any(h_counts > 0): im = im_current

        ax.set_xlabel(param_label, fontsize=font_size_label)
        ax.set_xlim(current_plot_param_min, current_plot_param_max)
        ax.tick_params(axis='both', which='major', labelsize=font_size_tick)

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Observations', fontsize=font_size_label)
        cbar.ax.tick_params(labelsize=font_size_tick)

    plt.suptitle(f'Distribution of {TE_DISPLAY_NAME} Observations', fontsize=font_size_suptitle, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.15, wspace=0.15)

    plt.savefig("electron_temperature_distribution.png", dpi=300, bbox_inches='tight')
    print("Saved plot to electron_temperature_distribution.png")
    plt.show()


def plot_altitude_vs_te_profile(df, te_col, ilat_col, alt_col, gmlt_col, verbose=False):
    """Generates line plots of Altitude vs. Electron Temperature for ILAT 40-50."""
    print("\n--- Generating Altitude vs. Electron Temperature Profile Plots ---")

    required_cols = [te_col, ilat_col, alt_col, gmlt_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}. Skipping.")
        return

    def _generate_single_profile_plot(data_subset, plot_title, output_filename):
        if data_subset.empty:
            print(f"No data available for '{plot_title}'. Skipping.")
            return

        print(f"\nProcessing plot: '{plot_title}' ({len(data_subset)} data points)")

        alt_min, alt_max = data_subset[alt_col].min(), data_subset[alt_col].max()
        bin_width = 20
        start_bin = np.floor(alt_min / bin_width) * bin_width
        bins = np.arange(start_bin, alt_max + bin_width, bin_width)

        data_subset = data_subset.copy()
        data_subset['alt_bin'] = pd.cut(data_subset[alt_col], bins=bins, right=False)

        profile_data = data_subset.groupby('alt_bin', observed=True).agg(
            mean_te=(te_col, 'mean'),
            min_te=(te_col, 'min'),
            max_te=(te_col, 'max'),
            mean_alt=(alt_col, 'mean'),
            count=('alt_bin', 'size')
        ).dropna()

        min_obs_per_bin = 10
        profile_data = profile_data[profile_data['count'] >= min_obs_per_bin]

        if profile_data.empty:
            print(f"No bins with >= {min_obs_per_bin} observations for '{plot_title}'.")
            return

        if verbose:
            print("--- Verbose Bin Information ---")
            for index, row in profile_data.iterrows():
                print(f"[{index.left}, {index.right}) | {int(row['count']):>5} | "
                      f"mean={row['mean_te']:.2f} min={row['min_te']:.2f} max={row['max_te']:.2f}")

        x_errors = [
            profile_data['mean_te'] - profile_data['min_te'],
            profile_data['max_te'] - profile_data['mean_te']
        ]

        plt.figure(figsize=(8, 10))
        plt.errorbar(
            x=profile_data['mean_te'], y=profile_data['mean_alt'],
            xerr=x_errors, marker='o', linestyle='-', capsize=3
        )
        plt.ylabel('Altitude (km)', fontsize=14)
        plt.xlabel(f'Electron Temperature ({TE_YAXIS_LABEL}) (K)', fontsize=14)
        plt.title(plot_title, fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_filename}")
        plt.show()

    print("Filtering data for ILAT between 40 and 50...")
    df_base_filtered = df[(df[ilat_col] >= 40) & (df[ilat_col] <= 50)].copy()
    df_base_filtered = df_base_filtered.dropna(subset=[te_col, alt_col, gmlt_col])
    df_base_filtered = df_base_filtered[(df_base_filtered[te_col] > 0) & (df_base_filtered[alt_col] > 0)]

    if df_base_filtered.empty:
        print("No data for ILAT 40-50 after cleaning.")
        return

    plot_configs = [
        {
            "filter": (df_base_filtered[gmlt_col] > 10) & (df_base_filtered[gmlt_col] < 17),
            "title": "Daytime Altitude vs Electron Temperature (ILAT 40-50)",
            "filename": "altitude_vs_te_profile_day.png"
        },
        {
            "filter": (df_base_filtered[gmlt_col] > 22) | (df_base_filtered[gmlt_col] < 4),
            "title": "Nighttime Altitude vs Electron Temperature (ILAT 40-50)",
            "filename": "altitude_vs_te_profile_night.png"
        },
        {
            "filter": pd.Series(True, index=df_base_filtered.index),
            "title": "Altitude vs Electron Temperature (ILAT 40-50)",
            "filename": "altitude_vs_te_profile_all.png"
        }
    ]

    for config in plot_configs:
        df_for_plot = df_base_filtered[config["filter"]]
        _generate_single_profile_plot(df_for_plot, config["title"], config["filename"])


if __name__ == "__main__":
    try:
        ALL_NECESSARY_COLUMNS = list(set([
            TE_COLUMN, ILAT_COLUMN_NAME, GLAT_COLUMN_NAME,
            GMLT_COLUMN_NAME, ALTITUDE_COLUMN_NAME
        ]))

        combined_df = load_and_combine_datasets(
            BASE_DATASET_DIR,
            columns_to_keep=ALL_NECESSARY_COLUMNS,
            max_rows_to_convert=MAX_ROWS_FOR_PANDAS_CONVERSION
        )

        if combined_df.empty:
            raise ValueError("Loaded DataFrame is empty.")

        if ILAT_COLUMN_NAME in combined_df.columns:
            combined_df[L_SHELL_COLUMN_NAME] = calculate_l_shell(combined_df[ILAT_COLUMN_NAME])

        param_cols_to_plot = [L_SHELL_COLUMN_NAME, GLAT_COLUMN_NAME, GMLT_COLUMN_NAME]
        param_labels_to_plot = ['L-Shell', 'MLAT [deg]', f'{GMLT_COLUMN_NAME} [hr]']
        param_ranges_to_plot = [(1.0, 12.0), (-90.0, 90.0), (0.0, 24.0)]

        plot_observation_heatmaps(
            combined_df,
            te_col=TE_COLUMN,
            param_cols=param_cols_to_plot,
            param_labels=param_labels_to_plot,
            param_ranges=param_ranges_to_plot,
            bins=BINS
        )

        plot_altitude_vs_te_profile(
            df=combined_df,
            te_col=TE_COLUMN,
            ilat_col=ILAT_COLUMN_NAME,
            alt_col=ALTITUDE_COLUMN_NAME,
            gmlt_col=GMLT_COLUMN_NAME,
            verbose=True
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Data Error: {e}")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())
