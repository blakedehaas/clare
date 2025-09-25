import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects

from datasets import load_from_disk, concatenate_datasets, Dataset
from tqdm import tqdm

# --- Define paths and constants ---

BASE_DATASET_DIR = os.path.join("../..", "dataset", "processed_dataset_01_31_storm") 

TE_COLUMN = 'Te1'
TE_DISPLAY_NAME = 'Electron Temperature'
TE_YAXIS_LABEL = 'Te'

BINS = 100 # Number of bins for histograms

ILAT_COLUMN_NAME = 'ILAT'
L_SHELL_COLUMN_NAME = 'L_SHELL'
GLAT_COLUMN_NAME = 'GLAT'
GMLT_COLUMN_NAME = 'GMLT'
ALTITUDE_COLUMN_NAME = 'Altitude' 

# Maximum rows to convert to Pandas. If dataset is larger, it will be sampled.
# Set to None to disable sampling and try to load all data (after column selection).
MAX_ROWS_FOR_PANDAS_CONVERSION = None

# --- Helper function ---
def find_dataset_dirs(base_dir):
    """Finds all directories containing dataset_info.json recursively."""
    dataset_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'dataset_info.json' in files:
            # Ensure we don't pick up the base_dir itself if it accidentally has dataset_info.json
            if os.path.abspath(root) != os.path.abspath(base_dir):
                dataset_dirs.append(root)
    return dataset_dirs

# --- L-shell Calculation Function ---
def calculate_l_shell(ilat_deg):
    """
    Calculates L-shell from Invariant Latitude (ILAT).
    """
    if not isinstance(ilat_deg, pd.Series):
        ilat_deg = pd.Series(ilat_deg, dtype=float)

    # Ensure input is float for calculations
    ilat_deg_float = ilat_deg.astype(float)
    ilat_rad = np.deg2rad(ilat_deg_float)
    cos_ilat_rad = np.cos(ilat_rad)
    cos_ilat_rad_sq = cos_ilat_rad**2

    with np.errstate(divide='ignore', invalid='ignore'):
        l_shell = 1.0 / cos_ilat_rad_sq
    
    # Replace inf with NaN and handle near-pole issues
    l_shell[np.isinf(l_shell) | (cos_ilat_rad_sq < 1e-9) | ilat_deg_float.abs().ge(90)] = np.nan
    return l_shell


# --- Data Loading Function ---
def load_and_combine_datasets(base_dir, columns_to_keep=None, max_rows_to_convert=None):
    """
    Loads dataset(s) from the specified base directory.
    Selects specified columns and optionally samples before converting to Pandas.
    """
    all_datasets = []
    dataset_info_path = os.path.join(base_dir, 'dataset_info.json')

    if os.path.exists(dataset_info_path):
        print(f"Found dataset info file directly in: {base_dir}")
        print("Loading single dataset...")
        try:
            ds = load_from_disk(base_dir)
            if isinstance(ds, dict):
                print(f"Warning: Found DatasetDict in {base_dir}, attempting to extract 'train' split.")
                if 'train' in ds: ds = ds['train']
                else:
                    first_split = next(iter(ds.keys()), None)
                    if first_split:
                        print(f"Warning: Using first available split '{first_split}' from DatasetDict.")
                        ds = ds[first_split]
                    else: raise ValueError(f"Error: Could not find a usable split in DatasetDict at {base_dir}.")
            all_datasets.append(ds)
            print(f"Successfully loaded dataset with {len(ds)} entries.")
        except Exception as e:
            print(f"Error loading single dataset from {base_dir}: {e}")
            raise
    else:
        print(f"No dataset_info.json found directly in {base_dir}. Searching for datasets recursively...")
        dataset_dirs = find_dataset_dirs(base_dir)

        if not dataset_dirs:
            raise FileNotFoundError(f"No datasets found recursively under {base_dir}. ")

        print(f"Found {len(dataset_dirs)} dataset directories to load.")
        for dir_path in tqdm(dataset_dirs, desc="Loading datasets"):
            try:
                ds = load_from_disk(dir_path)
                if isinstance(ds, dict): # Handle DatasetDict
                    if 'train' in ds: ds = ds['train']
                    else:
                        first_split = next(iter(ds.keys()), None)
                        if first_split: ds = ds[first_split]
                        else:
                            print(f"Error: Could not find a usable split in DatasetDict at {dir_path}. Skipping.")
                            continue
                all_datasets.append(ds)
            except Exception as e:
                print(f"Error loading dataset from {dir_path}: {e}. Skipping.")

    if not all_datasets: raise ValueError("No datasets could be successfully loaded.")

    if len(all_datasets) == 1:
        print("Using the single loaded dataset.")
        combined_dataset = all_datasets[0]
    else:
        print("Combining loaded datasets...")
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"Combined dataset contains {len(combined_dataset)} entries.")

    dataset_for_pandas = combined_dataset
    print(f"Original combined dataset columns: {dataset_for_pandas.column_names}")

    if columns_to_keep:
        actual_cols_to_keep_in_dataset = [col for col in columns_to_keep if col in dataset_for_pandas.column_names]
        
        missing_requested_cols = [col for col in columns_to_keep if col not in actual_cols_to_keep_in_dataset]
        if missing_requested_cols:
            print(f"Warning: The following requested columns are not in the loaded dataset and will be ignored: {missing_requested_cols}")

        if not actual_cols_to_keep_in_dataset:
            print("Warning: None of the requested columns_to_keep are present in the dataset. Proceeding with all columns.")
        elif set(actual_cols_to_keep_in_dataset) == set(dataset_for_pandas.column_names):
            print("All dataset columns are among those requested. No subset selection performed.")
        else:
            print(f"Attempting to select columns: {actual_cols_to_keep_in_dataset}")
            try:
                dataset_for_pandas = dataset_for_pandas.select_columns(actual_cols_to_keep_in_dataset)
                print(f"Successfully selected columns. New dataset columns: {dataset_for_pandas.column_names}")
            except Exception as e:
                print(f"Error during select_columns: {e}. Proceeding with previously defined set of columns for Pandas conversion.")
    else:
        print("No specific columns_to_keep requested. Proceeding with all columns from dataset.")

    if max_rows_to_convert is not None and len(dataset_for_pandas) > max_rows_to_convert:
        print(f"Dataset has {len(dataset_for_pandas)} rows, which is > max_rows_to_convert ({max_rows_to_convert}). Sampling down...")
        dataset_for_pandas = dataset_for_pandas.shuffle(seed=42).select(range(max_rows_to_convert))
        print(f"Sampled down to {len(dataset_for_pandas)} rows.")
    
    print(f"Converting {len(dataset_for_pandas)} rows with columns {dataset_for_pandas.column_names} to Pandas DataFrame...")
    df = dataset_for_pandas.to_pandas()
    print("DataFrame conversion complete.")
    print("DataFrame memory usage:")
    df.info(memory_usage='deep')
    return df


# --- Plotting Function ---
def plot_observation_heatmaps(df, te_col, param_cols, param_labels, param_ranges, bins=50):
    df_filtered_te = df[df[te_col].notna() & (df[te_col] > 0)].copy()
    if len(df_filtered_te) == 0:
        print(f"Warning: No data remaining after filtering for positive and non-NaN {te_col}. Cannot plot.")
        return
    if len(df_filtered_te) < len(df):
        print(f"Filtered out {len(df) - len(df_filtered_te)} rows with non-positive or NaN {te_col} values.")

    min_te_percentile = np.percentile(df_filtered_te[te_col], 1)
    max_te_percentile = np.percentile(df_filtered_te[te_col], 99.9)

    min_te = max(0, min_te_percentile)
    max_te = max_te_percentile

    if max_te <= min_te:
        print(f"Warning: max_te ({max_te:.2f}) <= min_te ({min_te:.2f}) after percentile calculation. Adjusting max_te.")
        max_te = min_te * 10 if min_te > 0 else 1000
        if max_te <= min_te : max_te = min_te + 1000

    print(f"Dynamically determined {TE_YAXIS_LABEL} y-axis range: {min_te:.2f} to {max_te:.2f} K")
    te_bins = np.linspace(min_te, max_te, bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True) # 1 row, 3 columns, shared Y-axis
    plot_panel_labels = ['(a)', '(b)', '(c)']
    text_outline_effect = [path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()]
    im = None # To store the mappable for the colorbar

    max_counts_overall = 0
    hist_plot_data = [] 

    # --- Define font sizes ---
    font_size_label = 28
    font_size_tick = 18
    font_size_panel_label = 28
    font_size_suptitle = 28

    for i, (param_col, param_range) in enumerate(zip(param_cols, param_ranges)):
        current_data = {'param_col': param_col, 'param_range': param_range, 'x_data': None, 'y_data': None, 'valid': False}
        if param_col not in df_filtered_te.columns:
            print(f"Warning: Parameter column '{param_col}' not found for panel {plot_panel_labels[i]}. Skipping this panel's count calculation.")
            hist_plot_data.append(current_data)
            continue

        current_param_min_cfg, current_param_max_cfg = param_range # Configured range
        
        # Ensure range is valid before filtering data with it
        if not (isinstance(current_param_min_cfg, (int, float)) and 
                isinstance(current_param_max_cfg, (int, float)) and 
                current_param_max_cfg > current_param_min_cfg):
            print(f"Warning: Invalid or non-numeric range for '{param_col}': {param_range}. Attempting to use data min/max.")
            # Fallback: Use actual min/max of the column if range is bad
            if df_filtered_te[param_col].notna().any():
                 actual_min = df_filtered_te[param_col].min()
                 actual_max = df_filtered_te[param_col].max()
                 if actual_max > actual_min:
                     current_param_min_data, current_param_max_data = actual_min, actual_max
                     print(f"Using data range for '{param_col}': ({current_param_min_data:.2f}, {current_param_max_data:.2f})")
                     current_data['param_range'] = (current_param_min_data, current_param_max_data)
                 else:
                     print(f"Could not determine valid data range for '{param_col}'. Skipping count calc.")
                     hist_plot_data.append(current_data)
                     continue
            else:
                print(f"No data for '{param_col}' to determine range. Skipping count calc.")
                hist_plot_data.append(current_data)
                continue
        else: # Configured range is good
            current_param_min_data, current_param_max_data = current_param_min_cfg, current_param_max_cfg


        valid_param_data_mask = (
            df_filtered_te[param_col].notna() &
            np.isfinite(df_filtered_te[param_col]) &
            (df_filtered_te[param_col] >= current_param_min_data) &
            (df_filtered_te[param_col] <= current_param_max_data)
        )
        x_data_temp = df_filtered_te.loc[valid_param_data_mask, param_col]
        y_data_temp = df_filtered_te.loc[valid_param_data_mask, te_col]

        if not x_data_temp.empty and not y_data_temp.empty:
            param_bins_temp = np.linspace(current_param_min_data, current_param_max_data, bins + 1)
            counts_temp, _, _ = np.histogram2d(x_data_temp, y_data_temp, bins=[param_bins_temp, te_bins])
            max_counts_overall = max(max_counts_overall, counts_temp.max() if counts_temp.size > 0 else 0)
            current_data['x_data'] = x_data_temp
            current_data['y_data'] = y_data_temp
            current_data['valid'] = True
            current_data['param_range'] = (current_param_min_data, current_param_max_data)
        else:
            print(f"Warning: No valid data for '{param_col}' in range {current_param_min_data, current_param_max_data} for panel {plot_panel_labels[i]}. Skipping count calc.")
        hist_plot_data.append(current_data)

    if max_counts_overall == 0:
        max_counts_overall = 1 
        print("Warning: max_counts_overall is 0. No data to plot in any panel or all counts are zero.")

    for i, ax in enumerate(axes):
        param_col_name = param_cols[i]
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
            ax.text(0.5, 0.5, f"No valid data for\n'{param_col_name}'\nin range ({current_plot_param_min:.2f}, {current_plot_param_max:.2f})",
                    ha='center', va='center', transform=ax.transAxes, color='red', fontsize=9)
            # --- Increased font size ---
            ax.set_xlabel(param_label, fontsize=font_size_label)
            ax.set_xlim(current_plot_param_min, current_plot_param_max)
            # --- Increased font size for tick labels ---
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

        # --- Increased font size ---
        ax.set_xlabel(param_label, fontsize=font_size_label)
        ax.set_xlim(current_plot_param_min, current_plot_param_max)
        # --- Increased font size for tick labels ---
        ax.tick_params(axis='both', which='major', labelsize=font_size_tick)


    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
        cbar = fig.colorbar(im, cax=cbar_ax)
        # --- Increased font size for colorbar label and ticks ---
        cbar.set_label('Observations', fontsize=font_size_label)
        cbar.ax.tick_params(labelsize=font_size_tick)
    else:
        print("Skipping colorbar as no valid plot generated a mappable image.")

    plt.suptitle(f'Distribution of {TE_DISPLAY_NAME} Observations', fontsize=font_size_suptitle, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.15, wspace=0.15) 

    plt.savefig("electron_temperature_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to electron_temperature_distribution.png")
    plt.show()

# --- PLOTTING FUNCTION ---
def plot_altitude_vs_te_profile(df, te_col, ilat_col, alt_col, gmlt_col, verbose=False):
    """
    Generates three line plots of Altitude vs. Electron Temperature for a specific ILAT range,
    separated by Day, Night, and All GMLT conditions, with min/max error bars.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        te_col (str): The name of the electron temperature column.
        ilat_col (str): The name of the invariant latitude column.
        alt_col (str): The name of the altitude column.
        gmlt_col (str): The name of the GMLT column for day/night filtering.
        verbose (bool): If True, prints detailed information about each bin.
    """
    print("\n--- Generating Altitude vs. Electron Temperature Profile Plots ---")

    # --- 1. Check for required columns ---
    required_cols = [te_col, ilat_col, alt_col, gmlt_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns for profile plots: {missing_cols}. Skipping plot generation.")
        return

    # --- 2. Inner helper function to perform binning and plotting for a given dataset ---
    def _generate_single_profile_plot(data_subset, plot_title, output_filename):
        """Helper to bin data, optionally print verbose info, and create a single plot."""
        if data_subset.empty:
            print(f"No data available for '{plot_title}'. Skipping this plot.")
            return

        print(f"\nProcessing plot: '{plot_title}' ({len(data_subset)} data points)")

        # Binning and aggregation
        alt_min, alt_max = data_subset[alt_col].min(), data_subset[alt_col].max()
        bin_width = 20  # km
        start_bin = np.floor(alt_min / bin_width) * bin_width
        bins = np.arange(start_bin, alt_max + bin_width, bin_width)
        
        data_subset = data_subset.copy()
        data_subset['alt_bin'] = pd.cut(data_subset[alt_col], bins=bins, right=False)
        
        # Add 'min' and 'max' to the aggregation
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
            print(f"No altitude bins with at least {min_obs_per_bin} observations found for '{plot_title}'. Cannot generate plot.")
            return

        # Verbose output now includes min and max Te
        if verbose:
            print("--- Verbose Bin Information ---")
            header = (f"{'Altitude Bin (km)':<22} | {'Data Points':^15} | "
                      f"{'Mean Te (°K)':>18} | {'Min Te (°K)':>15} | {'Max Te (°K)':>15}")
            print(header)
            print("-" * len(header))
            for index, row in profile_data.iterrows():
                altitude_range_str = f"[{index.left}, {index.right})"
                count = int(row['count'])
                mean_te, min_te, max_te = row['mean_te'], row['min_te'], row['max_te']
                print(f"{altitude_range_str:<22} | {count:^15} | "
                      f"{mean_te:>18.2f} | {min_te:>15.2f} | {max_te:>15.2f}")
            print("-" * len(header))

        font_size_label = 14
        font_size_title = 16
        font_size_tick = 12

        # Use plt.errorbar to plot with horizontal bars
        # Calculate the error values (distance from mean to min/max)
        x_errors = [
            profile_data['mean_te'] - profile_data['min_te'],
            profile_data['max_te'] - profile_data['mean_te']
        ]

        plt.figure(figsize=(8, 10))
        plt.errorbar(
            x=profile_data['mean_te'], 
            y=profile_data['mean_alt'], 
            xerr=x_errors, 
            marker='o', 
            linestyle='-',
            capsize=3  # Adds caps to the error bars
        )
        
        # --- Plot configuration ---
        plt.ylabel('Altitude (km)', fontsize=font_size_label)
        plt.xlabel(f'Electron Temperature ({TE_YAXIS_LABEL}) (°K)', fontsize=font_size_label)
        plt.title(plot_title, fontsize=font_size_title)
        plt.tick_params(axis='both', which='major', labelsize=font_size_tick)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_filename}")
        plt.show()

    # --- 3. Prepare base filtered data (ILAT and cleaning) ---
    print("Filtering data for ILAT between 40 and 50...")
    df_base_filtered = df[(df[ilat_col] >= 40) & (df[ilat_col] <= 50)].copy()
    df_base_filtered = df_base_filtered.dropna(subset=[te_col, alt_col, gmlt_col])
    df_base_filtered = df_base_filtered[(df_base_filtered[te_col] > 0) & (df_base_filtered[alt_col] > 0)]

    if df_base_filtered.empty:
        print("No data available for the specified ILAT range (40-50) after initial cleaning. Cannot generate any plots.")
        return

    # --- 4. Define plot configurations and loop through them ---
    plot_configs = [
        {
            "condition": "Daytime",
            "filter": (df_base_filtered[gmlt_col] > 10) & (df_base_filtered[gmlt_col] < 17),
            "title": "Daytime Altitude vs Electron Temperature (ILAT 40-50)",
            "filename": "altitude_vs_te_profile_day.png"
        },
        {
            "condition": "Nighttime",
            "filter": (df_base_filtered[gmlt_col] > 22) | (df_base_filtered[gmlt_col] < 4),
            "title": "Nighttime Altitude vs Electron Temperature (ILAT 40-50)",
            "filename": "altitude_vs_te_profile_night.png"
        },
        {
            "condition": "All Times",
            "filter": pd.Series(True, index=df_base_filtered.index), # No GMLT filter
            "title": "Altitude vs Electron Temperature (ILAT 40-50)",
            "filename": "altitude_vs_te_profile_all.png"
        }
    ]

    for config in plot_configs:
        df_for_plot = df_base_filtered[config["filter"]]
        _generate_single_profile_plot(df_for_plot, config["title"], config["filename"])


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Define ALL columns that will be absolutely needed from the dataset
        ALL_NECESSARY_COLUMNS = list(set([
            TE_COLUMN,
            ILAT_COLUMN_NAME,
            GLAT_COLUMN_NAME,
            GMLT_COLUMN_NAME,
            ALTITUDE_COLUMN_NAME 
        ]))
        
        combined_df = load_and_combine_datasets(
            BASE_DATASET_DIR,
            columns_to_keep=ALL_NECESSARY_COLUMNS,
            max_rows_to_convert=MAX_ROWS_FOR_PANDAS_CONVERSION
        )

        if combined_df.empty:
            raise ValueError("Loaded DataFrame is empty. Cannot proceed.")

        # L-shell calculation
        if ILAT_COLUMN_NAME in combined_df.columns:
            print(f"\nCalculating L-shell using '{ILAT_COLUMN_NAME}' column...")
            combined_df[L_SHELL_COLUMN_NAME] = calculate_l_shell(combined_df[ILAT_COLUMN_NAME])
            print(f"L-shell calculation complete. '{L_SHELL_COLUMN_NAME}' column added/updated.")
        else:
            print(f"Warning: ILAT column '{ILAT_COLUMN_NAME}' not found in DataFrame after loading. Cannot calculate L-shell.")
            if L_SHELL_COLUMN_NAME not in combined_df.columns:
                 combined_df[L_SHELL_COLUMN_NAME] = np.nan

        # Define parameters for the 3-panel plot
        param_cols_to_plot = [L_SHELL_COLUMN_NAME, GLAT_COLUMN_NAME, GMLT_COLUMN_NAME]
        param_labels_to_plot = [
            'L-Shell',
            'MLAT [deg]',
            f'{GMLT_COLUMN_NAME} [hr]'
        ]
        param_ranges_to_plot = [
            (1.0, 12.0),
            (-90.0, 90.0),
            (0.0, 24.0)
        ]

        # Check for required columns for heatmap plot
        required_cols_in_df_for_heatmap = [TE_COLUMN] + param_cols_to_plot
        missing_cols_in_df_heatmap = [col for col in required_cols_in_df_for_heatmap if col not in combined_df.columns]

        if missing_cols_in_df_heatmap:
            raise ValueError(f"Missing required columns in the DataFrame for heatmap plotting: {missing_cols_in_df_heatmap}. "
                             f"Available columns: {combined_df.columns.tolist()}")

        print(f"\nDataFrame Info for plotting columns ({TE_YAXIS_LABEL} from '{TE_COLUMN}' column):")
        cols_to_show_info_df = list(set(required_cols_in_df_for_heatmap + [L_SHELL_COLUMN_NAME, ALTITUDE_COLUMN_NAME]))
        cols_to_show_info_df = [col for col in cols_to_show_info_df if col in combined_df.columns]
        
        if cols_to_show_info_df:
            df_for_describe = combined_df[cols_to_show_info_df].copy()
            if L_SHELL_COLUMN_NAME in df_for_describe.columns:
                 df_for_describe[L_SHELL_COLUMN_NAME] = df_for_describe[L_SHELL_COLUMN_NAME].replace([np.inf, -np.inf], np.nan)

            print("\n--- Column Info (subset for plotting) ---")
            df_for_describe.info(verbose=True, show_counts=True)
            print("\n--- Basic Statistics (subset for plotting) ---")
            print(df_for_describe.describe(percentiles=[.005, .01, .05, .25, .5, .75, .95, .99, .995, .999]))
        
        # --- Generate heatmap plots ---
        plot_observation_heatmaps(
            combined_df,
            te_col=TE_COLUMN,
            param_cols=param_cols_to_plot,
            param_labels=param_labels_to_plot,
            param_ranges=param_ranges_to_plot,
            bins=BINS
        )

        # --- Generate the new Altitude vs Te Profile Plots (Day, Night, All) ---
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