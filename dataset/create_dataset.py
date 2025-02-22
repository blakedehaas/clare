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
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from scipy.spatial.distance import cdist

GENERATE_VISUALIZATIONS = True  # Flag to control visualization generation
VISUALIZATION_DIR = os.path.join('visualizations')  # Visualization output directory
NUM_CLUSTERS = 5

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

def add_solar_storm_classification(filtered_df, generate_viz=True, use_known_storm_centroids=False):
    """Add solar storm classification based on indices and known storm period."""
    print("Adding solar storm classification...")
    
    # Create visualization directory if it doesn't exist and visualization is enabled
    if generate_viz:
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Print available columns in the filtered_df to debug
    print("Available columns in filtered_df:", filtered_df.columns.tolist())
    
    # Get all features for clustering
    solar_features = pd.DataFrame({
        'SYM_H': filtered_df['SYM_H_0'],
        'AL_index': filtered_df['AL_index_0'],
        'f107': filtered_df['f107_index_0']
    })
    
    # Normalize features
    scaler = MinMaxScaler()
    solar_features_scaled = pd.DataFrame(
        scaler.fit_transform(solar_features),
        columns=solar_features.columns,
        index=filtered_df.index
    )
    
    # Generate elbow plot to determine optimal k
    #generate_elbow_plot(solar_features_scaled)

    # Create known storm period mask
    known_storm_mask = (filtered_df.index >= '1991-06-02') & (filtered_df.index < '1991-06-08')
    
    # Perform K-means clustering with 5 clusters
    if use_known_storm_centroids:
        # Use known storm data to initialize centroids
        known_storm_data = solar_features_scaled[known_storm_mask]
        kmeans = KMeans(n_clusters=NUM_CLUSTERS , init=known_storm_data.mean().values.reshape(1, -1), n_init=1, verbose=True)
    else:
        kmeans = KMeans(n_clusters=NUM_CLUSTERS , verbose=True)
    
    cluster_labels = kmeans.fit_predict(solar_features_scaled)
    
    # Find which cluster contains most of the known storm period
    storm_period_clusters = cluster_labels[known_storm_mask]
    high_intensity_cluster = pd.Series(storm_period_clusters).mode()[0]
    
    # After fitting the K-means model
    centroids = kmeans.cluster_centers_

    # Get the centroid of the high intensity cluster
    high_intensity_centroid = centroids[high_intensity_cluster]

    # Calculate distances from the high intensity centroid
    distances_from_high_intensity = cdist(centroids, [high_intensity_centroid], metric='euclidean').flatten()

    # Sort clusters based on distance
    sorted_clusters = sorted(range(NUM_CLUSTERS), key=lambda i: distances_from_high_intensity[i])

    # Assign intensity levels based on sorted order
    intensity_mapping = {cluster: NUM_CLUSTERS - 1 - idx for idx, cluster in enumerate(sorted_clusters)}

    # Print the intensity mapping for verification
    print("Intensity Mapping:", intensity_mapping)
    
    filtered_df['solar_storm_intensity'] = [intensity_mapping[label] for label in cluster_labels]
    
    # Create visualization if enabled
    if generate_viz:
        print("Generating 3D clustering visualization...")
        
        # Count the number of points in each cluster
        cluster_counts = filtered_df['solar_storm_intensity'].value_counts()
        print("Cluster counts:", cluster_counts.to_dict())
        
        # Create a mask for the known storm period
        known_storm_mask = (filtered_df.index >= '1991-06-02') & (filtered_df.index < '1991-06-08')
        
        # Filter out known storm period data
        non_storm_data = filtered_df[~known_storm_mask]
        
        # Sample 100,000 points from the non-storm data
        sampled_points = non_storm_data.sample(n=100000)
        print(f"Total points sampled for visualization (excluding known storm): {len(sampled_points)}")
        
        # Check if sampled_points is empty
        if sampled_points.empty:
            print("Warning: No points available for visualization.")
            return
        
        # Combine sampled points with all known storm period points
        known_storm_data = filtered_df[known_storm_mask]
        combined_points = pd.concat([sampled_points, known_storm_data])
        print(f"Total points for visualization (including known storm): {len(combined_points)}")
        
        # Normalize the three solar indices for the combined dataset
        scaler = MinMaxScaler()
        combined_points[['f107_index_0', 'AL_index_0', 'SYM_H_0']] = scaler.fit_transform(
            combined_points[['f107_index_0', 'AL_index_0', 'SYM_H_0']]
        )
        
        # Normalize the known storm data using the same scaler
        known_storm_data.loc[:, ['f107_index_0', 'AL_index_0', 'SYM_H_0']] = scaler.transform(
            known_storm_data[['f107_index_0', 'AL_index_0', 'SYM_H_0']]
        )
        
        # Define colors for each cluster
        cluster_colors = {
            0: 'blue',
            1: 'yellow',
            2: 'orange',
            3: 'red',
            4: 'purple'
        }
        
        # Create the 3D scatter plot using go.Figure
        fig = go.Figure()

        # Add scatter3d trace for each cluster individually
        for label in range(NUM_CLUSTERS):
            cluster_data = combined_points[combined_points['solar_storm_intensity'] == label]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['f107_index_0'],
                y=cluster_data['AL_index_0'],
                z=cluster_data['SYM_H_0'],
                mode='markers',
                marker=dict(
                    size=2,  # Set all points to size 2
                    color=cluster_colors[label],  # Use the color for the specific cluster
                    line=dict(width=0)  # Remove the white outline
                ),
                name=f'Cluster ({intensity_mapping[label]})',  # Label with cluster number and intensity
                opacity=0.7  # Set opacity directly in the trace
            ))

        # Set layout properties
        fig.update_layout(
            title='Solar Storm Intensity Clustering (Normalized)',
            scene=dict(
                xaxis_title='f107_index',
                yaxis_title='AL_index',
                zaxis_title='SYM_H'
            )
        )
        
        # Highlight known storm period using the combined dataset
        fig.add_scatter3d(
            x=known_storm_data['f107_index_0'],
            y=known_storm_data['AL_index_0'],
            z=known_storm_data['SYM_H_0'],
            mode='markers',
            marker=dict(color='green', size=3),  # Set size to 3 for smaller points
            name='Known Storm Period (June 2-7, 1991)'
        )
        
        # Plot centroids with corresponding colors
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers',
            marker=dict(color=[cluster_colors[label] for label in range(NUM_CLUSTERS)], size=5, symbol='x'),
            name='Centroids'
        ))
        
        # Save the visualization
        viz_path = os.path.join(VISUALIZATION_DIR, 'solar_storm_clustering.html')
        fig.write_html(viz_path)
        print(f"Visualization saved to: {viz_path}")
    
    # Print statistics
    print("\nSolar Storm Intensity Distribution:")
    print(filtered_df['solar_storm_intensity'].value_counts().sort_index())
    
    return filtered_df

def generate_elbow_plot(data, k_range=(2, 10), generate_plot=True):
    """Generate an elbow plot to determine the optimal number of clusters (k) for K-means."""
    inertia = []
    
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, verbose=False)  # Set verbose to False for cleaner output
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    if generate_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(k_range[0], k_range[1] + 1), inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(range(k_range[0], k_range[1] + 1))
        plt.grid()
        plt.show()

# Define known solar storm periods for training dataset
def create_training_dataset():
    """Create a training dataset of known solar storms."""
    training_data = []
    
    # Define known storm periods with intensity labels
    storm_periods = [
        {"storm_id": "0", "storm_start_date": "1990-06-08", "storm_end_date": "1990-06-11", "storm_intensity": "5", "storm_name": "Solar Storm A"},
        {"storm_id": "1", "storm_start_date": "1991-03-24", "storm_end_date": "1991-03-27", "storm_intensity": "4", "storm_name": "Solar Storm B"},
        {"storm_id": "2", "storm_start_date": "1992-05-15", "storm_end_date": "1992-05-18", "storm_intensity": "3", "storm_name": "Solar Storm C"},
        {"storm_id": "3", "storm_start_date": "1995-08-01", "storm_end_date": "1995-08-03", "storm_intensity": "2", "storm_name": "Solar Storm D"},
        {"storm_id": "4", "storm_start_date": "1998-11-15", "storm_end_date": "1998-11-18", "storm_intensity": "1", "storm_name": "Solar Storm E"},
    ]
    
    # Create entries for each day in the storm periods
    for storm in storm_periods:
        start_date = pd.to_datetime(storm["storm_start_date"])
        end_date = pd.to_datetime(storm["storm_end_date"])
        date_range = pd.date_range(start=start_date, end=end_date)
        
        for date in date_range:
            training_data.append({
                "storm_id": storm["storm_id"],
                "storm_date": date.strftime('%Y-%m-%d'),
                "storm_intensity": storm["storm_intensity"],
                "storm_name": storm["storm_name"]
            })
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Save the training dataset
    output_path = os.path.join(VISUALIZATION_DIR, 'solar_storm_training_dataset.csv')
    training_df.to_csv(output_path, index=False)
    print(f"Training dataset saved to: {output_path}")

# Call the function to create the training dataset
create_training_dataset()

# Define updated relative paths with better error handling
akebono_file_path = 'Akebono_combined.tsv'
omni_al_symh_path = os.path.join('omni_al_index_symh', '*.lst')  # Use os.path.join for path handling
f107_file_path = os.path.join('omni_f107', '*.lst')  # Use os.path.join for path handling
train_output_path = 'akebono_solar_combined_train'
val_output_path = 'akebono_solar_combined_val'
test_output_path = 'akebono_solar_combined_test'

# Add file existence checks
def check_data_files():
    """Check if required data files exist and return proper paths."""
    import glob
    
    # Check Akebono file
    if not os.path.exists(akebono_file_path):
        raise FileNotFoundError(f"Akebono data file not found at: {akebono_file_path}")
    
    # Check OMNI AL and SYM-H directory exists
    omni_dir = os.path.dirname(omni_al_symh_path)
    if not os.path.exists(omni_dir):
        raise FileNotFoundError(f"OMNI AL/SYM-H directory not found at: {omni_dir}")
    
    # Check F10.7 directory exists
    f107_dir = os.path.dirname(f107_file_path)
    if not os.path.exists(f107_dir):
        raise FileNotFoundError(f"F10.7 directory not found at: {f107_dir}")
    
    # Get list of files
    al_symh_files = glob.glob(omni_al_symh_path)
    f107_files = glob.glob(f107_file_path)
    
    if not al_symh_files:
        raise FileNotFoundError(f"No .lst files found in: {omni_dir}")
    if not f107_files:
        raise FileNotFoundError(f"No .lst files found in: {f107_dir}")
    
    print("Data files found:")
    print(f"Akebono: {akebono_file_path}")
    print(f"OMNI AL/SYM-H files: {len(al_symh_files)} files")
    print(f"F10.7 files: {len(f107_files)} files")
    
    return al_symh_files, f107_files

# Add this near the start of your script
al_symh_files, f107_files = check_data_files()

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

# Update the file reading section
df_list = []

for file in tqdm(al_symh_files, desc="Loading omni_al_index_symh data"):
    try:
        # Define column names
        columns = ['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H']
        
        # Read the OMNI data file with fixed whitespace separator
        df = pd.read_csv(
            file, 
            sep=r'\s+',  # Use raw string for separator
            names=columns
        )
        
        # Create the 'DateTime' column
        df['DateTime'] = pd.to_datetime(df['Year'] * 1000 + df['Day'], format='%Y%j') \
                         + pd.to_timedelta(df['Hour'], unit='h') \
                         + pd.to_timedelta(df['Minute'], unit='m')
        
        # Keep only the necessary columns
        df = df[['DateTime', 'AL_index', 'SYM_H']]
        
        df_list.append(df)
    except Exception as e:
        print(f"Error reading file {file}: {str(e)}")
        continue

if not df_list:
    raise ValueError("No valid data files were processed. Please ensure your data files exist and are properly formatted.")

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

# -----------------------------------
# F10.7 Solar Flux Index
# -----------------------------------
print("Processing F10.7 solar flux index data...")

# Update the file reading section
f107_list = []

for file in tqdm(f107_files, desc="Loading f107 data"):
    try:
        columns = ['Year', 'Day', 'Hour', 'f107_index']
        f107_df = pd.read_csv(file, sep=r'\s+', names=columns)
        f107_df['DateTime'] = pd.to_datetime(f107_df['Year'] * 1000 + f107_df['Day'], format='%Y%j') \
                              + pd.to_timedelta(f107_df['Hour'], unit='h')
        f107_df = f107_df[['DateTime', 'f107_index']]
        f107_list.append(f107_df)
    except Exception as e:
        print(f"Error reading file {file}: {str(e)}")
        continue

if not f107_list:
    raise ValueError("No valid F10.7 files were processed. Please ensure your data files exist and are properly formatted.")

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

# Add solar storm classification before splitting the dataset
print("Adding solar storm classification...")
filtered_df = add_solar_storm_classification(filtered_df, generate_viz=GENERATE_VISUALIZATIONS)

# Now proceed with splitting the dataset
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
train, test = train_test_split(filtered_df, test_size=100000)
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
