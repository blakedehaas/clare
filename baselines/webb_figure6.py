"""
Replicate figure 6 in the webb paper for v1 and v2
"""
import datasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.ticker import LogLocator, FixedFormatter


ds = datasets.load_from_disk("/home/michael/auroral-precipitation-ml/dataset/output_dataset/test-normal-baseline-ready")
# Function to calculate B(h) using equation (13b)
def B_h(h, h0, L, R0):
    log_term = np.log10(h / h0) # Note: Modification made changing natural log to log 10, as per Webb paper
    return (0.05 / (2 * L - R0)) * (88 + log_term * (10.5 - log_term))

# Calculate T_e(h) using equation (13a)
def Te_h(h, T0, Bh, G0, h0, heq, R0, Rh):
    term1 = (heq - h0) / R0**2
    term2 = (heq - h) / Rh**2
    inner_expression = 1 + Bh * (G0 / T0) * (term1 - term2)
    return T0 * (inner_expression)**(2/7)

coeffs = {
    "Day T_0": [1230, 2200, -3290, -0.26, -0.68],
    "Day G_0": [4.65, -8.55, 4.14, -2.16, 1.45],
    "Night T_0": [985, -963, 1125, -0.60, 0.10],
    "Night G_0": [0.756, -0.88, 0.29, -2.63, 1.84]
}

def X(s_L, a0, a1, a2, a3, a4):
    numerator = a0 + a1 * s_L + a2 * s_L**2
    denominator = 1 + a3 * s_L + a4 * s_L**2
    return numerator / denominator

# V1

def is_day_or_night(time_str):
    # Convert the time string to a datetime object
    time = pd.to_datetime(time_str)
    hour = time.hour
    
    if 9 <= hour < 16:
        return 'Day'
    elif (hour >= 21) or (hour < 4):
        return 'Night'
    else:
        return 'Transition'

# Filter dataset to include only Day and Night records based on DateTimeFormatted
ds = ds.filter(lambda x: is_day_or_night(x['DateTimeFormatted']) in ['Day', 'Night'], desc="Filtering dataset for Day and Night records")

# Filter out all ILAT that is outside of 15-55 degrees
ds = ds.filter(lambda x: 15 <= x['ILAT'] <= 55, desc="Filtering dataset for ILAT between 15 and 55 degrees")

# Map dataset to calculate v1_T_0 and v1_G_0 using coefficients specific to Day or Night
ds = ds.map(lambda x: {
    'v1_T_0': X(np.sin(np.radians(x['GLAT']))**2, *coeffs[is_day_or_night(x['DateTimeFormatted']) + " T_0"]),
    'v1_G_0': X(np.sin(np.radians(x['GLAT']))**2, *coeffs[is_day_or_night(x['DateTimeFormatted']) + " G_0"])
}, num_proc=os.cpu_count(), desc="Mapping dataset to calculate v1_T_0 and v1_G_0")

# Define ILAT bin centers (20, 30, 40, ..., 90)
ilat_centers = np.arange(20, 60, 10)  # [20, 30, 40, ..., 90]

# V1

# Initialize dictionaries to store results
mean_daytime_T_0 = {center: [] for center in ilat_centers}
mean_daytime_G_0 = {center: [] for center in ilat_centers}
mean_nighttime_T_0 = {center: [] for center in ilat_centers}
mean_nighttime_G_0 = {center: [] for center in ilat_centers}

# Function to assign ILAT to the nearest bin center
def get_nearest_bin(ilat):
    if 0 <= ilat < 90:
        return ilat_centers[np.argmin(np.abs(ilat_centers - ilat))]
    return None

# Process each record in the dataset and group by nearest ILAT bin center
for record in tqdm(ds):
    ilat = record["ILAT"]
    bin_center = get_nearest_bin(ilat)
    if bin_center is None:
        continue  # Skip ILAT values outside 20-90

    # Filter and assign to daytime or nighttime
    time_of_day = is_day_or_night(record['DateTimeFormatted'])
    if time_of_day == 'Day':
        mean_daytime_T_0[bin_center].append(record['v1_T_0'])
        mean_daytime_G_0[bin_center].append(record['v1_G_0'])
    elif time_of_day == 'Night':
        mean_nighttime_T_0[bin_center].append(record['v1_T_0'])
        mean_nighttime_G_0[bin_center].append(record['v1_G_0'])

# Calculate mean values for each ILAT bin center
for center in tqdm(ilat_centers):
    # Daytime means
    daytime_T_0_values = mean_daytime_T_0[center]
    daytime_G_0_values = mean_daytime_G_0[center]
    mean_daytime_T_0[center] = np.mean(daytime_T_0_values) if daytime_T_0_values else np.nan
    mean_daytime_G_0[center] = np.mean(daytime_G_0_values) if daytime_G_0_values else np.nan

    # Nighttime means
    nighttime_T_0_values = mean_nighttime_T_0[center]
    nighttime_G_0_values = mean_nighttime_G_0[center]
    mean_nighttime_T_0[center] = np.mean(nighttime_T_0_values) if nighttime_T_0_values else np.nan
    mean_nighttime_G_0[center] = np.mean(nighttime_G_0_values) if nighttime_G_0_values else np.nan

# Print results
print("Mean Daytime T_0 and G_0 by nearest ILAT bin center:")
for center in ilat_centers:
    print(f"ILAT {center}: T_0 = {mean_daytime_T_0[center]:.2f}, G_0 = {mean_daytime_G_0[center]:.2f}")

print("\nMean Nighttime T_0 and G_0 by nearest ILAT bin center:")
for center in ilat_centers:
    print(f"ILAT {center}: T_0 = {mean_nighttime_T_0[center]:.2f}, G_0 = {mean_nighttime_G_0[center]:.2f}")


# V2


# Initialize dictionaries to store v2 results
mean_daytime_T_0_v2 = {center: [] for center in ilat_centers}
mean_daytime_G_0_v2 = {center: [] for center in ilat_centers}
mean_nighttime_T_0_v2 = {center: [] for center in ilat_centers}
mean_nighttime_G_0_v2 = {center: [] for center in ilat_centers}

# Process each record in the original dataset
for record in tqdm(ds):
    ilat = record["ILAT"]
    bin_center = get_nearest_bin(ilat)
    if bin_center is None:
        continue  # Skip ILAT values outside 20-90

    # Triage based on is_day_or_night
    time_of_day = is_day_or_night(record['DateTimeFormatted'])
    if time_of_day == 'Day':
        mean_daytime_T_0_v2[bin_center].append(record['z_iri_reference_T0'])
        mean_daytime_G_0_v2[bin_center].append(record['z_best_G0'])
    elif time_of_day == 'Night':
        mean_nighttime_T_0_v2[bin_center].append(record['z_iri_reference_T0'])
        mean_nighttime_G_0_v2[bin_center].append(record['z_best_G0'])

# Calculate mean values for each ILAT bin center
for center in tqdm(ilat_centers):
    # Daytime means for v2
    daytime_T_0_v2_values = mean_daytime_T_0_v2[center]
    daytime_G_0_v2_values = mean_daytime_G_0_v2[center]
    mean_daytime_T_0_v2[center] = np.mean(daytime_T_0_v2_values) if daytime_T_0_v2_values else np.nan
    mean_daytime_G_0_v2[center] = np.mean(daytime_G_0_v2_values) if daytime_G_0_v2_values else np.nan

    # Nighttime means for v2
    nighttime_T_0_v2_values = mean_nighttime_T_0_v2[center]
    nighttime_G_0_v2_values = mean_nighttime_G_0_v2[center]
    mean_nighttime_T_0_v2[center] = np.mean(nighttime_T_0_v2_values) if nighttime_T_0_v2_values else np.nan
    mean_nighttime_G_0_v2[center] = np.mean(nighttime_G_0_v2_values) if nighttime_G_0_v2_values else np.nan

# Print results for v2
print("Mean Daytime T_0 (v2) and G_0 (v2) by nearest ILAT bin center:")
for center in ilat_centers:
    print(f"ILAT {center}: T_0 = {mean_daytime_T_0_v2[center]:.2f}, G_0 = {mean_daytime_G_0_v2[center]:.2f}")

print("\nMean Nighttime T_0 (v2) and G_0 (v2) by nearest ILAT bin center:")
for center in ilat_centers:
    print(f"ILAT {center}: T_0 = {mean_nighttime_T_0_v2[center]:.2f}, G_0 = {mean_nighttime_G_0_v2[center]:.2f}")


# Constants
h0 = 400   # Reference height (km)
Re = 6371  # Earth radius (km)
R0 = 1 + h0 / Re  # Radius at reference altitude (Earth radius + h0, km)
h_values = np.arange(0, 5125, 25)  # Altitude range (km)
ILAT_values = range(20, 51, 10)  # ILAT bin centers (20, 30, ..., 90)

# Function to calculate B(h) using equation (13b)
def B_h(h, h0, L, R0):
    log_term = np.log10(h / h0)  # Log base 10 as per Webb paper
    return (0.05 / (2 * L - R0)) * (88 + log_term * (10.5 - log_term))

# Calculate T_e(h) using equation (13a)
def Te_h(h, T0, Bh, G0, h0, heq, R0, Rh):
    term1 = (heq - h0) / R0**2
    term2 = (heq - h) / Rh**2
    inner_expression = 1 + Bh * (G0 / T0) * (term1 - term2)
    return T0 * (inner_expression)**(2/7)

# Plot for each ILAT bin center
for ILAT in ILAT_values:
    daytime_Te_values_v1 = []
    nighttime_Te_values_v1 = []
    daytime_Te_values_v2 = []
    nighttime_Te_values_v2 = []

    # Get the binned mean values for the current ILAT
    T0_day_v1 = mean_daytime_T_0[ILAT]
    G0_day_v1 = mean_daytime_G_0[ILAT]
    T0_night_v1 = mean_nighttime_T_0[ILAT]
    G0_night_v1 = mean_nighttime_G_0[ILAT]
    
    T0_day_v2 = mean_daytime_T_0_v2[ILAT]
    G0_day_v2 = mean_daytime_G_0_v2[ILAT]
    T0_night_v2 = mean_nighttime_T_0_v2[ILAT]
    G0_night_v2 = mean_nighttime_G_0_v2[ILAT]

    # Skip ILAT if any mean value is NaN
    if any(np.isnan([T0_day_v1, G0_day_v1, T0_night_v1, G0_night_v1, 
                     T0_day_v2, G0_day_v2, T0_night_v2, G0_night_v2])):
        print(f"Skipping ILAT {ILAT} due to NaN values")
        continue

    ILAT_rad = np.radians(ILAT)
    L = 1 / (np.cos(ILAT_rad)**2)  # L-shell parameter
    heq = (L - 1) * Re  # Height at the top of the field line (km)

    for h in h_values:
        Rh = 1 + h / Re  # Radius at altitude h (km)

        # v1 calculations
        Bh_day = B_h(h, h0, L, R0)
        Te_day_v1 = Te_h(h, T0_day_v1, Bh_day, G0_day_v1, h0, heq, R0, Rh)
        daytime_Te_values_v1.append(Te_day_v1)

        Bh_night = B_h(h, h0, L, R0)
        Te_night_v1 = Te_h(h, T0_night_v1, Bh_night, G0_night_v1, h0, heq, R0, Rh)
        nighttime_Te_values_v1.append(Te_night_v1)

        # v2 calculations
        Te_day_v2 = Te_h(h, T0_day_v2, Bh_day, G0_day_v2, h0, heq, R0, Rh)
        daytime_Te_values_v2.append(Te_day_v2)

        Te_night_v2 = Te_h(h, T0_night_v2, Bh_night, G0_night_v2, h0, heq, R0, Rh)
        nighttime_Te_values_v2.append(Te_night_v2)

    # Convert lists to numpy arrays
    daytime_Te_values_v1 = np.array(daytime_Te_values_v1)
    nighttime_Te_values_v1 = np.array(nighttime_Te_values_v1)
    daytime_Te_values_v2 = np.array(daytime_Te_values_v2)
    nighttime_Te_values_v2 = np.array(nighttime_Te_values_v2)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(daytime_Te_values_v1, h_values, label='Titheridge Daytime Te', color='blue')
    plt.plot(nighttime_Te_values_v1, h_values, label='Titheridge Nighttime Te', color='red')
    plt.plot(daytime_Te_values_v2, h_values, label='Titheridge-IRI Daytime Te', linestyle='--', color='blue')
    plt.plot(nighttime_Te_values_v2, h_values, label='Titheridge-IRI Nighttime Te', linestyle='--', color='red')

    plt.yscale('log')  # Set y-axis to log scale
    tick_locations = [500, 1000, 1500, 2000, 3000, 4000, 5000]
    tick_labels = ['500', '1000', '1500', '2000', '3000', '4000', '5000']
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    plt.gca().set_yticks(tick_locations)
    plt.gca().yaxis.set_major_formatter(FixedFormatter(tick_labels))

    plt.xlabel('Electron Temperature (Te) (°K)')
    plt.ylabel('Altitude (km)')
    plt.title(f'Altitude vs Electron Temperature (ILAT = {ILAT})')
    plt.xlim(0, 5000)
    plt.ylim(100, 5000)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'fig6_ILAT_{ILAT}.png')
    plt.close()
