"""
Replicate figure 6 in the webb paper for v1 and v2
"""
import datasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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

# Map dataset to calculate v1_T_0 and v1_G_0 using coefficients specific to Day or Night
ds = ds.map(lambda x: {
    'v1_T_0': X(np.sin(np.radians(x['GLAT']))**2, *coeffs[is_day_or_night(x['DateTimeFormatted']) + " T_0"]),
    'v1_G_0': X(np.sin(np.radians(x['GLAT']))**2, *coeffs[is_day_or_night(x['DateTimeFormatted']) + " G_0"])
}, num_proc=os.cpu_count(), desc="Mapping dataset to calculate v1_T_0 and v1_G_0")

# Calculate mean daytime and nighttime values for v1_T_0 and v1_G_0
daytime_values = ds.filter(lambda x: is_day_or_night(x['DateTimeFormatted']) == 'Day', desc="Filtering daytime records").map(lambda x: {'v1_T_0': x['v1_T_0'], 'v1_G_0': x['v1_G_0']})
nighttime_values = ds.filter(lambda x: is_day_or_night(x['DateTimeFormatted']) == 'Night', desc="Filtering nighttime records").map(lambda x: {'v1_T_0': x['v1_T_0'], 'v1_G_0': x['v1_G_0']})

mean_daytime_T_0_v1 = np.mean(daytime_values['v1_T_0'])
mean_daytime_G_0_v1 = np.mean(daytime_values['v1_G_0'])

mean_nighttime_T_0_v1 = np.mean(nighttime_values['v1_T_0'])
mean_nighttime_G_0_v1 = np.mean(nighttime_values['v1_G_0'])

# Filter dataset for daytime values based on GMLT (9am to 4pm) and map relevant columns
daytime_values_gmlt = ds.filter(lambda x: 9 <= x['GMLT'] < 16, desc="Filtering daytime records by GMLT").map(lambda x: {'gMLT': x['GMLT'], 'z_best_G0': x['z_best_G0'], 'z_iri_reference_T0': x['z_iri_reference_T0'], 'v1_T_0': x['v1_T_0'], 'v1_G_0': x['v1_G_0']})
# Filter dataset for nighttime values based on GMLT (9pm to 4am) and map relevant columns
nighttime_values_gmlt = ds.filter(lambda x: (21 <= x['GMLT']) or (0 <= x['GMLT'] < 4), desc="Filtering nighttime records by GMLT").map(lambda x: {'gMLT': x['GMLT'], 'z_best_G0': x['z_best_G0'], 'z_iri_reference_T0': x['z_iri_reference_T0'], 'v1_T_0': x['v1_T_0'], 'v1_G_0': x['v1_G_0']})

mean_daytime_G_0_v2 = np.mean(daytime_values_gmlt['z_best_G0'])

mean_daytime_T_0_v2 = np.mean(daytime_values_gmlt['z_iri_reference_T0'])

mean_nighttime_G_0_v2 = np.mean(nighttime_values_gmlt['z_best_G0'])
mean_nighttime_T_0_v2 = np.mean(nighttime_values_gmlt['z_iri_reference_T0'])

print(f"Daytime G0 v2 - Mean: {mean_daytime_G_0_v2}")
print(f"Daytime T0 v2 - Mean: {mean_daytime_T_0_v2}")
print(f"Nighttime G0 v2 - Mean: {mean_nighttime_G_0_v2}")
print(f"Nighttime T0 v2 - Mean: {mean_nighttime_T_0_v2}")

print(f"Daytime v1 G0 (GMLT) - Mean: {mean_daytime_G_0_v1}")
print(f"Nighttime v1 G0 (GMLT) - Mean: {mean_nighttime_G_0_v1}")
print(f"Daytime v1 T0 (GMLT) - Mean: {mean_daytime_T_0_v1}")
print(f"Nighttime v1 T0 (GMLT) - Mean: {mean_nighttime_T_0_v1}")


# Constants
h0 = 400   # Reference height (km)
Re = 6371  # Earth radius (km)
R0 = 1 + h0/Re  # Radius at reference altitude (Earth radius + h0, km)

h_values = np.arange(0, 5125, 25)

daytime_Te_values_v1 = []
nighttime_Te_values_v1 = []

daytime_Te_value_s_v2 = []
nighttime_Te_value_s_v2 = []

ILAT_values = range(20, 91, 10)
for ILAT in ILAT_values:
    daytime_Te_values_v1 = []
    nighttime_Te_values_v1 = []
    
    daytime_Te_value_s_v2 = []
    nighttime_Te_value_s_v2 = []
    
    for h in tqdm(h_values):
        print("h", h)
        ILAT_rad = np.radians(ILAT)
        L = 1 / (np.cos(ILAT_rad)**2) # L-shell parameter / "normal field line parameter"
        heq = (L - 1) * Re # Height at the top of the field line (km).
        Rh = 1 + h/Re  # Radius at altitude h (km)
        
        Bh_day = B_h(h, h0, L, R0)
        Te_day_v1 = Te_h(h, mean_daytime_T_0_v1, Bh_day, mean_daytime_G_0_v1, h0, heq, R0, Rh)
        daytime_Te_values_v1.append(Te_day_v1)
        
        Bh_night = B_h(h, h0, L, R0)
        Te_night_v1 = Te_h(h, mean_nighttime_T_0_v1, Bh_night, mean_nighttime_G_0_v1, h0, heq, R0, Rh)
        nighttime_Te_values_v1.append(Te_night_v1)

        # For v2 (assuming similar calculation as v1 for demonstration)
        Te_da_y_v2 = Te_h(h, mean_daytime_T_0_v2, Bh_day, mean_daytime_G_0_v2, h0, heq, R0, Rh)
        daytime_Te_value_s_v2.append(Te_da_y_v2)
        
        Te_nigh_t_v2 = Te_h(h, mean_nighttime_T_0_v2, Bh_night, mean_nighttime_G_0_v2, h0, heq, R0, Rh)
        nighttime_Te_value_s_v2.append(Te_nigh_t_v2)

    daytime_Te_values_v1 = np.array(daytime_Te_values_v1)
    nighttime_Te_values_v1 = np.array(nighttime_Te_values_v1)

    daytime_Te_value_s_v2 = np.array(daytime_Te_value_s_v2)
    nighttime_Te_value_s_v2 = np.array(nighttime_Te_value_s_v2)

    plt.figure(figsize=(10, 6))
    plt.plot(daytime_Te_values_v1, h_values, label='V1 Daytime Te')
    plt.plot(nighttime_Te_values_v1, h_values, label='V1 Nighttime Te')
    
    plt.plot(daytime_Te_value_s_v2, h_values, label='V2 Daytime Te', linestyle='--')
    plt.plot(nighttime_Te_value_s_v2, h_values, label='V2 Nighttime Te', linestyle='--')
    
    plt.xlabel('Electron Temperature (Te) (°K)')
    plt.ylabel('Altitude (km)')
    plt.title(f'Altitude vs Electron Temperature (ILAT = {ILAT})')
    plt.xlim(0, 5000)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'fig6_ILAT_{ILAT}.png')
    plt.show()





