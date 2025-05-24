"""
Evaluate titheridge model on the test set using Equation 19 from the paper.

- Calculates T_0 and G_0 value using the titheridge v1 model with the day/night coefficients from the paper
- We use different coefficients based on whether it is night or day
- We use GLAT as a proxy for s_L

Paper: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/97JA03031
"""
import datasets
import numpy as np
import os
import pandas as pd

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

# Load dataset
ds = datasets.load_from_disk('/home/michael/auroral-precipitation-ml/dataset/output_dataset/test-storm')


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

# Filter out rows where it is neither day nor night. TODO: Check if we should do diurnal transition stuff.
original_size = len(ds)
print(f"Original size: {original_size}")
ds = ds.filter(lambda x: is_day_or_night(x['DateTimeFormatted']) in ['Day', 'Night'])
filtered_size = len(ds)
print(f"Filtered size: {filtered_size}")

# Calculate s_L, T_0, and G_0 for each row using max cores
ds = ds.map(lambda x: {
    's_L': np.sin(np.radians(x['GLAT']))**2,
    'T_0': X(np.sin(np.radians(x['GLAT']))**2, *coeffs[is_day_or_night(x['DateTimeFormatted']) + " T_0"]),
    'G_0': X(np.sin(np.radians(x['GLAT']))**2, *coeffs[is_day_or_night(x['DateTimeFormatted']) + " G_0"])
}, num_proc=os.cpu_count())

# Calculate Te1 using Equation 13
h0 = 400   # Reference height (km)
Re = 6371  # Earth radius (km)
R0 = 1 + h0/Re  # Radius at reference altitude (Earth radius + h0, km)
def calculate_Te(batch):
    # Extract the necessary values from the batch
    T0 = batch['T_0']
    G0 = batch['G_0']
    ILAT = batch['ILAT']
    h = batch['Altitude']

    ILAT_rad = np.radians(ILAT)
    # heq: we can derive it from L-shell which we can derive from ILAT.
    L = 1 / (np.cos(ILAT_rad)**2) # L-shell parameter / "normal field line parameter"
    heq = (L - 1) * Re # Height at the top of the field line (km).
    
    Rh = 1 + h/Re  # Radius at altitude h (km)
    Bh = B_h(h, h0, L, R0)
    Te = Te_h(h, T0, Bh, G0, h0, heq, R0, Rh)

    batch["z_titheridge_Te"] = Te
    return batch

ds = ds.map(calculate_Te, num_proc=os.cpu_count())

# Save the dataset to disk
ds.save_to_disk("/home/michael/auroral-precipitation-ml/dataset/output_dataset/test-storm-baseline-v1-ready")