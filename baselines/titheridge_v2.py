"""
Evaluate titheridge model on the test set using Equation 13 from the paper.

- Gets reference temperature and predictions for altitude range (400km to 2000km) using IRI2020 model
- Finds optimal G0 value
- Calculates Te using Titheridge value using T0 from IRI and G0 from least squares fit of IRI.

Paper: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/97JA03031
"""
import numpy as np
import iricore
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import datasets
import os

iricore.update() # Updates indicies

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

# For each timestep, get the G0 value that best matches against the IRI model then use that value to calculate the Titheridge prediction + IRI prediction.
ds = datasets.load_from_disk('/home/michael/auroral-precipitation-ml/dataset/output_dataset/test-normal')

# Constants
h0 = 400   # Reference height (km)
Re = 6371  # Earth radius (km)
R0 = 1 + h0/Re  # Radius at reference altitude (Earth radius + h0, km)

def calculate_IRI2020_predictions(batch):
    # Extract datetime and coordinates from the batch
    dt = batch['DateTimeFormatted']
    lat, lon = batch['XXLAT'], batch['XXLON']

    # Convert pandas timestamp to datetime UTC
    dt = dt.to_pydatetime()

    # Run IRI model for this location and time
    altrange = [400, 2000, 100]  # Altitude range in km
    out = iricore.iri(dt, altrange, lat, lon, version=20)
    
    # Store the electron temperature from IRI
    batch['IRI2020_range'] = out.etemp.tolist()
    
    return batch

# Map the function across the dataset
ds = ds.map(calculate_IRI2020_predictions, num_proc=os.cpu_count())

# For what value of G0 does the Titheridge model best fit the IRI model?
def calculate_g0(batch):
    T0 = batch["IRI2020_range"][0] # First value is 400km reference value
    ILAT = batch["ILAT"]
    ILAT_rad = np.radians(ILAT)
    # heq: we can derive it from L-shell which we can derive from ILAT.
    L = 1 / (np.cos(ILAT_rad)**2) # L-shell parameter / "normal field line parameter"
    heq = (L - 1) * Re # Height at the top of the field line (km).

    # Search through the G0 space
    G0_search_space = np.linspace(-20.0, 50.0, 701)
    altrange = np.arange(400, 2100, 100)  # Altitude range in km
    current_lowest_squared_error = float('inf')
    best_G0 = None
    for G0 in G0_search_space:
        Te_list = []
        for alt in altrange:
            h = alt
            Rh = 1 + h/Re  # Radius at altitude h (km)
            
            Bh = B_h(h, h0, L, R0)
            Te = Te_h(h, T0, Bh, G0, h0, heq, R0, Rh)
            Te_list.append(Te)

        # Compute total square error against IRI2020_range
        total_squared_error = 0
        for iri_val, titheridge_val in zip(batch["IRI2020_range"], Te_list):
            squared_error = (iri_val - titheridge_val) ** 2
            total_squared_error += squared_error

        if total_squared_error < current_lowest_squared_error:
            current_lowest_squared_error = squared_error
            best_G0 = G0

    # Round to closest 0.1
    best_G0 = round(best_G0, 1)

    # Update the batch with the best G0 value
    batch['z_best_G0'] = best_G0

    # Update T0 value with IRI model prediction
    batch['z_iri_reference_T0'] = T0

    # Calculate the actual Titheridge model prediction using the best G0 value
    h = batch["Altitude"]
    Rh = 1 + h/Re  # Radius at altitude h (km)
    G0 = best_G0
    Bh = B_h(h, h0, L, R0)
    Te = Te_h(h, T0, Bh, G0, h0, heq, R0, Rh)

    batch["z_titheridge_Te"] = Te
    return batch

ds = ds.map(calculate_g0, num_proc=os.cpu_count())

# Remove the IRI2020_range column since we no longer need it
ds = ds.remove_columns(['IRI2020_range'])

# Save the dataset to disk
ds.save_to_disk("/home/michael/auroral-precipitation-ml/dataset/output_dataset/test-normal-baseline-ready")