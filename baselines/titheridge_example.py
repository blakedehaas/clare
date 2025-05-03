"""
Simple example of how to run the Titheridge model.

Used to verify correctionness against the paper.
"""
import numpy as np

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


# Free param
T0 = 2000
G0 = 4
ILAT = 25
h = 1500

# Constants
h0 = 400   # Reference height (km)
Re = 6371  # Earth radius (km)
R0 = 1 + h0/Re  # Radius at reference altitude (Earth radius + h0, km)


ILAT_rad = np.radians(ILAT)
# heq: we can derive it from L-shell which we can derive from ILAT.
L = 1 / (np.cos(ILAT_rad)**2) # L-shell parameter / "normal field line parameter"
heq = (L - 1) * Re # Height at the top of the field line (km).
Rh = 1 + h/Re  # Radius at altitude h (km)
Bh = B_h(h, h0, L, R0)
Te = Te_h(h, T0, Bh, G0, h0, heq, R0, Rh)

print("Te", Te)