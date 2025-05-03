"""
Evaluate titheridge model on the test set using Equation 13 from the paper.

Paper: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/97JA03031
"""
import numpy as np
import matplotlib.pyplot as plt

# Run IRI





# Define parameters (adjust these based on your specific conditions)
T0 = 1500  # Reference temperature at reference height 400 km. TODO: obtain from IRI model
G0 = 1e6   # Height gradient at reference height 400km. TODO: What gradient value would fit the curve implied by IRI best.
ILAT = 60 # Use ILAT value to derive L-shell. 


h0 = 400   # Reference height (km)
Re = 6371  # Earth radius (km)
R0 = 1 + h0/Re  # Radius at reference altitude (Earth radius + h0, km)
h = np.linspace(h0, 2000, 100)  # Altitude range (km)
Rh = 1 + h/Re  # Radius at altitude h (km)

# heq: we can derive it from L-shell which we can derive from ILAT.
ILAT_rad = np.radians(ILAT)
L = 1 / (np.cos(ILAT_rad)**2) # L-shell parameter / "normal field line parameter"
heq = (L - 1) * Re # Height at the top of the field line (km).

# Function to calculate B(h) using equation (13b)
def B_h(h, h0, L, R0):
    log_term = np.log10(h / h0) # Note: Modification made changing natural log to log 10, as per Webb paper
    return (0.05 / (2 * L - R0)) * (88 + log_term * (10.5 - log_term))

# Calculate B(h) for all altitudes
Bh = B_h(h, h0, L, R0)

# Calculate T_e(h) using equation (13a)
def Te_h(h, T0, Bh, G0, h0, heq, R0, Rh):
    term1 = (heq - h0) / R0**2
    term2 = (heq - h) / Rh**2
    inner_expression = 1 + Bh * (G0 / T0) * (term1 - term2)
    return T0 * (inner_expression)**(2/7)

# Compute T_e for all altitudes
Te = Te_h(h, T0, Bh, G0, h0, heq, R0, Rh)

# Plot the electron temperature profile
plt.figure(figsize=(8, 6))
plt.plot(Te, h, 'b-', label='Electron Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (km)')
plt.title('Titheridge Electron Temperature Profile')
plt.grid(True)
plt.legend()
plt.savefig('titheridge_te_profile.png')