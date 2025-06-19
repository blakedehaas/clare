"""
Simple example of how to run the simpler Titheridge Equation 19 polynominal model.
"""
import numpy as np
import matplotlib.pyplot as plt

def X(s_L, a0, a1, a2, a3, a4):
    numerator = a0 + a1 * s_L + a2 * s_L**2
    denominator = 1 + a3 * s_L + a4 * s_L**2
    return numerator / denominator

# "a" values are provided in the paper
# s_L is sin^2(Lat300) where Lat300 is the latitude of the field line in the ionosphere (at h=300km).
# s_L is our free variable, for AKEBONO we can use GLAT as a proxy although GLAT is typically measured at ground level

# Coefficients from Table 2
coeffs = {
    "Day T_0": [1230, 2200, -3290, -0.26, -0.68],
    "Day G_0": [4.65, -8.55, 4.14, -2.16, 1.45],
    "Night T_0": [985, -963, 1125, -0.60, 0.10],
    "Night G_0": [0.756, -0.88, 0.29, -2.63, 1.84]
}

# Generate s_L values from 0 to 1 in 0.05 increments
s_L = np.arange(0, 1.05, 0.05)

# Compute T_0 and G_0 for Day and Night
results = {}
for key, (a0, a1, a2, a3, a4) in coeffs.items():
    results[key] = [X(s, a0, a1, a2, a3, a4) for s in s_L]

# Plotting
plt.figure(figsize=(8, 10))

# (a) Plot for T_0 (at 400 km)
plt.subplot(2, 1, 1)
plt.plot(s_L, results["Day T_0"], 'o-', label="Day", markersize=5)
plt.plot(s_L, results["Night T_0"], 'x-', label="Night", markersize=5)
plt.xlabel(r"$s_L = \sin^2(Lat_{300})$")
plt.ylabel(r"Start. Temperature, $T_0$")
plt.title("(a) Fitted values of $T_0$ (at 400 km)")
plt.legend()
plt.grid(True)

# (b) Plot for G_0
plt.subplot(2, 1, 2)
plt.plot(s_L, results["Day G_0"], 'o-', label="Day", markersize=5)
plt.plot(s_L, results["Night G_0"], 'x-', label="Night", markersize=5)
plt.xlabel(r"$s_L = \sin^2(Lat_{300})$")
plt.ylabel(r"Start. Gradient, $G_0$")
plt.title("(b) Fitted values of $G_0$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('T0_G0_plots.png')