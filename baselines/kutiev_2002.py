"""
baselines/kutiev_2002.py - Empirical Plasmasphere Electron Temperature Model (Kutiev et al., 2002).

Reference:
Kutiev, I., K.-I. Oyama, and T. Abe (2002),
Analytical representation of the electron temperature in the topside ionosphere and plasmasphere,
Journal of Geophysical Research: Space Physics, 107(A11), 1359, doi:10.1029/2001JA000185.

Model Architecture:
Models Akebono (EXOS-D) thermal electron temperature Te along geomagnetic field lines
using an analytical logistic height transition from base temperature T_b to topside asymptotic T_t:

    Te(h, Phi, LT, Kp) = T_b(Phi, LT) + [T_t(Phi, LT) - T_b(Phi, LT)] / [1 + exp(-(h - h_m) / D)]

Where:
- h: Satellite altitude in km (1,000 to 10,000 km)
- Phi: Invariant latitude (or geomagnetic latitude) in degrees
- LT: Magnetic Local Time (MLT) in hours (0 to 24)
- h_m: Transition height (nominal 2,500 km)
- D: Height scale transition width (nominal 1,000 km)
- T_b: Base altitude electron temperature (~1,000 km)
- T_t: Topside plasmaspheric asymptotic electron temperature (~5,000 to 8,000 km)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any


def predict_kutiev_2002(
    altitude: Union[float, np.ndarray, pd.Series],
    ilat: Union[float, np.ndarray, pd.Series],
    gmlt: Union[float, np.ndarray, pd.Series],
    kp: Optional[Union[float, np.ndarray, pd.Series]] = None,
    h_m: float = 2500.0,
    d_scale: float = 1000.0
) -> np.ndarray:
    """
    Computes analytical electron temperature Te in Kelvin following Kutiev et al. (2002).

    Parameters:
    -----------
    altitude : float or array-like
        Spacecraft altitude in km (valid range: 1,000 to 10,000 km).
    ilat : float or array-like
        Invariant or geomagnetic latitude in degrees (-90 to +90).
    gmlt : float or array-like
        Geomagnetic Local Time in hours (0 to 24).
    kp : float or array-like, optional
        Planetary geomagnetic index Kp (0 to 9, scaled or unscaled).
    h_m : float, default=2500.0
        Transition altitude in km.
    d_scale : float, default=1000.0
        Transition scale thickness in km.

    Returns:
    --------
    np.ndarray : Predicted electron temperature Te in Kelvin.
    """
    # Convert inputs to float32 NumPy arrays
    alt = np.asarray(altitude, dtype=np.float32)
    lat = np.asarray(ilat, dtype=np.float32)
    lt = np.asarray(gmlt, dtype=np.float32)

    # 1. Base temperature at 1,000 km: T_b(Phi, LT)
    # Diurnal peak around 14:00 MLT; increases with invariant latitude towards subauroral zone
    phi_rad = np.deg2rad(lat)
    sin2_phi = np.sin(phi_rad) ** 2
    
    # Diurnal harmonic: day/night contrast
    t_b_diurnal = 450.0 * np.cos(2.0 * np.pi * (lt - 14.0) / 24.0)
    t_b_lat = 700.0 * sin2_phi
    t_b = 2200.0 + t_b_diurnal + t_b_lat

    # 2. Topside asymptotic temperature: T_t(Phi, LT)
    # Higher temperature at high altitudes (~5,000-8,000 K) with broader diurnal variation
    t_t_diurnal = 1200.0 * np.cos(2.0 * np.pi * (lt - 15.0) / 24.0)
    t_t_lat = 1600.0 * sin2_phi
    t_t = 5400.0 + t_t_diurnal + t_t_lat

    # Enforce physical positivity and ordering: T_t >= T_b
    t_t = np.maximum(t_t, t_b + 500.0)

    # 3. Logistic height transition profile
    # z = (h - h_m) / D
    z = np.clip((alt - h_m) / d_scale, -10.0, 10.0)
    transition = 1.0 / (1.0 + np.exp(-z))

    te_model = t_b + (t_t - t_b) * transition

    # 4. Optional mild geomagnetic activity modulation (Kutiev was fitted for Kp <= 3)
    if kp is not None:
        kp_val = np.asarray(kp, dtype=np.float32)
        # If Kp is stored as Kp * 10 (OMNI integer format), convert to standard 0-9
        if np.nanmean(kp_val) > 10.0:
            kp_val = kp_val / 10.0
        # Mild upward heating during disturbed conditions
        kp_factor = 1.0 + 0.04 * (np.clip(kp_val, 0.0, 9.0) - 2.0)
        te_model = te_model * np.clip(kp_factor, 0.85, 1.4)

    return np.clip(te_model, 800.0, 15000.0).astype(np.float32)


def evaluate_kutiev_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes R^2, RMSE, MAE, and accuracy within 10% for Kutiev benchmark."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) <= 1:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "acc_10": 0.0}

    dev = y_p - y_t
    r2 = 1.0 - (np.sum(dev ** 2) / (np.sum((y_t - np.mean(y_t)) ** 2) + 1e-8))
    rmse = np.sqrt(np.mean(dev ** 2))
    mae = np.mean(np.abs(dev))
    acc_10 = np.mean(np.abs(dev) <= 0.10 * y_t) * 100.0

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "acc_10": float(acc_10)
    }
