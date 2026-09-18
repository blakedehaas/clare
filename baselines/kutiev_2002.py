"""Average Akebono electron-temperature model from Kutiev et al. (2002).

Reference: https://doi.org/10.1029/2002JA009494, equations (1)-(2) and
Tables 2-3. Their altitude regressions are defined from 1000-6370 km, for
L <= 3, daytime (09-16 MLT), and nighttime (22-04 MLT). Unsupported samples
return NaN rather than an invented interpolation or extrapolation.
"""

from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

EARTH_RADIUS_KM = 6371.0
MAX_REGRESSION_ALTITUDE_KM = 6370.0

# zone: c [K], d [K/km], latitude/L-shell scale factor
EQUATORIAL = {
    "DNE": (2778.0, 0.63, 2.2),
    "DSE": (3114.0, 0.50, 0.4),
    "NNE": (1536.0, 0.17, 1.3),
    "NSE": (1676.0, 0.14, 0.7),
}
MIDLATITUDE = {
    "DNM": (5326.0, 0.24, 18.9),
    "DSM": (4558.0, 0.37, 20.6),
    "NNM": (2161.0, 0.10, 19.0),
    "NSM": (2077.0, 0.11, 19.2),
}


def predict_kutiev_2002(altitude: ArrayLike, glat: ArrayLike, gmlt: ArrayLike) -> np.ndarray:
    """Return the paper's average ``Te`` in kelvin; unsupported inputs are NaN."""
    alt, lat, lt = np.broadcast_arrays(
        np.asarray(altitude, dtype=float),
        np.asarray(glat, dtype=float),
        np.asarray(gmlt, dtype=float),
    )
    result = np.full(alt.shape, np.nan, dtype=float)
    l_shell = (EARTH_RADIUS_KM + alt) / (EARTH_RADIUS_KM * np.cos(np.deg2rad(lat)) ** 2)
    day = (lt >= 9.0) & (lt <= 16.0)
    night = (lt >= 22.0) | (lt <= 4.0)
    valid = (
        np.isfinite(alt + lat + lt)
        & (alt >= 1000.0)
        & (alt <= MAX_REGRESSION_ALTITUDE_KM)
        & (lt >= 0.0)
        & (lt < 24.0)
        & (l_shell <= 3.0)
    )

    for is_day, prefix in ((day, "D"), (night, "N")):
        for north, hemisphere in ((lat >= 0.0, "N"), (lat < 0.0, "S")):
            mask = valid & is_day & north & (l_shell < 2.0)
            c, d, b = EQUATORIAL[prefix + hemisphere + "E"]
            result[mask] = c + d * alt[mask] + b * lat[mask] ** 2

            mask = valid & is_day & north & (l_shell >= 2.0)
            c, d, b = MIDLATITUDE[prefix + hemisphere + "M"]
            result[mask] = c + d * alt[mask] + b * (l_shell[mask] ** 5 - 32.0)

    return result.astype(np.float32)


def evaluate_kutiev_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics over finite observations covered by the paper's model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) <= 1:
        return {"n": int(len(y_t)), "r2": 0.0, "rmse": 0.0, "mae": 0.0, "acc_10": 0.0}

    dev = y_p - y_t
    denominator = np.sum((y_t - np.mean(y_t)) ** 2)
    return {
        "n": int(len(y_t)),
        "r2": float(1.0 - np.sum(dev ** 2) / denominator) if denominator else 0.0,
        "rmse": float(np.sqrt(np.mean(dev ** 2))),
        "mae": float(np.mean(np.abs(dev))),
        "acc_10": float(np.mean(np.abs(dev) <= 0.10 * np.abs(y_t)) * 100.0),
    }
