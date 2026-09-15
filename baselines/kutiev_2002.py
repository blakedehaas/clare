"""Average plasmaspheric electron-temperature model from Kutiev et al. (2002).

This follows equations (1) and (2), with the average coefficients reported in
Tables 2 and 3 of doi:10.1029/2002JA009494. Predictions outside the paper's
published domain are returned as NaN rather than silently extrapolated.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ZoneCoefficients:
    """Coefficients for one local-time, hemisphere, and L-shell zone."""

    intercept_k: float
    altitude_gradient_k_per_km: float
    latitude_or_l_scale: float


@dataclass(frozen=True)
class KutievPrediction:
    """Predictions plus the applicability mask and selected zone names."""

    temperature_k: np.ndarray
    eligible: np.ndarray
    zone: np.ndarray


# D/N = day/night, N/S = north/south, E/M = equatorial/midlatitude.
ZONE_COEFFICIENTS: Mapping[str, ZoneCoefficients] = {
    "DNE": ZoneCoefficients(2778.0, 0.63, 2.2),
    "DSE": ZoneCoefficients(3114.0, 0.50, 0.4),
    "NNE": ZoneCoefficients(1536.0, 0.17, 1.3),
    "NSE": ZoneCoefficients(1676.0, 0.14, 0.7),
    "DNM": ZoneCoefficients(5326.0, 0.24, 18.9),
    "DSM": ZoneCoefficients(4558.0, 0.37, 20.6),
    "NNM": ZoneCoefficients(2161.0, 0.10, 19.0),
    "NSM": ZoneCoefficients(2077.0, 0.11, 19.2),
}


def l_shell_from_invariant_latitude(invariant_latitude_deg):
    """Return dipole L shell using L = sec^2(invariant latitude)."""

    latitude_rad = np.deg2rad(np.asarray(invariant_latitude_deg, dtype=float))
    return 1.0 / np.cos(latitude_rad) ** 2


def _local_time_sector(magnetic_local_time_hours: np.ndarray) -> np.ndarray:
    """Map MLT to the two sectors modeled by Kutiev et al."""

    mlt = np.mod(magnetic_local_time_hours, 24.0)
    sector = np.full(mlt.shape, "", dtype="<U1")
    sector[(mlt >= 9.0) & (mlt < 16.0)] = "D"
    sector[(mlt >= 22.0) | (mlt < 4.0)] = "N"
    return sector


def predict_kutiev_2002(
    altitude_km,
    geomagnetic_latitude_deg,
    invariant_latitude_deg,
    magnetic_local_time_hours,
) -> KutievPrediction:
    """Evaluate the published average Kutiev model on broadcastable inputs.

    Equatorial zones use ``Te = (c + d*altitude) + b_e*glat^2``.
    Midlatitude zones use ``Te = (c + d*altitude) + b_a*(L^5 - 32)``.
    """

    altitude, glat, ilat, mlt = np.broadcast_arrays(
        np.asarray(altitude_km, dtype=float),
        np.asarray(geomagnetic_latitude_deg, dtype=float),
        np.asarray(invariant_latitude_deg, dtype=float),
        np.asarray(magnetic_local_time_hours, dtype=float),
    )
    l_shell = l_shell_from_invariant_latitude(ilat)
    sector = _local_time_sector(mlt)

    finite = (
        np.isfinite(altitude)
        & np.isfinite(glat)
        & np.isfinite(ilat)
        & np.isfinite(mlt)
        & np.isfinite(l_shell)
    )
    boundary_tolerance = 1e-10
    eligible = (
        finite
        & (altitude >= 1000.0)
        & (altitude <= 10000.0)
        & (np.abs(glat) <= 70.0)
        & (l_shell >= 1.0 - boundary_tolerance)
        & (l_shell <= 3.0 + boundary_tolerance)
        & (sector != "")
    )

    hemisphere = np.where(glat >= 0.0, "N", "S")
    radial_zone = np.where(l_shell < 2.0 - boundary_tolerance, "E", "M")
    zone = np.char.add(np.char.add(sector, hemisphere), radial_zone)
    zone = np.where(eligible, zone, "")

    temperature = np.full(altitude.shape, np.nan, dtype=float)
    for zone_name, coeffs in ZONE_COEFFICIENTS.items():
        selected = eligible & (zone == zone_name)
        if not np.any(selected):
            continue
        base_temperature = (
            coeffs.intercept_k
            + coeffs.altitude_gradient_k_per_km * altitude[selected]
        )
        if zone_name.endswith("E"):
            spatial_term = coeffs.latitude_or_l_scale * glat[selected] ** 2
        else:
            spatial_term = coeffs.latitude_or_l_scale * (
                l_shell[selected] ** 5 - 32.0
            )
        temperature[selected] = base_temperature + spatial_term

    return KutievPrediction(temperature, eligible, zone)
