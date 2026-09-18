"""Deterministic feature transformations shared by CLARE training and evaluation."""

import numpy as np

ALTITUDE_MIN_KM = 1000.0
ALTITUDE_MAX_KM = 8000.0
LATITUDE_SCALE_DEG = 90.0
KP_MIN = 0.0
KP_MAX = 90.0
MLT_PERIOD_HOURS = 24.0


def _float32(values):
    return np.asarray(values, dtype=np.float32)


def _scale_to_minus_one_one(values, minimum, maximum):
    values = _float32(values)
    return (2.0 * (values - minimum) / (maximum - minimum) - 1.0).astype(np.float32)


def normalize_altitude(values):
    return _scale_to_minus_one_one(values, ALTITUDE_MIN_KM, ALTITUDE_MAX_KM)


def normalize_latitude(values):
    return (_float32(values) / LATITUDE_SCALE_DEG).astype(np.float32)


def normalize_kp(values):
    return _scale_to_minus_one_one(values, KP_MIN, KP_MAX)


NORMALIZATIONS = {
    "Altitude": normalize_altitude,
    "GCLAT": normalize_latitude,
    "ILAT": normalize_latitude,
    "GLAT": normalize_latitude,
    "XXLAT": normalize_latitude,
    "Kp_index": normalize_kp,
}


def encode_longitude(values, prefix):
    radians = np.deg2rad(_float32(values))
    return {
        f"{prefix}_sin": np.sin(radians).astype(np.float32),
        f"{prefix}_cos": np.cos(radians).astype(np.float32),
    }


def encode_gclon(values):
    return encode_longitude(values, "GCLON")


def encode_xxlon(values):
    return encode_longitude(values, "XXLON")


def encode_gmlt(values):
    radians = _float32(values) * (2.0 * np.pi / MLT_PERIOD_HOURS)
    return {
        "GMLT_sin": np.sin(radians).astype(np.float32),
        "GMLT_cos": np.cos(radians).astype(np.float32),
    }


CIRCULAR_ENCODINGS = {
    "GCLON": encode_gclon,
    "XXLON": encode_xxlon,
    "GMLT": encode_gmlt,
}
