import numpy as np

NORMALIZATIONS = {
    'Altitude': lambda x: (np.array(x, dtype=np.float32) - 4000) / 4000,  # Scale to [-1, 1]
    'GCLON': lambda x: np.sin(np.deg2rad(np.array(x, dtype=np.float32))),  # Convert longitude to sine
    'GCLAT': lambda x: np.array(x, dtype=np.float32) / 90,  # Scale latitude to [-1, 1]
    'ILAT': lambda x: np.array(x, dtype=np.float32) / 90,  # Scale latitude to [-1, 1]
    'GLAT': lambda x: np.array(x, dtype=np.float32) / 90,  # Scale latitude to [-1, 1]
    'GMLT': lambda x: np.sin(np.array(x, dtype=np.float32) * np.pi / 12),  # Convert MLT (0-24) to sine
    'XXLAT': lambda x: np.array(x, dtype=np.float32) / 90,  # Scale latitude to [-1, 1]
    'XXLON': lambda x: np.sin(np.deg2rad(np.array(x, dtype=np.float32))),  # Convert longitude to sine
    # "Kp_index": lambda x: np.array(x, dtype=np.float32) / 45 - 1,  # Scale Kp index (0-90) to [-1, 1]
}