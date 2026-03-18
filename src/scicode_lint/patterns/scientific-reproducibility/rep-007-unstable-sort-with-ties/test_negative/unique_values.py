import numpy as np


def sort_unique_timestamps(timestamps: np.ndarray) -> np.ndarray:
    unique_ts = np.unique(timestamps)
    return np.sort(unique_ts)


def rank_unique_float_measurements(measurements: np.ndarray) -> np.ndarray:
    order = np.argsort(measurements)
    return order
