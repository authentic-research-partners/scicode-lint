import numpy as np


def normalize_spectrum(spectrum):
    """Normalize a frequency spectrum to unit energy."""
    result = spectrum.copy()
    total_energy = np.sum(result**2)
    result /= np.sqrt(total_energy + 1e-12)
    return result


def center_coordinates(positions):
    """Shift particle positions so the centroid is at the origin."""
    centered = positions.copy()
    centroid = centered.mean(axis=0)
    centered -= centroid
    return centered


def apply_baseline_correction(signal, baseline):
    """Subtract a baseline from a measured signal and clip negatives."""
    corrected = signal.copy()
    corrected -= baseline
    corrected[corrected < 0] = 0.0
    return corrected


def scale_features_to_unit_range(features):
    """Rescale each feature column to [0, 1] range."""
    scaled = features.copy()
    col_min = scaled.min(axis=0)
    col_range = scaled.max(axis=0) - col_min + 1e-10
    scaled -= col_min
    scaled /= col_range
    return scaled


rng = np.random.default_rng(42)
spec = rng.random(256)
normed = normalize_spectrum(spec)

coords = rng.random((50, 3))
shifted = center_coordinates(coords)

raw_signal = rng.random(100)
bg = rng.random(100) * 0.3
clean = apply_baseline_correction(raw_signal, bg)
