import numpy as np


def decibel_conversion(power_values):
    """Convert power to decibels - input guaranteed positive by physics."""
    return 10 * np.log10(power_values)


def logarithmic_scale_plot(data):
    """Plot data on log scale after filtering positives."""
    positive_mask = data > 0
    positive_data = data[positive_mask]
    return np.log(positive_data)


def ph_calculation(hydrogen_concentration):
    """Calculate pH from H+ concentration.

    Concentration is always positive in valid chemistry.
    Using np.log10 is safe here.
    """
    return -np.log10(hydrogen_concentration)


def information_gain(counts):
    """Information gain using scipy's approach with where()."""
    total = np.sum(counts)
    probs = counts / total
    safe_probs = np.maximum(probs, 1e-15)
    log_probs = np.log2(safe_probs)
    log_probs[probs == 0] = 0
    return -np.sum(probs * log_probs)
