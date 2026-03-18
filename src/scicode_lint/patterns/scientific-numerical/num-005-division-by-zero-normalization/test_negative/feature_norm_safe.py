import numpy as np


def z_score_with_where(data):
    """Z-score normalization using np.where to skip constant features."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.where(std > 0, (data - mean) / np.where(std > 0, std, 1.0), 0.0)


def l2_normalize_divide(vectors):
    """L2 normalize using np.divide with where parameter."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out = np.zeros_like(vectors)
    np.divide(vectors, norms, out=out, where=(norms != 0))
    return out


def minmax_with_clip(data):
    """Min-max scaling using np.clip on the denominator."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    range_val = np.clip(max_val - min_val, a_min=1e-10, a_max=None)
    return (data - min_val) / range_val


def unit_variance_scaling(X):
    """Scale to unit variance using np.where to guard zero std."""
    std = X.std(axis=0)
    mean = X.mean(axis=0)
    centered = X - mean
    result = np.where(std == 0, centered, centered / np.where(std > 0, std, 1.0))
    return result


data = np.random.randn(100, 10)
normalized = z_score_with_where(data)
