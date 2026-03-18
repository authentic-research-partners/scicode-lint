import numpy as np


def z_score_with_std_check(data):
    """Z-score normalization with explicit zero-std check."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    result = np.zeros_like(data, dtype=float)
    for col in range(data.shape[1]):
        if std[col] == 0:
            result[:, col] = 0.0
        else:
            result[:, col] = (data[:, col] - mean[col]) / std[col]
    return result


def minmax_scale_guarded(arr):
    """Min-max scaling with explicit range check before division."""
    min_val = arr.min(axis=0)
    max_val = arr.max(axis=0)
    data_range = max_val - min_val
    if np.any(data_range == 0):
        safe_range = np.where(data_range == 0, 1.0, data_range)
        return np.where(data_range == 0, 0.0, (arr - min_val) / safe_range)
    return (arr - min_val) / data_range


def normalize_vectors_masked(vectors):
    """L2 normalization using boolean mask to skip zero-norm rows."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    nonzero_mask = (norms > 0).flatten()
    result = np.zeros_like(vectors)
    result[nonzero_mask] = vectors[nonzero_mask] / norms[nonzero_mask]
    return result


def robust_feature_scaling(X):
    """Per-feature scaling with explicit std check and fallback."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    scaled = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        centered = X[:, j] - means[j]
        if stds[j] > 0:
            scaled[:, j] = centered / stds[j]
        else:
            scaled[:, j] = 0.0
    return scaled


features = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
normed = z_score_with_std_check(features)
