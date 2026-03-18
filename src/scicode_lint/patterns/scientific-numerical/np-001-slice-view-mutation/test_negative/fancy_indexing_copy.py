import numpy as np


def filter_by_mask(data, threshold):
    mask = data > threshold
    filtered = data[mask]
    filtered_scaled = filtered * 2.0
    return filtered_scaled


def select_by_indices(matrix, row_ids):
    selected = matrix[row_ids]
    centered = selected - selected.mean(axis=0)
    return centered


def transform_with_where(signal, cutoff):
    cleaned = np.where(signal > cutoff, signal, 0.0)
    return cleaned / (cleaned.max() + 1e-8)


def aggregate_slices(data, n_chunks):
    chunk_size = len(data) // n_chunks
    means = np.array([
        data[i * chunk_size:(i + 1) * chunk_size].mean()
        for i in range(n_chunks)
    ])
    return means


arr = np.random.rand(100)
result = filter_by_mask(arr, 0.5)
