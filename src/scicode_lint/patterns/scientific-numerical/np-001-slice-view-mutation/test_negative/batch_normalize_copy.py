import numpy as np


def normalize_new_array(data, target_mean=0, target_std=1):
    """Normalize data into a pre-allocated output array."""
    result = np.empty_like(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    result[:] = (data - mean) / std * target_std + target_mean
    return result


def rolling_mean_output(arr, window_size=5):
    """Compute rolling mean into pre-allocated output array."""
    n = len(arr) - window_size + 1
    output = np.zeros(n)
    for i in range(n):
        output[i] = arr[i : i + window_size].mean()
    return output


def stack_processed_chunks(data, chunk_size=10):
    """Process data in chunks and concatenate results."""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size] * 2
        chunks.append(chunk)
    return np.concatenate(chunks)
