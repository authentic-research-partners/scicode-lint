from multiprocessing import Pool

import numpy as np


def compute_partial_sum(chunk):
    return np.sum(chunk**2)


def parallel_total_energy(signal, n_workers=4):
    chunks = np.array_split(signal, n_workers)
    with Pool(n_workers) as pool:
        total = sum(pool.imap_unordered(compute_partial_sum, chunks))
    return total


data = np.random.randn(100000)
energy = parallel_total_energy(data)
