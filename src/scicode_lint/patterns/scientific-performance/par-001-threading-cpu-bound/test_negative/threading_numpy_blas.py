from concurrent.futures import ThreadPoolExecutor

import numpy as np


def svd_worker(matrix):
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U @ np.diag(s) @ Vt


def parallel_svd_reconstruction(matrices, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        reconstructed = list(executor.map(svd_worker, matrices))
    return reconstructed


def compute_gram_matrices(data_batches, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(np.dot, batch.T, batch) for batch in data_batches]
        results = [f.result() for f in futures]
    return results


batches = [np.random.randn(500, 200) for _ in range(8)]
gram = compute_gram_matrices(batches)
reconstructions = parallel_svd_reconstruction(batches)
