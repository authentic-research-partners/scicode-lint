from multiprocessing import Pool

import numpy as np


def process_chunk(chunk):
    normalized = (chunk - chunk.mean(axis=0)) / (chunk.std(axis=0) + 1e-8)
    cov = np.cov(normalized, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov)
    return eigenvalues[-5:]


def parallel_pca_analysis(data, num_workers=4):
    chunks = np.array_split(data, num_workers)

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    return np.concatenate(results)


sensor_readings = np.random.randn(100000, 50)
top_eigenvalues = parallel_pca_analysis(sensor_readings)
