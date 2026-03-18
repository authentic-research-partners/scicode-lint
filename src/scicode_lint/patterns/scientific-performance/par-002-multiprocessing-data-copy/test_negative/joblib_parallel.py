import numpy as np
from joblib import Parallel, delayed


def process_chunk(chunk):
    centered = chunk - np.mean(chunk, axis=0)
    return np.cov(centered.T)


def analyze_dataset(data, n_jobs=4):
    chunks = np.array_split(data, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(c) for c in chunks)
    return np.mean(results, axis=0)


dataset = np.random.randn(100000, 50)
cov_matrix = analyze_dataset(dataset)
