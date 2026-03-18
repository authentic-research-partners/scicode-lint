import numpy as np


def compute_pairwise_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        diff = X[i] - X
        squared = diff**2
        summed = np.sum(squared, axis=1)
        distances[i] = np.sqrt(summed)
    return distances


def normalize_rows(matrix):
    diff = matrix - np.mean(matrix, axis=1, keepdims=True)
    squared = diff**2
    row_norms = np.sqrt(np.sum(squared, axis=1, keepdims=True))
    return diff / (row_norms + 1e-8)


points = np.random.randn(2000, 50)
dist_matrix = compute_pairwise_distances(points)
normed = normalize_rows(points)
