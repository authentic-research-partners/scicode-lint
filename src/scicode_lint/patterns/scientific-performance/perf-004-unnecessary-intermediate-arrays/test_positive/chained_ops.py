import numpy as np


def process_data(data, weights):
    diff = data - weights
    squared = diff**2
    total = np.sum(squared, axis=1)
    result = np.sqrt(total)
    return result


def compute_cosine_similarity(a, b):
    prod = a * b
    dot = np.sum(prod, axis=1)
    norm_a_sq = np.sum(a**2, axis=1)
    norm_a = np.sqrt(norm_a_sq)
    norm_b_sq = np.sum(b**2, axis=1)
    norm_b = np.sqrt(norm_b_sq)
    similarity = dot / (norm_a * norm_b + 1e-8)
    return similarity


X = np.random.randn(10000, 100)
W = np.random.randn(10000, 100)
distances = process_data(X, W)
sims = compute_cosine_similarity(X, W)
