import numpy as np


def normalize_vector(vector):
    """Normalize a vector to unit length, returning zeros for null vectors."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def compute_mean_embedding(embeddings, counts):
    """Average embedding vectors weighted by occurrence counts."""
    total = 0
    weighted_sum = np.zeros(embeddings.shape[1])
    for emb, count in zip(embeddings, counts):
        if count == 0:
            continue
        weighted_sum += emb * count
        total += count
    if total == 0:
        return np.zeros(embeddings.shape[1])
    return weighted_sum / total


def compute_ratio(numerator, denominator):
    """Compute ratio of two values, returning infinity when denominator is absent."""
    if denominator != 0:
        return numerator / denominator
    return float("inf")


def project_onto_subspace(vectors, basis):
    """Project each vector onto the subspace spanned by basis vectors."""
    projected = np.zeros_like(vectors)
    for i, v in enumerate(vectors):
        for b in basis:
            b_norm = np.linalg.norm(b)
            if b_norm == 0:
                continue
            projected[i] += np.dot(v, b) / (b_norm**2) * b
    return projected
