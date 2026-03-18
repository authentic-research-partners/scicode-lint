import numpy as np


def column_stddev(data):
    mean = np.mean(data, axis=0)
    diff = data - mean
    squared = diff**2
    variance = np.mean(squared, axis=0)
    stddev = np.sqrt(variance)
    return stddev


def frobenius_distance(A, B):
    diff = A - B
    squared = diff**2
    total = np.sum(squared)
    dist = np.sqrt(total)
    return dist


measurements = np.random.randn(50000, 200)
stds = column_stddev(measurements)
ref = np.random.randn(50000, 200)
fdist = frobenius_distance(measurements, ref)
