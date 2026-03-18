import numpy as np


def normalize_data(arr):
    data = arr.copy()
    data = data - data.mean()
    data = data / data.std()
    return data
