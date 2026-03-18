import random

import numpy as np


def main():
    data = np.random.randn(1000, 10)
    labels = np.random.randint(0, 3, size=1000)

    noisy = data + np.random.normal(0, 0.1, data.shape)

    indices = list(range(1000))
    random.shuffle(indices)
    train_idx = indices[:800]
    test_idx = indices[800:]

    train_data = noisy[train_idx]
    test_data = noisy[test_idx]
    return train_data, test_data, labels[train_idx], labels[test_idx]


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = main()
    print(f"Train: {train_x.shape}, Test: {test_x.shape}")
