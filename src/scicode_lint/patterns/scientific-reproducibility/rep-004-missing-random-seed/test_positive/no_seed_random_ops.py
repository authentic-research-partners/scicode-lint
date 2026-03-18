import numpy as np
import torch


def run_pipeline(n_samples=1000, test_ratio=0.2):
    features = np.random.randn(n_samples, 50)
    noise = np.random.uniform(-0.5, 0.5, n_samples)
    targets = features.sum(axis=1) + noise

    indices = np.random.permutation(n_samples)
    split_point = int(n_samples * (1 - test_ratio))

    X_train = features[indices[:split_point]]
    y_train = targets[indices[:split_point]]
    X_test = features[indices[split_point:]]
    y_test = targets[indices[split_point:]]

    model = torch.nn.Linear(50, 1)
    return model, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    model, X_tr, y_tr, X_te, y_te = run_pipeline()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
