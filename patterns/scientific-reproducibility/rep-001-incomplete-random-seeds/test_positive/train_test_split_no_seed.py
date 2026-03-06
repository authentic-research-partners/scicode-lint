"""train_test_split without random_state - non-reproducible splits."""
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data():
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # BUG: No random_state parameter - different splits each run
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
