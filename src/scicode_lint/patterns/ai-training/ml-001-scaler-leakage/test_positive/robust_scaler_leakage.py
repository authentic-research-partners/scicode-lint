import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def preprocess_features(X, y):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.25, random_state=42)


def scale_all_data(train_features, test_features):
    combined = np.vstack([train_features, test_features])
    scaler = RobustScaler()
    scaled = scaler.fit_transform(combined)
    n_train = len(train_features)
    return scaled[:n_train], scaled[n_train:]
