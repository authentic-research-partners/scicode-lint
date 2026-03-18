import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def evaluate_sensor_anomaly_detector(sensor_log: pd.DataFrame):
    sensor_log = sensor_log.sort_values("timestamp")
    X = sensor_log[["vibration", "temperature", "pressure", "rpm"]].values
    y = sensor_log["is_anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4)
    fold_scores = []
    for train_idx, test_idx in skf.split(X_scaled, y):
        model.fit(X_scaled[train_idx], y[train_idx])
        fold_scores.append(model.score(X_scaled[test_idx], y[test_idx]))
    return fold_scores
