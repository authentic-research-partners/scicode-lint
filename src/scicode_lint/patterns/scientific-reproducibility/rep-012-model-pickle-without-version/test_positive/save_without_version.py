import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def save_model_pickle(pipeline_dir: Path, X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_train)

    with open(pipeline_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(pipeline_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(pipeline_dir / "metadata.pkl", "wb") as f:
        pickle.dump({"n_features": X_train.shape[1], "classes": list(model.classes_)}, f)
