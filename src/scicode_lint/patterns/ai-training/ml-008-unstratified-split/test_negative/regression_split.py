from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split


def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_squared_error(y_test, predictions)


def cross_validate_regression(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    mse_scores = []
    for train_idx, test_idx in kf.split(X):
        model = Ridge(alpha=0.5)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        mse_scores.append(mean_squared_error(y[test_idx], preds))
    return sum(mse_scores) / len(mse_scores)
