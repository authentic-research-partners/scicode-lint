import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TimeSeriesValidator:
    """Validator for stock price prediction with cross-validation."""

    def __init__(self, n_splits=5):
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=50))]
        )

    def validate(self, df: pd.DataFrame, target: str):
        X = df.drop(columns=[target, "timestamp"])
        y = df[target]
        scores = []
        for train_idx, test_idx in self.cv.split(X):
            self.pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(self.pipeline.score(X.iloc[test_idx], y.iloc[test_idx]))
        return scores


def evaluate_daily_forecast(sales_df):
    from sklearn.model_selection import StratifiedKFold

    X = sales_df.drop(columns=["sales", "date"])
    y = (sales_df["sales"] > sales_df["sales"].median()).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = []
    for train, test in skf.split(X, y):
        results.append((train.shape[0], test.shape[0]))
    return results
