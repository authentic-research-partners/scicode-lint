from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


def evaluate_stock_prediction_shuffled(df):
    df = df.sort_values("date")
    X = df[["open", "high", "low", "volume"]]
    y = df["close"]
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    return scores


def cross_validate_temperature(weather_df):
    weather_df = weather_df.sort_values("timestamp")
    X = weather_df[["humidity", "pressure", "wind_speed"]]
    y = weather_df["temperature"]
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return scores
