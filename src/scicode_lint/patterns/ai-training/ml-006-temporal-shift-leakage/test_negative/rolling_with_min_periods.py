def rolling_with_min_periods(df):
    df["price_rolling"] = df["price"].rolling(window=7, min_periods=3).mean()
    X = df[["price_rolling"]]
    y = df["target"]
    return X, y


def shift_by_variable_amount(df):
    shift_amount = 1
    df["lagged_price"] = df["price"].shift(shift_amount)
    X = df[["lagged_price"]]
    y = df["target"]
    return X, y
