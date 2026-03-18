"""Series-level sampling in an analysis pipeline without random_state."""

import pandas as pd


def select_representative_features(df: pd.DataFrame, n_features: int = 20) -> list[str]:
    """Pick a random subset of feature columns for quick analysis."""
    feature_cols = pd.Series([c for c in df.columns if c.startswith("feat_")])
    selected = feature_cols.sample(n=n_features)
    return selected.tolist()


def estimate_quantiles(series: pd.Series, sample_size: int = 500) -> dict:
    """Estimate quantiles from a random subset of a large series."""
    subset = series.sample(n=sample_size)
    return {
        "q25": subset.quantile(0.25),
        "q50": subset.quantile(0.50),
        "q75": subset.quantile(0.75),
    }


def build_calibration_set(
    predictions: pd.Series, labels: pd.Series, frac: float = 0.1
) -> pd.DataFrame:
    """Sample a calibration set from predictions."""
    idx = predictions.sample(frac=frac).index
    return pd.DataFrame({"pred": predictions.loc[idx], "label": labels.loc[idx]})
