import numpy as np
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_score,
)


def evaluate_with_seeded_kfold(estimator, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(estimator, X, y, cv=kfold)


def evaluate_with_seeded_stratified(estimator, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    return cross_val_score(estimator, X, y, cv=skf)


def evaluate_with_seeded_shuffle(estimator, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    return cross_val_score(estimator, X, y, cv=ss)


def evaluate_repeated(estimator, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    return cross_val_score(estimator, X, y, cv=rskf)
