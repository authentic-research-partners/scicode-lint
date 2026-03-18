import numpy as np


def rank_with_stable_sort(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores, kind="stable")


def sort_stable(values: np.ndarray) -> np.ndarray:
    return np.sort(values, kind="stable")


def lexsort_tiebreaker(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    return np.lexsort((secondary, primary))
