import numpy as np
import pandas as pd


class CrossValidationSampler:
    def __init__(self, df: pd.DataFrame, n_folds: int = 5, seed: int = 42):
        self.df = df
        self.n_folds = n_folds
        self.rng = np.random.RandomState(seed)
        self._fold_indices = self._create_folds()

    def _create_folds(self) -> list[np.ndarray]:
        shuffled = self.df.sample(frac=1.0, random_state=self.rng).index
        return np.array_split(shuffled, self.n_folds)

    def get_fold(self, fold_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_idx = self._fold_indices[fold_idx]
        train_idx = self.df.index.difference(test_idx)
        return self.df.loc[train_idx], self.df.loc[test_idx]

    def bootstrap_evaluate(self, n_rounds: int = 100) -> list[pd.DataFrame]:
        samples = []
        for i in range(n_rounds):
            boot = self.df.sample(frac=1.0, replace=True, random_state=self.rng.randint(0, 2**31))
            samples.append(boot)
        return samples
