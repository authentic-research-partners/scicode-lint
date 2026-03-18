from sklearn.model_selection import GroupShuffleSplit, RepeatedStratifiedKFold


def get_group_splitter(n_splits: int = 5, test_size: float = 0.2):
    return GroupShuffleSplit(n_splits=n_splits, test_size=test_size)


def get_repeated_stratified_cv(n_splits: int = 5, n_repeats: int = 3):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)


def evaluate_with_group_cv(model, X, y, groups):
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.2)
    scores = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return scores
