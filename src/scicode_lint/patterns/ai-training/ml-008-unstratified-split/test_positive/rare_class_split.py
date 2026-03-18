import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def evaluate_rare_event_detector(features, labels, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    accuracies = []
    scaler = StandardScaler()
    for fold_train, fold_test in kf.split(features):
        X_tr = scaler.fit_transform(features[fold_train])
        X_te = scaler.transform(features[fold_test])
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
        rf.fit(X_tr, labels[fold_train])
        accuracies.append(rf.score(X_te, labels[fold_test]))
    return np.median(accuracies)
