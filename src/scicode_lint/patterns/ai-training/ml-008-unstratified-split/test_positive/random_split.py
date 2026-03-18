from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def split_and_classify(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


def create_holdout(features, labels, num_classes=5):
    n_test = int(len(features) * 0.3)
    indices = list(range(len(features)))
    import random

    random.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]
