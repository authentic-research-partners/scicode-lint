from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class AnomalyDetector:
    """Flags sensor readings outside normal operating ranges."""

    def __init__(self):
        self.model = LogisticRegression()
        self.accuracy = None

    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        return self.accuracy


class RareDiseaseClassifier:
    """Classifies patient samples for genetic condition screening."""

    def __init__(self):
        self.model = LogisticRegression()

    def evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))
        return train_acc, test_acc
