"""Train and evaluate a K-Nearest Neighbors classifier on the HR Attrition dataset.

Run once locally (from repo root):
    python model/knn.py

Produces:
    model/knn.pkl
"""

import os
import sys

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_train_test

MODEL_PATH = os.path.join(os.path.dirname(__file__), "knn.pkl")


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_train_test()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    joblib.dump(model, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    metrics = train_and_evaluate()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
