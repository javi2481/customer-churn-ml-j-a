"""Metricas de clasificacion. Accuracy sola no alcanza: tambien recall."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def print_metrics(nombre, metrics):
    print(f"\n{nombre}")
    print("accuracy:", round(metrics["accuracy"], 3))
    print("precision:", round(metrics["precision"], 3))
    print("recall:", round(metrics["recall"], 3))
    print("f1:", round(metrics["f1"], 3))
    print("roc_auc:", round(metrics["roc_auc"], 3))
