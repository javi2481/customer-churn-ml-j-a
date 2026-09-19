"""Metricas de los modelos.

Aca no se entrena: solo se mira que tan bien salio el test.
Accuracy sola no sirve (el dummy ya acierta ~74% sin detectar churn).
En este negocio nos importa mas el recall.
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    """Calcula metricas. zero_division=0: si no predice ningun churn, precision queda 0."""
    pred = model.predict(X_test)
    # probabilidad de clase 1 (churn) para el ROC-AUC
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
