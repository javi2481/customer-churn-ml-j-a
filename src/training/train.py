"""Entrenamiento por consola con MLflow.

    python -m src.training.train

Corre varios modelos, guarda cada uno como un Run en MLflow
y deja el candidato (logreg C=1) en models/.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_data, split_data
from src.evaluation import evaluate_model, print_metrics
from src.features import build_preprocessor

# src/training/ -> raiz del repo
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = CARPETA_PROYECTO / "models" / "churn_pipeline.joblib"
MLRUNS_PATH = CARPETA_PROYECTO / "mlruns"

# nombre del experimento en MLflow
EXPERIMENTO = "churn-entrega-1"

# el candidato que guardamos en joblib (tiene que coincidir con un run)
CANDIDATO = "logreg_C1"


def build_model_pipeline(modelo, X_train):
    """Pega el preprocessing al modelo. Se guarda junto, no aparte."""
    return Pipeline(
        steps=[
            ("preprocesar", build_preprocessor(X_train)),
            ("modelo", modelo),
        ]
    )


def correr_run(nombre, modelo, X_train, y_train, X_test, y_test, params, usar_pipeline=True):
    """Entrena un modelo, lo evalua y lo anota en MLflow."""
    with mlflow.start_run(run_name=nombre):
        # que modelo y con que hiperparametros
        mlflow.log_param("modelo", nombre)
        for clave, valor in params.items():
            mlflow.log_param(clave, valor)

        if usar_pipeline:
            pipe = build_model_pipeline(modelo, X_train)
            pipe.fit(X_train, y_train)
            entrenado = pipe
        else:
            # el dummy no necesita preprocessing
            modelo.fit(X_train, y_train)
            entrenado = modelo

        metrics = evaluate_model(entrenado, X_test, y_test)
        print_metrics(nombre.upper(), metrics)

        for clave, valor in metrics.items():
            mlflow.log_metric(clave, valor)

        # guardamos el modelo adentro del run
        mlflow.sklearn.log_model(entrenado, "model")

        return entrenado


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("filas, columnas:", df.shape)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    # los runs se guardan en la carpeta mlruns/ del proyecto
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    mlflow.set_experiment(EXPERIMENTO)

    # --- 6 runs razonados (no son 6 veces lo mismo) ---

    # 1) piso: siempre dice que el cliente se queda
    correr_run(
        "baseline",
        DummyClassifier(strategy="most_frequent"),
        X_train,
        y_train,
        X_test,
        y_test,
        params={"strategy": "most_frequent"},
        usar_pipeline=False,
    )

    # 2) candidato del notebook / README
    pipe_candidato = correr_run(
        CANDIDATO,
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        X_train,
        y_train,
        X_test,
        y_test,
        params={"C": 1.0, "max_iter": 1000, "class_weight": "none"},
    )

    # 3) arboles default del notebook
    correr_run(
        "rf_100",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X_train,
        y_train,
        X_test,
        y_test,
        params={"n_estimators": 100, "max_depth": "none"},
    )

    # 4) logreg mas regularizada
    correr_run(
        "logreg_C0.5",
        LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        X_train,
        y_train,
        X_test,
        y_test,
        params={"C": 0.5, "max_iter": 1000, "class_weight": "none"},
    )

    # 5) logreg con peso para la clase churn (desbalance)
    correr_run(
        "logreg_balanced",
        LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=42
        ),
        X_train,
        y_train,
        X_test,
        y_test,
        params={"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    )

    # 6) RF un poco mas limitado (menos overfitting)
    correr_run(
        "rf_depth10",
        RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        X_train,
        y_train,
        X_test,
        y_test,
        params={"n_estimators": 100, "max_depth": 10},
    )

    # joblib del candidato (mismo que el run logreg_C1)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe_candidato, MODEL_PATH)
    print("\nCandidato guardado:", MODEL_PATH)
    print("Runs en MLflow. Para verlos: mlflow ui --backend-store-uri", MLRUNS_PATH)


if __name__ == "__main__":
    main()
