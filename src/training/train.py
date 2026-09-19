"""Entrenamiento por consola: python -m src.training.train

Carga datos, arma el Pipeline, compara baseline / logreg / RF
y guarda el candidato (logreg).
"""

from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_data, split_data
from src.evaluation import evaluate_model, print_metrics
from src.features import build_preprocessor

# este archivo esta en src/training/ -> subir 3 carpetas = raiz del proyecto
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = CARPETA_PROYECTO / "models" / "churn_pipeline.joblib"


def build_model_pipeline(modelo, X_train):
    # preprocessing + estimador juntos, para que inferencia use lo mismo
    return Pipeline(
        steps=[
            ("preprocesar", build_preprocessor(X_train)),
            ("modelo", modelo),
        ]
    )


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("filas, columnas:", df.shape)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    # Siempre predice No churn. No usa las features: es el piso.
    modelo_baseline = DummyClassifier(strategy="most_frequent")
    modelo_baseline.fit(X_train, y_train)
    print_metrics("BASELINE", evaluate_model(modelo_baseline, X_test, y_test))

    # Candidato del README / notebook: logreg C=1
    modelo_logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    pipe_logreg = build_model_pipeline(modelo_logreg, X_train)
    pipe_logreg.fit(X_train, y_train)
    print_metrics("LOGISTIC REGRESSION", evaluate_model(pipe_logreg, X_test, y_test))

    # Modelo de arboles
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pipe_rf = build_model_pipeline(modelo_rf, X_train)
    pipe_rf.fit(X_train, y_train)
    print_metrics("RANDOM FOREST", evaluate_model(pipe_rf, X_test, y_test))

    # Guardamos el pipeline entero (prepro + modelo), no solo el estimador
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe_logreg, MODEL_PATH)
    print("\nCandidato guardado:", MODEL_PATH)


if __name__ == "__main__":
    main()
