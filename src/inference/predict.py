"""Inferencia por consola (y la logica que usa la API).

    python -m src.inference.predict examples/predict_request_valid.json

Carga el joblib del candidato, predice churn y arma
prediction / probability / risk_level / model_version.
"""

from pathlib import Path

import argparse
import json

import joblib
import pandas as pd

# src/inference/ -> raiz del repo
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = CARPETA_PROYECTO / "models" / "churn_pipeline.joblib"

# misma version que en el Model Registry (churn-classifier)
MODEL_VERSION = "1"

# columnas de X en el train (sin customerID ni Churn)
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# LOW / MEDIUM / HIGH — umbrales del equipo (nos importa el recall)
RISK_LOW_MAX = 0.30
RISK_MEDIUM_MAX = 0.70

_modelo = None


def get_model():
    """Carga el joblib. Lo deja en memoria para no leerlo en cada request."""
    global _modelo
    if _modelo is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No esta el modelo en {MODEL_PATH}. "
                "Corre primero: python -m src.training.train"
            )
        _modelo = joblib.load(MODEL_PATH)
    return _modelo


def risk_from_probability(probability):
    """Pasa la probabilidad a LOW / MEDIUM / HIGH."""
    if probability < RISK_LOW_MAX:
        return "LOW"
    if probability < RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def predict_churn(features: dict) -> dict:
    """Corre el pipeline completo. Devuelve el dict de respuesta de la API."""
    modelo = get_model()

    faltan = [c for c in FEATURE_COLUMNS if c not in features]
    if faltan:
        raise ValueError(f"Faltan features: {faltan}")

    X = pd.DataFrame([{c: features[c] for c in FEATURE_COLUMNS}])
    pred = int(modelo.predict(X)[0])
    # clase 1 = se va (churn)
    proba = float(modelo.predict_proba(X)[0, 1])

    return {
        "prediction": "CHURN" if pred == 1 else "NO_CHURN",
        "probability": round(proba, 6),
        "risk_level": risk_from_probability(proba),
        "model_version": MODEL_VERSION,
    }


def main():
    parser = argparse.ArgumentParser(description="Predice churn desde un JSON.")
    parser.add_argument(
        "json_path",
        type=Path,
        help="ej. examples/predict_request_valid.json",
    )
    args = parser.parse_args()

    features = json.loads(args.json_path.read_text(encoding="utf-8"))
    resultado = predict_churn(features)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
