"""Contrato de inferencia de predict_churn (sin pasar por la API)."""

import pytest

from src.inference.predict import MODEL_VERSION, predict_churn, risk_from_probability


# Esta funcion sirve para comprobar que la prediccion trae los 4 campos con tipos y valores correctos
def test_predict_contract(sample_customer):
    # predict_churn: corre el pipeline y arma prediction / probability / risk_level / model_version
    result = predict_churn(sample_customer)

    assert set(result.keys()) == {"prediction", "probability", "risk_level", "model_version"}
    assert result["prediction"] in ["CHURN", "NO_CHURN"]
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["model_version"] == MODEL_VERSION


# Esta funcion sirve para comprobar que el risk_level corresponde a la probabilidad
def test_risk_level_matches_probability(sample_customer):
    result = predict_churn(sample_customer)
    # risk_from_probability: pasa la probabilidad a LOW / MEDIUM / HIGH
    assert result["risk_level"] == risk_from_probability(result["probability"])


# Esta funcion sirve para comprobar que los umbrales LOW / MEDIUM / HIGH cortan donde dijimos
# parametrize: corre este test una vez por cada par (probability, esperado)
@pytest.mark.parametrize(
    "probability, esperado",
    [
        (0.0, "LOW"),
        (0.29, "LOW"),
        (0.30, "MEDIUM"),
        (0.69, "MEDIUM"),
        (0.70, "HIGH"),
        (1.0, "HIGH"),
    ],
)
def test_risk_thresholds(probability, esperado):
    assert risk_from_probability(probability) == esperado


# Esta funcion sirve para comprobar que si falta una feature tira ValueError
def test_missing_feature_raises_error(sample_customer):
    del sample_customer["tenure"]
    # pytest.raises: el test pasa solo si adentro se tira ValueError
    with pytest.raises(ValueError):
        predict_churn(sample_customer)
