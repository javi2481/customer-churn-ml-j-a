"""Contrato de inferencia de predict_churn (sin pasar por la API)."""

import pytest

from src.inference.predict import MODEL_VERSION, predict_churn, risk_from_probability


def test_predict_contract(sample_customer):
    """La prediccion trae los 4 campos con tipos y valores correctos."""
    result = predict_churn(sample_customer)

    assert set(result.keys()) == {"prediction", "probability", "risk_level", "model_version"}
    assert result["prediction"] in ["CHURN", "NO_CHURN"]
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["model_version"] == MODEL_VERSION


def test_risk_level_matches_probability(sample_customer):
    """El risk_level corresponde a la probabilidad devuelta."""
    result = predict_churn(sample_customer)
    assert result["risk_level"] == risk_from_probability(result["probability"])


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
    """Los umbrales LOW / MEDIUM / HIGH cortan donde dijimos."""
    assert risk_from_probability(probability) == esperado


def test_missing_feature_raises_error(sample_customer):
    """Si falta una feature tira ValueError."""
    del sample_customer["tenure"]
    with pytest.raises(ValueError):
        predict_churn(sample_customer)
