from pathlib import Path

import json

import pytest

# tests/ -> raiz del repo
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
EJEMPLO_INVALIDO = CARPETA_PROYECTO / "examples" / "predict_request_invalid.json"


def test_predict_success(client, sample_customer):
    """Cliente valido: 200 y los 4 campos con sus tipos."""
    # client.post: manda el cliente como JSON a /predict
    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert "model_version" in data

    assert isinstance(data["prediction"], str)
    assert isinstance(data["probability"], float)
    assert isinstance(data["risk_level"], str)
    assert isinstance(data["model_version"], str)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_payload(client):
    """El JSON invalido de examples/ devuelve 422."""
    # json.loads: lee el archivo y lo pasa a dict
    payload = json.loads(EJEMPLO_INVALIDO.read_text(encoding="utf-8"))

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# parametrize: corre el test una vez por cada (campo, valor invalido)
@pytest.mark.parametrize(
    "campo, valor",
    [
        ("tenure", -1),
        ("MonthlyCharges", "alto"),
        ("Contract", "Weekly"),
        ("InternetService", None),
        ("SeniorCitizen", 2),
    ],
)
def test_invalid_customer_is_rejected(client, sample_customer, campo, valor):
    """Un campo con valor invalido devuelve 422."""
    sample_customer[campo] = valor

    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 422


def test_missing_field_is_rejected(client, sample_customer):
    """Si falta un campo devuelve 422."""
    del sample_customer["Contract"]

    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 422


def test_empty_body_is_rejected(client):
    """Un JSON vacio devuelve 422."""
    response = client.post("/predict", json={})

    assert response.status_code == 422
