"""Fixtures compartidas por los tests."""

from pathlib import Path

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.data import load_data

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
EJEMPLO_VALIDO = CARPETA_PROYECTO / "examples" / "predict_request_valid.json"


# Esta funcion sirve para darle a los tests un cliente valido (el JSON de examples/)
@pytest.fixture
def sample_customer():
    return json.loads(EJEMPLO_VALIDO.read_text(encoding="utf-8"))


# Esta funcion sirve para darle a los tests el CSV historico (necesita dvc pull antes)
@pytest.fixture
def df_raw():
    return load_data()


# Esta funcion sirve para levantar la API en memoria y pegarle sin uvicorn
@pytest.fixture
def client():
    # como context manager para que corra el startup de la API
    with TestClient(app) as c:
        yield c
