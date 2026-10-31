"""Fixtures compartidas por los tests."""

from pathlib import Path

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.data import load_data

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
EJEMPLO_VALIDO = CARPETA_PROYECTO / "examples" / "predict_request_valid.json"


@pytest.fixture
def sample_customer():
    """Cliente valido (el mismo JSON de examples/)."""
    return json.loads(EJEMPLO_VALIDO.read_text(encoding="utf-8"))


@pytest.fixture
def df_raw():
    """CSV historico. Necesita dvc pull antes."""
    return load_data()


@pytest.fixture
def client():
    """Levanta la API en memoria para pegarle sin uvicorn."""
    # como context manager para que corra el startup de la API
    with TestClient(app) as c:
        yield c
