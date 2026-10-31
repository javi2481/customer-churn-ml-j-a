"""El joblib existe, carga y expone predict / predict_proba."""

from src.inference.predict import MODEL_PATH, get_model


def test_model_file_exists():
    """El joblib esta en models/."""
    # MODEL_PATH: ruta a models/churn_pipeline.joblib (se genera con python -m src.training.train)
    assert MODEL_PATH.exists()


def test_model_loads():
    """El modelo carga y tiene predict y predict_proba."""
    # get_model: lee el joblib y lo deja en memoria
    modelo = get_model()
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")
