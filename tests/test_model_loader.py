"""El joblib existe, carga y expone predict / predict_proba."""

from src.inference.predict import MODEL_PATH, get_model


# Esta funcion sirve para comprobar que el joblib del modelo existe en models/
def test_model_file_exists():
    # MODEL_PATH: ruta a models/churn_pipeline.joblib (se genera con python -m src.training.train)
    assert MODEL_PATH.exists()


# Esta funcion sirve para comprobar que el modelo carga y puede predecir
def test_model_loads():
    # get_model: lee el joblib y lo deja en memoria
    modelo = get_model()
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")
