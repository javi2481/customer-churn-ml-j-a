"""El pipeline (preprocessing + modelo) entrena y predice sin romperse."""

from sklearn.linear_model import LogisticRegression

from src.data import split_data
from src.training.train import build_model_pipeline


def test_preprocessor_handles_missing_values(df_raw):
    """Entrena el pipeline con nulos en los datos y igual predice 5 filas."""
    # split_data: separa el CSV en train/test 80/20 (X sin Churn ni customerID)
    X_train, X_test, y_train, y_test = split_data(df_raw)

    # TotalCharges trae nulos despues del to_numeric
    assert X_train["TotalCharges"].isna().sum() > 0

    # build_model_pipeline: arma preprocessing (imputar, escalar, one-hot) + el modelo
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000), X_train)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test.head(5))

    assert len(preds) == 5


def test_split_does_not_include_target_or_id(df_raw):
    """Churn y customerID no quedan en X (no son predictores)."""
    # split_data: separa el CSV en train/test 80/20
    X_train, X_test, y_train, y_test = split_data(df_raw)

    for col in ["Churn", "customerID"]:
        assert col not in X_train.columns
        assert col not in X_test.columns
