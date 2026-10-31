"""Contrato minimo del dataset: columnas y target."""

from src.inference.predict import FEATURE_COLUMNS


def test_required_columns_exist(df_raw):
    assert set(FEATURE_COLUMNS).issubset(df_raw.columns)


def test_target_exists_in_training(df_raw):
    assert "Churn" in df_raw.columns
    assert set(df_raw["Churn"].unique()) == {"Yes", "No"}


def test_total_charges_is_numeric(df_raw):
    # load_data lo pasa de texto a numero
    assert df_raw["TotalCharges"].dtype == "float64"
