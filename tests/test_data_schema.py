"""Contrato minimo del dataset: columnas y target."""

from src.inference.predict import FEATURE_COLUMNS


# Esta funcion sirve para comprobar que el CSV tiene todas las columnas que usa el modelo
def test_required_columns_exist(df_raw):
    # FEATURE_COLUMNS: lista de columnas que recibe el modelo
    # issubset: todas las de la lista estan en el CSV
    assert set(FEATURE_COLUMNS).issubset(df_raw.columns)


# Esta funcion sirve para comprobar que esta la columna Churn y solo tiene Yes / No
def test_target_exists_in_training(df_raw):
    assert "Churn" in df_raw.columns
    assert set(df_raw["Churn"].unique()) == {"Yes", "No"}


# Esta funcion sirve para comprobar que TotalCharges queda como numero
def test_total_charges_is_numeric(df_raw):
    # load_data lo pasa de texto a numero
    assert df_raw["TotalCharges"].dtype == "float64"
