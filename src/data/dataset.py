"""Carga del CSV y particion train/test.

Primera parte del train: leer el dataset, tipar TotalCharges
y armar X/y. El id del cliente no se usa para predecir.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Este archivo esta en src/data/ -> subir 3 carpetas llega a la raiz del repo
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent.parent
CSV_PATH = CARPETA_PROYECTO / "data" / "raw" / "customer_churn_historical.csv"

# misma semilla y tamanio que el notebook, para que el split no cambie
RANDOM_STATE = 42
TEST_SIZE = 0.20


# Esta funcion sirve para leer el CSV historico y pasar TotalCharges a numero
def load_data(path=None) -> pd.DataFrame:
    if path is None:
        path = CSV_PATH

    df = pd.read_csv(path)

    # TotalCharges a veces viene como texto; lo pasamos a numerico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


# Esta funcion sirve para separar el dataset en train/test 80/20 y armar X / y
def split_data(df: pd.DataFrame):
    """stratify mantiene el % de churn en train y en test."""
    df = df.copy()

    # Yes -> 1 (se va), No -> 0 (se queda)
    df["churn_0_1"] = 0
    df.loc[df["Churn"] == "Yes", "churn_0_1"] = 1

    y = df["churn_0_1"]

    # customerID no es predictor; Churn ya esta copiado en y
    X = df.drop(columns=["Churn", "churn_0_1", "customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # train_test_split siempre devuelve DataFrames/Series aca
    return X_train, X_test, y_train, y_test
