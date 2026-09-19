"""Carga y split del CSV. Misma logica que el notebook 02."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Este archivo esta en primer_parcial/src/data/
# .parent = src/data  ->  .parent.parent = src  ->  .parent.parent.parent = primer_parcial
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent.parent
CSV_PATH = CARPETA_PROYECTO / "data" / "raw" / "customer_churn_historical.csv"

# para que el split salga siempre igual
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path=None):
    if path is None:
        path = CSV_PATH

    df = pd.read_csv(path)

    # TotalCharges a veces viene como texto; lo pasamos a numerico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def split_data(df):
    # Columna nueva: Yes -> 1, No -> 0
    df = df.copy()
    df["churn_0_1"] = 0
    df.loc[df["Churn"] == "Yes", "churn_0_1"] = 1

    y = df["churn_0_1"]

    # El id y el target no se usan como predictor
    X = df.drop(columns=["Churn", "churn_0_1", "customerID"])

    # Particion train/test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
