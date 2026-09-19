"""Preprocessing: impute, encode y scale. Misma logica que el notebook 02."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# igual que el notebook: estas 4 son numericas, el resto categoricas
NUMERICAS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def build_preprocessor(X):
    numericas = [c for c in NUMERICAS if c in X.columns]
    categoricas = [c for c in X.columns if c not in numericas]

    # Numericas: mediana + escalado
    pipe_num = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # Categoricas: moda + one-hot
    pipe_cat = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", pipe_num, numericas),
            ("cat", pipe_cat, categoricas),
        ]
    )
    return preprocesador
