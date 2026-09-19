"""Preprocessing del Pipeline (impute, encode, scale).

Segunda parte del train: las numericas y las categoricas
no se tratan igual. Esto se mete adentro del Pipeline
para que train e inferencia usen lo mismo.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# estas 4 son numericas; el resto de X son categoricas
NUMERICAS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def build_preprocessor(X):
    """Arma el ColumnTransformer segun las columnas de X."""
    # el resto de columnas de X son categoricas
    categoricas = [c for c in X.columns if c not in NUMERICAS]

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
            ("num", pipe_num, NUMERICAS),
            ("cat", pipe_cat, categoricas),
        ]
    )
    return preprocesador
