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
    numericas = [c for c in NUMERICAS if c in X.columns]
    categoricas = [c for c in X.columns if c not in numericas]

    # nulos de TotalCharges (tenure=0) -> mediana
    pipe_num = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # texto -> numeros (one-hot). handle_unknown: si aparece una categoria nueva, no explota
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
