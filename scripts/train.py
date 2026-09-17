"""
Predicción de Churn de Clientes (ISTEA)

1. Se carga el dataset 
2. Limpieza  necesaria
3. ASe genera la variable objetivo (churn_0_1)
4. Dividimos los datos en entrenamiento (80%) y prueba (20%)
   manteniendo la misma proporcion de clientes que abandonan
   y clientes que se quedan.
5. Pipeline de preprocessing:
   - Imputación de valores faltantes.
   - Escalado de variables numeericas.
   - One-hot Encoding para variables categoricas.
6. Entrena y evalua:
   - DummyClassifier (baseline)
   - Logistic Regression
   - Random Forest
7. Muestra metricas:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC

"""

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# CARGA


df = pd.read_csv(
    "data/raw/customer_churn_historical.csv"
)


# LIMPIEZA


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["churn_0_1"] = (
    df["Churn"] == "Yes"
).astype(int)


# FEATURES


X = df.drop(
    columns=[
        "customerID",
        "Churn",
        "churn_0_1"
    ]
)

y = df["churn_0_1"]


# SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

print("Target:")
print(y.value_counts())


# COLUMNAS


num_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

cat_cols = [
    c
    for c in X.columns
    if c not in num_cols
]

print("\nNumericas:")
print(num_cols)

print("\nCategoricas:")
print(cat_cols)


# PREPROCESSING


num_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

cat_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        num_pipe,
        num_cols
    ),
    (
        "cat",
        cat_pipe,
        cat_cols
    )
])

print("\nPreprocessor creado")


# BASELINE


modelo_baseline = DummyClassifier(
    strategy="most_frequent"
)

modelo_baseline.fit(
    X_train,
    y_train
)

pred_baseline = modelo_baseline.predict(
    X_test
)

proba_baseline = (
    modelo_baseline
    .predict_proba(X_test)[:, 1]
)

print("\nBASELINE")

print(
    "accuracy:",
    round(
        accuracy_score(
            y_test,
            pred_baseline
        ),
        3
    )
)

print(
    "precision:",
    round(
        precision_score(
            y_test,
            pred_baseline,
            zero_division=0
        ),
        3
    )
)

print(
    "recall:",
    round(
        recall_score(
            y_test,
            pred_baseline,
            zero_division=0
        ),
        3
    )
)

print(
    "f1:",
    round(
        f1_score(
            y_test,
            pred_baseline,
            zero_division=0
        ),
        3
    )
)

print(
    "roc_auc:",
    round(
        roc_auc_score(
            y_test,
            proba_baseline
        ),
        3
    )
)


# LOGISTIC REGRESSION


modelo_logreg = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)

pipe_logreg = Pipeline([
    (
        "preprocess",
        preprocessor
    ),
    (
        "model",
        modelo_logreg
    )
])

pipe_logreg.fit(
    X_train,
    y_train
)

pred_logreg = pipe_logreg.predict(
    X_test
)

proba_logreg = (
    pipe_logreg
    .predict_proba(X_test)[:, 1]
)

print("\nLOGISTIC REGRESSION")

print(
    "accuracy:",
    round(
        accuracy_score(
            y_test,
            pred_logreg
        ),
        3
    )
)

print(
    "precision:",
    round(
        precision_score(
            y_test,
            pred_logreg,
            zero_division=0
        ),
        3
    )
)

print(
    "recall:",
    round(
        recall_score(
            y_test,
            pred_logreg
        ),
        3
    )
)

print(
    "f1:",
    round(
        f1_score(
            y_test,
            pred_logreg
        ),
        3
    )
)

print(
    "roc_auc:",
    round(
        roc_auc_score(
            y_test,
            proba_logreg
        ),
        3
    )
)


# RANDOM FOREST


modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pipe_rf = Pipeline([
    (
        "preprocess",
        preprocessor
    ),
    (
        "model",
        modelo_rf
    )
])

pipe_rf.fit(
    X_train,
    y_train
)

pred_rf = pipe_rf.predict(
    X_test
)

proba_rf = (
    pipe_rf
    .predict_proba(X_test)[:, 1]
)

print("\nRANDOM FOREST")

print(
    "accuracy:",
    round(
        accuracy_score(
            y_test,
            pred_rf
        ),
        3
    )
)

print(
    "precision:",
    round(
        precision_score(
            y_test,
            pred_rf,
            zero_division=0
        ),
        3
    )
)

print(
    "recall:",
    round(
        recall_score(
            y_test,
            pred_rf
        ),
        3
    )
)

print(
    "f1:",
    round(
        f1_score(
            y_test,
            pred_rf
        ),
        3
    )
)

print(
    "roc_auc:",
    round(
        roc_auc_score(
            y_test,
            proba_rf
        ),
        3
    )
)