# customer-churn-ml-j-a

Prediccion de abandono de clientes (churn) — Laboratorio de Mineria de Datos (ISTEA).  
Equipo: Javier + Andrea.

Clasificacion binaria: estimar si un cliente de telecomunicaciones se va (`Churn`).

## Que hay hecho

Solo los notebooks de eda y exploracion:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_exploracion_modelos.ipynb`

El resto de la estructura esta creada vacia, para ir completando (src, DVC, MLflow).

## Seleccion del modelo candidato

Evaluacion en test (80/20 estratificado, `random_state=42`).  
Metrica guia: **recall** (menos falsos negativos). Un FN es un cliente que se va y el modelo no avisa.

| modelo | accuracy | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|---:|
| baseline (Dummy most_frequent) | 0.736 | 0.000 | 0.000 | 0.000 | 0.500 |
| **logreg** | **0.794** | **0.663** | **0.449** | **0.535** | **0.812** |
| rf (RandomForest 100) | 0.783 | 0.642 | 0.401 | 0.493 | 0.790 |

### Decision

**Candidato: regresion logistica** (`LogisticRegression`, C=1.0).

- El baseline demuestra que accuracy engaña: ~74% sin detectar ningun abandono (recall 0).
- Logreg supera al baseline en recall, F1 y ROC-AUC.
- Random Forest queda cerca, pero con peor recall y AUC que logreg en esta exploracion.
- Preferimos un modelo lineal interpretable y simple de servir en las proximas etapas.

Detalle y matrices: `notebooks/02_exploracion_modelos.ipynb`.  
Los numeros se pueden confirmar despues con runs de MLflow.

## Estructura

```text
customer-churn-ml-j-a/
├── data/raw/            # csv historico
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── app/
├── tests/
├── monitoring/
├── models/
├── scripts/
├── .github/
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

## Dataset

`data/raw/customer_churn_historical.csv`
