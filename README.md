# customer-churn-ml-j-a

Prediccion de abandono de clientes (churn) — Laboratorio de Mineria de Datos (ISTEA).  
Equipo: Javier + Andrea.

Clasificacion binaria: estimar si un cliente de telecomunicaciones se va (`Churn`).

## Que hay hecho

Notebooks de eda y exploracion:

1. [notebooks/01_eda.ipynb](https://github.com/javi2481/customer-churn-ml-j-a/blob/main/notebooks/01_eda.ipynb)
2. [notebooks/02_exploracion_modelos.ipynb](https://github.com/javi2481/customer-churn-ml-j-a/blob/main/notebooks/02_exploracion_modelos.ipynb)

Entrenamiento por consola (misma logica que el notebook 02):

```text
python -m src.training.train
```

El train unico se partio en `src/` asi:

- `src/data/dataset.py` — lee el CSV y hace el split 80/20
- `src/features/preprocessor.py` — impute, one-hot y scale
- `src/training/train.py` — corre baseline, logreg y random forest
- `src/evaluation/metrics.py` — accuracy, precision, recall, F1, ROC-AUC

Falta DVC y MLflow.

## Candidato oficial

Fuente de verdad: metricas del notebook ejecutado `notebooks/02_exploracion_modelos.ipynb` (split 80/20, `random_state=42`, `stratify=y`).

**Candidato:** `LogisticRegression(C=1.0, max_iter=1000, random_state=42)` dentro del Pipeline de preprocessing.

| modelo | accuracy | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|---:|
| baseline (`most_frequent`) | 0.736 | 0.000 | 0.000 | 0.000 | 0.500 |
| **logreg C=1 (candidato)** | **0.794** | 0.663 | **0.449** | **0.535** | **0.812** |
| random forest 100 | 0.783 | 0.642 | 0.401 | 0.493 | 0.790 |

![Recall por modelo](notebooks/figuras/recall_por_modelo.png)

Nos importa sobre todo el **recall**: un falso negativo (cliente que se va y el modelo no avisa) cuesta mas que un falso positivo (llamar de mas).

- El baseline siempre predice “no se va”. Accuracy ~74%, recall 0. Por eso Accuracy no decide.
- Logreg gana a RF en recall, F1 y ROC-AUC.
- `class_weight='balanced'` **no es el candidato todavia**: no esta corrido en el notebook. Entra como run extra de MLflow. Si mejora recall, se actualizan juntos README, tabla y Model Registry.

El detalle (graficos y matriz) esta en [02_exploracion_modelos.ipynb](https://github.com/javi2481/customer-churn-ml-j-a/blob/main/notebooks/02_exploracion_modelos.ipynb).

## Estructura

```text
customer-churn-ml-j-a/
├── data/raw/            # csv historico
├── notebooks/
│   └── figuras/         # graficos de la exploracion
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
