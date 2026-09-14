# customer-churn-ml-j-a

Prediccion de abandono de clientes (churn) — Laboratorio de Mineria de Datos (ISTEA).  
Equipo: Javier + Andrea.

Clasificacion binaria: estimar si un cliente de telecomunicaciones se va (`Churn`).

## Que hay hecho

Solo los notebooks de eda y exploracion:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_exploracion_modelos.ipynb`

El resto de la estructura esta creada vacia, para que Andrea continue.

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
