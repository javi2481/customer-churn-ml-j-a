"""API de inferencia con FastAPI.

    uvicorn app.main:app --reload --port 8000

GET /health  — si el servicio esta arriba y que version de modelo cargo
POST /predict — valida el JSON y corre el pipeline
Swagger queda en /docs
"""

from fastapi import FastAPI, HTTPException

from app.schemas import CustomerFeatures, HealthResponse, PredictResponse
from src.inference.predict import MODEL_VERSION, get_model, predict_churn

app = FastAPI(title="Customer Churn API", version=MODEL_VERSION)


@app.on_event("startup")
def startup():
    # si no esta el joblib, mejor fallar al arrancar
    get_model()


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=MODEL_VERSION,
        )
    except Exception:
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_version=None,
        )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: CustomerFeatures):
    try:
        return predict_churn(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
