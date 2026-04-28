from pathlib import Path
import joblib
from fastapi import FastAPI
from app.api.schemas import PredictionRequest, PredictionResponse, HealthRiskResponse
from app.api.utils import prepare_features, get_health_risk

app = FastAPI(
    title="EcoCast AI API",
    description="Air Quality Forecasting and Health Risk Prediction API",
    version="0.1.0"
)

MODEL_PATH = Path("models/xgboost_model.pkl")
model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "EcoCast AI API is running",
        "model_loaded": True
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_df = prepare_features(request.model_dump())
    pred = float(model.predict(input_df)[0])

    return PredictionResponse(predicted_pm25=round(pred, 2))


@app.post("/health-risk", response_model=HealthRiskResponse)
def health_risk(request: PredictionRequest):
    input_df = prepare_features(request.model_dump())
    pred = float(model.predict(input_df)[0])

    category, message = get_health_risk(pred)

    return HealthRiskResponse(
        predicted_pm25=round(pred, 2),
        risk_category=category,
        health_message=message
    )
