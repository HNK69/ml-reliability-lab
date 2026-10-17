from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pandas as pd

from src.predict import predict_income

app = FastAPI(
    title="ML Reliability & Robustness API",
    description="Income prediction using our trained ML pipeline",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    features: dict


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features_df = pd.DataFrame([request.features])
        result = predict_income(features_df)

        return {
            "prediction": result["predictions"][0],
            "probability": result["probabilities"][0],
            "threshold": 0.39,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc
