import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_income

app = FastAPI(
    title="ML Reliability & Robustness API",
    description="Income prediction using our trained ML pipeline",
    version="1.1.0",
)

NUMERIC_FEATURES = {
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
}

CATEGORICAL_FEATURES = {
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
}

EXPECTED_FEATURES = NUMERIC_FEATURES | CATEGORICAL_FEATURES


class PredictionRequest(BaseModel):
    features: dict[str, object]


def validate_features(features: dict) -> dict:
    supplied = set(features)
    missing = EXPECTED_FEATURES - supplied
    unexpected = supplied - EXPECTED_FEATURES

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Missing required features", "fields": sorted(missing)},
        )

    if unexpected:
        raise HTTPException(
            status_code=422,
            detail={"error": "Unexpected features", "fields": sorted(unexpected)},
        )

    cleaned = {}

    for name in NUMERIC_FEATURES:
        value = features[name]

        if value is None:
            cleaned[name] = np.nan
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be a number or null.",
            )
        elif not math.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be finite.",
            )
        else:
            cleaned[name] = value

    for name in CATEGORICAL_FEATURES:
        value = features[name]

        if value is None:
            cleaned[name] = np.nan
        elif not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be a non-empty string or null.",
            )
        else:
            cleaned[name] = value.strip()

    return cleaned


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    validated = validate_features(request.features)
    features_df = pd.DataFrame([validated])
    result = predict_income(features_df)

    return {
        "prediction": result["predictions"][0],
        "probability": result["probabilities"][0],
        "threshold": result["threshold"],
    }
