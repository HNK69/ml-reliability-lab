import json
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "adult_income_pipeline.joblib"
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"


def load_model():
    """Load the trained pipeline and its decision threshold."""
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return model, float(config["threshold"])


def predict_income(features):
    """Predict income class and probability for input feature rows."""
    model, threshold = load_model()
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }
