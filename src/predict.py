from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "adult_income_pipeline.joblib"
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline and threshold once per process."""
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    threshold = float(config["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Model threshold must be between 0 and 1.")

    return model, threshold


def predict_income(features):
    """Return binary predictions, probabilities, and threshold."""
    model, threshold = load_model()

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "threshold": threshold,
    }
