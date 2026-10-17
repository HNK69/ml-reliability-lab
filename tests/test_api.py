import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    features = {
        "age": 39,
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1
    assert result["threshold"] == 0.39


def test_invalid_request():
    response = client.post(
        "/predict",
        json={"wrong_field": "invalid"},
    )

    assert response.status_code == 422


def test_missing_feature_is_rejected():
    response = client.post(
        "/predict",
        json={"features": {"age": 39, "workclass": "Private"}},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "Missing required features"


def test_unexpected_feature_is_rejected():
    features = {
        "age": 39,
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "unexpected_field": "invalid",
    }

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "Unexpected features"


def test_invalid_numeric_feature_is_rejected():
    features = {
        "age": "thirty-nine",
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert "must be a number" in response.json()["detail"]


def test_null_feature_is_supported():
    features = {
        "age": 39,
        "workclass": None,
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]
