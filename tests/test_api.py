import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data

def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_models" in data
    assert len(data["available_models"]) == 5

def test_predict_valid():
    resp = client.post("/predict", json={
        "tv": 50.0,
        "radio": 15.0,
        "social_media": 3.0,
        "influencer": "Mega",
        "model_name": "GradientBoosting"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_sales" in data
    assert data["predicted_sales"] > 0
    assert "roi_estimate" in data

def test_predict_invalid_influencer():
    resp = client.post("/predict", json={
        "tv": 50.0,
        "radio": 15.0,
        "social_media": 3.0,
        "influencer": "Invalid",
        "model_name": "GradientBoosting"
    })
    assert resp.status_code == 422

def test_predict_negative_budget():
    resp = client.post("/predict", json={
        "tv": -10.0,
        "radio": 15.0,
        "social_media": 3.0,
        "influencer": "Mega",
        "model_name": "GradientBoosting"
    })
    assert resp.status_code == 422

def test_all_models_predict():
    models = ["LinearRegression", "RandomForest", "GradientBoosting", "XGBoost"]
    for model in models:
        resp = client.post("/predict", json={
            "tv": 50.0, "radio": 15.0, "social_media": 3.0,
            "influencer": "Mega", "model_name": model
        })
        assert resp.status_code == 200, f"Echec pour modele : {model}"
