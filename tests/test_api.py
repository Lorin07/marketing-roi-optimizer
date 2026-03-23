import pytest
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app, MODEL_STORE, MODEL_METRICS

# Pre-chargement des modeles pour les tests (simule le lifespan)
@pytest.fixture(autouse=True)
def load_models_for_tests():
    models_dir = Path("outputs/models/")
    for name in ["LinearRegression", "RandomForest", "GradientBoosting", "XGBoost"]:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            MODEL_STORE[name] = joblib.load(path)
    mlp_path = models_dir / "MLP_bundle.joblib"
    if mlp_path.exists():
        MODEL_STORE["MLP"] = joblib.load(mlp_path)
    yield
    MODEL_STORE.clear()

client = TestClient(app, raise_server_exceptions=False)

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
        "tv": 50.0, "radio": 15.0, "social_media": 3.0,
        "influencer": "Invalid", "model_name": "GradientBoosting"
    })
    assert resp.status_code == 422

def test_predict_negative_budget():
    resp = client.post("/predict", json={
        "tv": -10.0, "radio": 15.0, "social_media": 3.0,
        "influencer": "Mega", "model_name": "GradientBoosting"
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
