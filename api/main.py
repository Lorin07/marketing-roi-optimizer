import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from src.utils.logger import get_logger

logger = get_logger()

# Modele charge au demarrage de l'application (singleton)
MODEL_STORE = {}
DEFAULT_MODEL = "GradientBoosting"

FEATURE_COLS = [
    "TV", "Radio", "Social Media",
    "total_budget", "tv_share", "radio_share", "social_share",
    "tv_social_interaction", "tv_radio_interaction",
    "Influencer"
]

# Modeles disponibles et leurs metriques test
MODEL_METRICS = {
    "LinearRegression": {"test_r2": 0.9956, "test_rmse": 6.16},
    "RandomForest":     {"test_r2": 0.9986, "test_rmse": 3.47},
    "GradientBoosting": {"test_r2": 0.9989, "test_rmse": 3.12},
    "XGBoost":          {"test_r2": 0.9988, "test_rmse": 3.20},
    "MLP":              {"test_r2": 0.9962, "test_rmse": 5.73},
}

def create_features_from_input(tv: float, radio: float, social_media: float,
                                influencer: str) -> pd.DataFrame:
    # Reproduit le feature engineering du pipeline de preprocessing
    total_budget = tv + radio + social_media
    eps = 1e-8
    row = {
        "TV": tv,
        "Radio": radio,
        "Social Media": social_media,
        "total_budget": total_budget,
        "tv_share": tv / (total_budget + eps),
        "radio_share": radio / (total_budget + eps),
        "social_share": social_media / (total_budget + eps),
        "tv_social_interaction": tv * social_media,
        "tv_radio_interaction": tv * radio,
        "Influencer": influencer,
    }
    return pd.DataFrame([row])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement des modeles au demarrage
    logger.info("Chargement des modeles...")
    models_dir = Path("outputs/models/")
    for name in MODEL_METRICS:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            MODEL_STORE[name] = joblib.load(path)
            logger.info(f"  Modele charge : {name}")
        else:
            logger.warning(f"  Modele introuvable : {name}")
    
    # Cas special MLP : charge le bundle preprocesseur + modele
    mlp_path = models_dir / "MLP_bundle.joblib"
    if mlp_path.exists():
        MODEL_STORE["MLP"] = joblib.load(mlp_path)
        logger.info("  Modele charge : MLP (bundle)")
    
    logger.info(f"API prete - {len(MODEL_STORE)} modeles charges")
    yield
    MODEL_STORE.clear()

# Application FastAPI
app = FastAPI(
    title="Marketing ROI Optimizer API",
    description="API de prediction des ventes et optimisation du ROI marketing",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas de validation Pydantic
class PredictRequest(BaseModel):
    tv: float = Field(..., ge=0, le=500, description="Budget TV en millions $")
    radio: float = Field(..., ge=0, le=200, description="Budget Radio en millions $")
    social_media: float = Field(..., ge=0, le=100, description="Budget Social Media en millions $")
    influencer: str = Field(..., description="Type d'influenceur : Mega, Micro, Nano, Macro")
    model_name: str = Field(default="GradientBoosting", description="Nom du modele a utiliser")

    @field_validator("influencer")
    @classmethod
    def validate_influencer(cls, v):
        allowed = ["Mega", "Micro", "Nano", "Macro"]
        if v not in allowed:
            raise ValueError(f"influencer doit etre parmi : {allowed}")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v):
        allowed = list(MODEL_METRICS.keys())
        if v not in allowed:
            raise ValueError(f"model_name doit etre parmi : {allowed}")
        return v

class PredictResponse(BaseModel):
    predicted_sales: float
    model_used: str
    model_r2: float
    inputs: dict
    roi_estimate: float

# Endpoints
@app.get("/health", tags=["Monitoring"])
def health_check():
    # Verifie que l'API est operationnelle et les modeles charges
    return {
        "status": "ok",
        "models_loaded": list(MODEL_STORE.keys()),
        "default_model": DEFAULT_MODEL,
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    # Prediction des ventes pour un budget marketing donne
    model_name = request.model_name
    
    if model_name not in MODEL_STORE:
        raise HTTPException(status_code=404, detail=f"Modele '{model_name}' non charge")
    
    # Construction du DataFrame avec feature engineering
    X = create_features_from_input(
        request.tv, request.radio, request.social_media, request.influencer
    )
    
    try:
        model = MODEL_STORE[model_name]
        
        if model_name == "MLP":
            # MLP bundle : preprocesseur + modele separes
            X_t = model["preprocessor"].transform(X)
            prediction = float(model["mlp"].predict(X_t)[0])
        else:
            prediction = float(model.predict(X)[0])
        
        # Calcul ROI estime : (ventes - budget) / budget * 100
        total_budget = request.tv + request.radio + request.social_media
        roi = ((prediction - total_budget) / (total_budget + 1e-8)) * 100
        
        logger.info(f"Prediction : {prediction:.2f} M$ | ROI : {roi:.1f}% | Modele : {model_name}")
        
        return PredictResponse(
            predicted_sales=round(prediction, 2),
            model_used=model_name,
            model_r2=MODEL_METRICS[model_name]["test_r2"],
            inputs={"tv": request.tv, "radio": request.radio,
                    "social_media": request.social_media, "influencer": request.influencer},
            roi_estimate=round(roi, 2),
        )
    
    except Exception as e:
        logger.error(f"Erreur prediction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

@app.get("/model-info", tags=["Modeles"])
def model_info():
    # Retourne les informations sur tous les modeles disponibles
    return {
        "available_models": [
            {
                "name": name,
                "loaded": name in MODEL_STORE,
                "test_r2": metrics["test_r2"],
                "test_rmse": metrics["test_rmse"],
                "recommended": name == DEFAULT_MODEL,
            }
            for name, metrics in MODEL_METRICS.items()
        ],
        "best_model": DEFAULT_MODEL,
        "features_used": FEATURE_COLS,
    }

@app.get("/models/{model_name}/predict-example", tags=["Modeles"])
def predict_example(model_name: str):
    # Retourne un exemple de prediction pour tester l'API
    example = PredictRequest(
        tv=50.0, radio=15.0, social_media=3.0,
        influencer="Mega", model_name=model_name
    )
    return predict(example)
