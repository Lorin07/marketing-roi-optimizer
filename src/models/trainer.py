import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger()

def get_models(config: dict) -> dict:
    # Definit les 4 modeles avec leurs hyperparametres depuis la config
    rs = config["models"]["random_state"]
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=config["models"]["random_forest"]["n_estimators"],
            max_depth=config["models"]["random_forest"]["max_depth"],
            random_state=rs,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=config["models"]["gradient_boosting"]["n_estimators"],
            learning_rate=config["models"]["gradient_boosting"]["learning_rate"],
            max_depth=config["models"]["gradient_boosting"]["max_depth"],
            random_state=rs,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=rs,
            verbosity=0,
            n_jobs=-1,
        ),
    }

def build_full_pipeline(preprocessor, model) -> Pipeline:
    # Combine preprocesseur et modele en un pipeline sklearn sans leakage
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                   cv_folds: int = 5) -> dict:
    # Cross-validation avec metriques R2, RMSE, MAE
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = cross_validate(
        pipeline, X, y,
        cv=kf,
        scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        return_train_score=True,
        n_jobs=-1,
    )
    
    return {
        "cv_r2_mean": cv_results["test_r2"].mean(),
        "cv_r2_std": cv_results["test_r2"].std(),
        "cv_rmse_mean": -cv_results["test_neg_root_mean_squared_error"].mean(),
        "cv_rmse_std": cv_results["test_neg_root_mean_squared_error"].std(),
        "cv_mae_mean": -cv_results["test_neg_mean_absolute_error"].mean(),
        "cv_mae_std": cv_results["test_neg_mean_absolute_error"].std(),
        "train_r2_mean": cv_results["train_r2"].mean(),
    }

def train_all_models(X: pd.DataFrame, y: pd.Series,
                     preprocessor, config: dict) -> tuple:
    # Entraine tous les modeles et retourne resultats + pipelines complets
    models = get_models(config)
    cv_folds = config["data"]["cv_folds"]
    results = {}
    trained_pipelines = {}
    
    logger.info(f"Entrainement de {len(models)} modeles avec CV {cv_folds}-fold...")
    
    for name, model in models.items():
        logger.info(f"  Modele : {name}")
        pipeline = build_full_pipeline(preprocessor, model)
        
        # Cross-validation
        metrics = evaluate_model(pipeline, X, y, cv_folds)
        results[name] = metrics
        
        # Entrainement final sur toutes les donnees
        pipeline.fit(X, y)
        trained_pipelines[name] = pipeline
        
        logger.info(
            f"    R2={metrics['cv_r2_mean']:.4f} (+/-{metrics['cv_r2_std']:.4f}) | "
            f"RMSE={metrics['cv_rmse_mean']:.4f} | MAE={metrics['cv_mae_mean']:.4f}"
        )
    
    return results, trained_pipelines

def save_models(trained_pipelines: dict, output_dir: str = "outputs/models/") -> None:
    # Sauvegarde chaque pipeline entraine avec joblib
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, pipeline in trained_pipelines.items():
        path = f"{output_dir}{name}.joblib"
        joblib.dump(pipeline, path)
        logger.info(f"Modele sauvegarde : {path}")

def load_model(name: str, output_dir: str = "outputs/models/") -> Pipeline:
    # Charge un pipeline depuis le disque
    path = f"{output_dir}{name}.joblib"
    return joblib.load(path)
