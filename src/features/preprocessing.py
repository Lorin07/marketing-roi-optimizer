import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger()

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    # Pipeline numerique : imputation mediane + standardisation
    # Mediane plutot que moyenne car robuste aux outliers
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    # Pipeline categoriel : imputation mode + encodage ordinal
    # OrdinalEncoder suffisant ici car peu de modalites et modeles ensemblistes
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")
    
    return preprocessor

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering : creation de features metier (ROI proxies, ratios, interactions)
    df = df.copy()
    
    # Budget total investi tous canaux confondus
    df["total_budget"] = df["TV"] + df["Radio"] + df["Social Media"]
    
    # Part de chaque canal dans le budget total (mix marketing)
    df["tv_share"] = df["TV"] / (df["total_budget"] + 1e-8)
    df["radio_share"] = df["Radio"] / (df["total_budget"] + 1e-8)
    df["social_share"] = df["Social Media"] / (df["total_budget"] + 1e-8)
    
    # Interaction TV x Social Media (synergie cross-canal)
    df["tv_social_interaction"] = df["TV"] * df["Social Media"]
    
    # Interaction TV x Radio
    df["tv_radio_interaction"] = df["TV"] * df["Radio"]
    
    logger.info(f"Features creees : {['total_budget', 'tv_share', 'radio_share', 'social_share', 'tv_social_interaction', 'tv_radio_interaction']}")
    return df

def prepare_data(df: pd.DataFrame, config: dict) -> tuple:
    # Pipeline complet : feature engineering + separation X/y + preprocesseur
    target = config["data"]["target"]
    numeric_features = config["data"]["features_num"]
    categorical_features = config["data"]["features_cat"]
    
    # Supprimer les lignes ou la cible est manquante
    df_clean = df.dropna(subset=[target]).copy()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.warning(f"{n_dropped} lignes supprimees (cible manquante)")
    
    # Feature engineering
    df_enriched = create_features(df_clean)
    
    # Features enrichies (originales + nouvelles)
    all_numeric = numeric_features + ["total_budget", "tv_share", "radio_share", "social_share", "tv_social_interaction", "tv_radio_interaction"]
    
    X = df_enriched[all_numeric + categorical_features]
    y = df_enriched[target]
    
    logger.info(f"X shape : {X.shape} | y shape : {y.shape}")
    logger.info(f"Features numeriques ({len(all_numeric)}) : {all_numeric}")
    logger.info(f"Features categorielles : {categorical_features}")
    
    # Construction du preprocesseur
    preprocessor = build_preprocessor(all_numeric, categorical_features)
    
    return X, y, preprocessor, all_numeric, categorical_features

def save_processed_data(X: pd.DataFrame, y: pd.Series, output_dir: str = "data/processed/") -> None:
    # Sauvegarde les donnees en format CSV pour tracabilite
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    X.to_csv(f"{output_dir}X_features.csv", index=False)
    y.to_csv(f"{output_dir}y_target.csv", index=False)
    logger.info(f"Donnees sauvegardees dans {output_dir}")
