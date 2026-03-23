import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger()

def load_raw_data(path: str = None) -> pd.DataFrame:
    # Charge le CSV brut depuis le chemin configure
    config = load_config()
    data_path = path or config["paths"]["data_raw"]
    logger.info(f"Chargement des donnees : {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def audit_data(df: pd.DataFrame) -> dict:
    # Genere un rapport d'audit complet : types, manquants, doublons, statistiques
    logger.info("Audit qualite des donnees en cours...")
    
    audit = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_count": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_stats": df.describe().to_dict(),
        "categorical_cols": df.select_dtypes(include="object").columns.tolist(),
        "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
    }
    
    # Affichage structure de l'audit
    logger.info(f"Shape : {audit['shape']}")
    logger.info(f"Doublons : {audit['duplicates']}")
    logger.info(f"Colonnes numeriques : {audit['numeric_cols']}")
    logger.info(f"Colonnes categorielle : {audit['categorical_cols']}")
    
    for col, pct in audit["missing_pct"].items():
        if pct > 0:
            logger.warning(f"Valeurs manquantes - {col} : {pct}% ({audit['missing_count'][col]} lignes)")
    
    # Modalites de la variable Influencer
    if "Influencer" in df.columns:
        modalities = df["Influencer"].value_counts()
        logger.info(f"Modalites Influencer :\n{modalities}")
        audit["influencer_modalities"] = modalities.to_dict()
    
    return audit

def print_audit_report(audit: dict) -> None:
    # Affiche le rapport d'audit de maniere lisible
    print("\n" + "="*60)
    print("RAPPORT AUDIT QUALITE DES DONNEES")
    print("="*60)
    print(f"Dimensions        : {audit['shape'][0]} lignes x {audit['shape'][1]} colonnes")
    print(f"Doublons          : {audit['duplicates']}")
    print(f"Colonnes num.     : {audit['numeric_cols']}")
    print(f"Colonnes cat.     : {audit['categorical_cols']}")
    print("\nValeurs manquantes :")
    for col, pct in audit["missing_pct"].items():
        status = "OK" if pct == 0 else f"ATTENTION : {pct}% ({audit['missing_count'][col]} lignes)"
        print(f"  {col:20s} : {status}")
    if "influencer_modalities" in audit:
        print("\nModalites Influencer :")
        for mod, count in audit["influencer_modalities"].items():
            print(f"  {mod:10s} : {count}")
    print("="*60 + "\n")
