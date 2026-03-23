import sys
from pathlib import Path

# Ajouter la racine du projet au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data, audit_data, print_audit_report
from src.features.preprocessing import prepare_data, save_processed_data
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger()

def main():
    logger.info("Demarrage du pipeline de preprocessing")
    
    # Chargement configuration
    config = load_config()
    
    # Chargement donnees brutes
    df = load_raw_data()
    
    # Audit qualite
    audit = audit_data(df)
    print_audit_report(audit)
    
    # Preparation des donnees
    X, y, preprocessor, numeric_feats, cat_feats = prepare_data(df, config)
    
    # Test du preprocesseur (fit sur tout pour validation)
    X_transformed = preprocessor.fit_transform(X)
    logger.info(f"Preprocessing OK - Shape transformee : {X_transformed.shape}")
    
    # Sauvegarde
    save_processed_data(X, y)
    
    # Resume final
    print("\n" + "="*60)
    print("RESUME PREPROCESSING")
    print("="*60)
    print(f"Lignes finales     : {len(X)}")
    print(f"Features totales   : {X_transformed.shape[1]}")
    print(f"  - Numeriques     : {len(numeric_feats)}")
    print(f"  - Categorielles  : {len(cat_feats)}")
    print(f"Target (Sales)     : min={y.min():.1f} | max={y.max():.1f} | mean={y.mean():.1f}")
    print("="*60)
    logger.info("Preprocessing termine avec succes")

if __name__ == "__main__":
    main()
