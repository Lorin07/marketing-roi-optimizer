import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from src.data.loader import load_raw_data
from src.features.preprocessing import prepare_data, build_preprocessor
from src.models.trainer import train_all_models, save_models
from src.evaluation.metrics import (compute_test_metrics, build_comparison_table,
                                     plot_model_comparison, plot_predictions_vs_actual)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger()

def main():
    config = load_config()
    
    # Chargement et preparation des donnees
    df = load_raw_data()
    X, y, preprocessor, numeric_feats, cat_feats = prepare_data(df, config)
    
    # Split train/test (20%) - seed fixe pour reproducibilite
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    logger.info(f"Train : {len(X_train)} | Test : {len(X_test)}")
    
    # Reconstruction du preprocesseur (fit uniquement sur train via CV)
    from src.features.preprocessing import build_preprocessor
    preprocessor = build_preprocessor(numeric_feats, cat_feats)
    
    # Entrainement + cross-validation
    cv_results, trained_pipelines = train_all_models(X_train, y_train, preprocessor, config)
    
    # Evaluation sur le jeu de test
    test_results = {}
    for name, pipeline in trained_pipelines.items():
        test_results[name] = compute_test_metrics(pipeline, X_test, y_test)
    
    # Tableau comparatif
    comparison = build_comparison_table(cv_results, test_results)
    print("\n" + "="*80)
    print("TABLEAU COMPARATIF DES MODELES")
    print("="*80)
    print(comparison.to_string())
    print("="*80)
    
    # Sauvegarde du tableau
    Path("outputs/reports/").mkdir(parents=True, exist_ok=True)
    comparison.to_csv("outputs/reports/model_comparison.csv")
    
    # Figures
    plot_model_comparison(cv_results)
    plot_predictions_vs_actual(trained_pipelines, X_test, y_test)
    
    # Sauvegarde des modeles
    save_models(trained_pipelines)
    
    # Meilleur modele
    best = max(cv_results, key=lambda n: cv_results[n]["cv_r2_mean"])
    logger.info(f"Meilleur modele (CV R2) : {best} = {cv_results[best]['cv_r2_mean']:.4f}")

if __name__ == "__main__":
    main()
