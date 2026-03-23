import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger()
OUTPUT_DIR = Path("outputs/figures/")

FEATURE_NAMES = [
    "TV", "Radio", "Social Media",
    "total_budget", "tv_share", "radio_share", "social_share",
    "tv_social_interaction", "tv_radio_interaction",
    "Influencer"
]

def get_shap_explainer(model_name: str, pipeline, X_train_transformed: np.ndarray):
    # Selectionne l'explainer SHAP adapte au type de modele
    model = pipeline.named_steps["model"]
    
    if model_name == "LinearRegression":
        # LinearExplainer : exact et rapide pour les modeles lineaires
        explainer = shap.LinearExplainer(model, X_train_transformed)
    elif model_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
        # TreeExplainer : exact et efficace pour les modeles a base d'arbres
        explainer = shap.TreeExplainer(model)
    else:
        # KernelExplainer : universel mais plus lent (fallback)
        background = shap.sample(X_train_transformed, 100)
        explainer = shap.KernelExplainer(model.predict, background)
    
    return explainer

def compute_shap_values(explainer, X_transformed: np.ndarray) -> np.ndarray:
    # Calcule les SHAP values sur un echantillon representatif
    shap_values = explainer.shap_values(X_transformed)
    # Certains explainers retournent une liste (classification) - on prend le premier
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return shap_values

def plot_shap_summary(shap_values: np.ndarray, X_transformed: np.ndarray,
                      model_name: str) -> None:
    # Beeswarm plot SHAP : importance + direction d'effet de chaque feature
    X_df = pd.DataFrame(X_transformed, columns=FEATURE_NAMES)
    
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_df, show=False, plot_size=None)
    plt.title(f"SHAP Summary Plot - {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = OUTPUT_DIR / f"10_shap_summary_{model_name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")

def plot_shap_bar(shap_values: np.ndarray, model_name: str) -> None:
    # Bar chart : importance moyenne absolue par feature (vision business)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1565C0" if i >= len(importance_df) - 3 else "#90CAF9"
              for i in range(len(importance_df))]
    ax.barh(importance_df["feature"], importance_df["importance"], color=colors)
    ax.set_title(f"SHAP Feature Importance - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance SHAP moyenne (|valeur|)")
    plt.tight_layout()
    path = OUTPUT_DIR / f"11_shap_bar_{model_name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")

def plot_permutation_importance(pipeline, X_test: pd.DataFrame,
                                y_test: pd.Series, model_name: str) -> None:
    # Permutation importance : mesure la degradation des performances
    # quand on permute aleatoirement les valeurs d'une feature
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10, random_state=42, scoring="r2", n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=True)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(perm_df["feature"], perm_df["importance_mean"],
            xerr=perm_df["importance_std"], color="#1976D2",
            error_kw={"linewidth": 1.5, "ecolor": "gray"})
    ax.set_title(f"Permutation Importance - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Diminution du R2 (moyenne sur 10 repetitions)")
    plt.tight_layout()
    path = OUTPUT_DIR / f"12_permutation_importance_{model_name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")

def plot_shap_dependence(shap_values: np.ndarray, X_transformed: np.ndarray,
                         feature_name: str, model_name: str) -> None:
    # Dependence plot : relation entre une feature et ses SHAP values
    # Revele les effets non-lineaires et interactions
    if feature_name not in FEATURE_NAMES:
        return
    
    feat_idx = FEATURE_NAMES.index(feature_name)
    X_df = pd.DataFrame(X_transformed, columns=FEATURE_NAMES)
    
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx, shap_values, X_df,
        show=False, ax=plt.gca()
    )
    plt.title(f"SHAP Dependence - {feature_name} ({model_name})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / f"13_shap_dependence_{feature_name}_{model_name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")

def run_interpretability(model_name: str = "GradientBoosting") -> None:
    # Execute l'analyse complete d'interpretabilite sur le meilleur modele
    logger.info(f"Interpretabilite sur le modele : {model_name}")
    
    from src.data.loader import load_raw_data
    from src.features.preprocessing import prepare_data, build_preprocessor
    from src.utils.config_loader import load_config
    
    config = load_config()
    df = load_raw_data()
    X, y, _, numeric_feats, cat_feats = prepare_data(df, config)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Chargement du pipeline entraine
    pipeline = joblib.load(f"outputs/models/{model_name}.joblib")
    
    # Transformation des donnees via le preprocesseur du pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    
    # SHAP values
    explainer = get_shap_explainer(model_name, pipeline, X_train_t)
    shap_values = compute_shap_values(explainer, X_test_t)
    
    # Figures SHAP
    plot_shap_summary(shap_values, X_test_t, model_name)
    plot_shap_bar(shap_values, model_name)
    plot_shap_dependence(shap_values, X_test_t, "TV", model_name)
    plot_shap_dependence(shap_values, X_test_t, "total_budget", model_name)
    
    # Permutation importance (sur le pipeline complet, donnees non transformees)
    plot_permutation_importance(pipeline, X_test, y_test, model_name)
    
    # Rapport textuel business
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "shap_importance": mean_abs_shap
    }).sort_values("shap_importance", ascending=False)
    
    print("\n" + "="*55)
    print(f"IMPORTANCE DES VARIABLES (SHAP) - {model_name}")
    print("="*55)
    for _, row in importance_df.iterrows():
        bar = "#" * int(row["shap_importance"] / mean_abs_shap.max() * 30)
        print(f"  {row['feature']:28s} : {row['shap_importance']:7.3f}  {bar}")
    print("="*55)
    
    # Sauvegarde du rapport
    importance_df.to_csv(f"outputs/reports/shap_importance_{model_name}.csv", index=False)
    logger.info("Rapport SHAP sauvegarde")
