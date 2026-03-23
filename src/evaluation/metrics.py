import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.utils.logger import get_logger

logger = get_logger()
OUTPUT_DIR = Path("outputs/figures/")

def compute_test_metrics(pipeline, X_test, y_test) -> dict:
    # Calcule les metriques sur le jeu de test
    y_pred = pipeline.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
    }

def build_comparison_table(cv_results: dict, test_results: dict) -> pd.DataFrame:
    # Construit un tableau comparatif CV + test pour tous les modeles
    rows = []
    for name in cv_results:
        cv = cv_results[name]
        test = test_results.get(name, {})
        rows.append({
            "Modele": name,
            "CV R2": f"{cv['cv_r2_mean']:.4f} +/- {cv['cv_r2_std']:.4f}",
            "CV RMSE": f"{cv['cv_rmse_mean']:.4f}",
            "CV MAE": f"{cv['cv_mae_mean']:.4f}",
            "Test R2": f"{test.get('r2', 0):.4f}",
            "Test RMSE": f"{test.get('rmse', 0):.4f}",
            "Test MAE": f"{test.get('mae', 0):.4f}",
            "Train R2": f"{cv['train_r2_mean']:.4f}",
        })
    return pd.DataFrame(rows).set_index("Modele")

def plot_model_comparison(cv_results: dict) -> None:
    # Graphique comparatif R2 et RMSE pour tous les modeles
    names = list(cv_results.keys())
    r2_means = [cv_results[n]["cv_r2_mean"] for n in names]
    r2_stds = [cv_results[n]["cv_r2_std"] for n in names]
    rmse_means = [cv_results[n]["cv_rmse_mean"] for n in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison des modeles (Cross-Validation 5-fold)", fontsize=14, fontweight="bold")
    
    colors = ["#1565C0", "#1976D2", "#42A5F5", "#90CAF9"]
    
    # R2
    bars = axes[0].bar(names, r2_means, yerr=r2_stds, color=colors,
                       capsize=5, error_kw={"linewidth": 2})
    axes[0].set_title("R2 Score (plus haut = meilleur)")
    axes[0].set_ylabel("R2")
    axes[0].set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    
    # RMSE
    bars2 = axes[1].bar(names, rmse_means, color=colors)
    axes[1].set_title("RMSE (plus bas = meilleur)")
    axes[1].set_ylabel("RMSE (M$)")
    for bar, val in zip(bars2, rmse_means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    path = OUTPUT_DIR / "07_model_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")

def plot_predictions_vs_actual(trained_pipelines: dict, X_test, y_test) -> None:
    # Predictions vs valeurs reelles pour chaque modele
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Predictions vs Valeurs Reelles", fontsize=14, fontweight="bold")
    
    for ax, (name, pipeline) in zip(axes.flatten(), trained_pipelines.items()):
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.3, color="#1565C0", s=15)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Prediction parfaite")
        ax.set_xlabel("Valeurs reelles (M$)")
        ax.set_ylabel("Predictions (M$)")
        ax.set_title(f"{name} (R2={r2:.4f})")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = OUTPUT_DIR / "08_predictions_vs_actual.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")
