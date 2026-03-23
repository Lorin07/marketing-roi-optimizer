import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from src.data.loader import load_raw_data
from src.features.preprocessing import prepare_data, build_preprocessor
from src.models.mlp import SklearnMLPWrapper
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger()
OUTPUT_DIR = Path("outputs/figures/")

def main():
    config = load_config()
    
    # Chargement et preparation
    df = load_raw_data()
    X, y, _, numeric_feats, cat_feats = prepare_data(df, config)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    # Pipeline : preprocesseur + MLP
    preprocessor = build_preprocessor(numeric_feats, cat_feats)
    
    # Fit du preprocesseur sur train pour obtenir input_dim
    X_train_transformed = preprocessor.fit_transform(X_train)
    input_dim = X_train_transformed.shape[1]
    logger.info(f"Input dimension MLP : {input_dim}")
    
    # Instanciation du MLP
    mlp = SklearnMLPWrapper(
        input_dim=input_dim,
        hidden_layers=[128, 64, 32],
        epochs=200,
        batch_size=64,
        learning_rate=0.001,
        dropout_rate=0.2,
        patience=20,
        random_state=config["data"]["random_state"],
    )
    
    # Entrainement sur donnees preprocessees
    logger.info("Entrainement MLP PyTorch...")
    mlp.fit(X_train_transformed, y_train.values)
    
    # Evaluation sur test
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = mlp.predict(X_test_transformed)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*50)
    print("RESULTATS MLP PYTORCH")
    print("="*50)
    print(f"R2   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"Epochs entraines : {len(mlp.train_losses_)}")
    print("="*50)
    
    # Courbe de perte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MLP PyTorch - Courbes d'entrainement", fontsize=14, fontweight="bold")
    
    axes[0].plot(mlp.train_losses_, label="Train Loss", color="#1565C0")
    axes[0].plot(mlp.val_losses_, label="Val Loss", color="#EF5350")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Courbe de perte")
    axes[0].legend()
    
    # Predictions vs Reelles
    axes[1].scatter(y_test, y_pred, alpha=0.3, color="#1565C0", s=15)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=2)
    axes[1].set_xlabel("Valeurs reelles (M$)")
    axes[1].set_ylabel("Predictions (M$)")
    axes[1].set_title(f"MLP Predictions vs Reelles (R2={r2:.4f})")
    
    plt.tight_layout()
    path = OUTPUT_DIR / "09_mlp_training.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")
    
    # Sauvegarde du wrapper MLP + preprocesseur ensemble
    mlp_bundle = {"preprocessor": preprocessor, "mlp": mlp}
    joblib.dump(mlp_bundle, "outputs/models/MLP_bundle.joblib")
    logger.info("MLP bundle sauvegarde : outputs/models/MLP_bundle.joblib")

if __name__ == "__main__":
    main()
