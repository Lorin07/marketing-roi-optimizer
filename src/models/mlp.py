import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger()

# Verification disponibilite GPU/MPS (Apple Silicon ou CPU)
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info(f"Device PyTorch : {DEVICE}")


class MLPRegressor(nn.Module):
    # Architecture MLP avec BatchNorm et Dropout pour regularisation
    def __init__(self, input_dim: int, hidden_layers: list, dropout_rate: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie : 1 neurone (regression)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class SklearnMLPWrapper(BaseEstimator, RegressorMixin):
    # Wrapper sklearn pour integrer le MLP PyTorch dans un Pipeline sklearn
    def __init__(self, input_dim: int = 10, hidden_layers: list = None,
                 epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 0.001, dropout_rate: float = 0.2,
                 patience: int = 15, random_state: int = 42):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.train_losses_ = []
        self.val_losses_ = []
    
    def _to_tensor(self, X, y=None):
        X_t = torch.FloatTensor(np.array(X)).to(DEVICE)
        if y is not None:
            y_t = torch.FloatTensor(np.array(y)).to(DEVICE)
            return X_t, y_t
        return X_t
    
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Split interne train/validation pour early stopping (10%)
        n_val = max(1, int(len(X) * 0.1))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]
        
        X_tr_t, y_tr_t = self._to_tensor(X_tr, y_tr)
        X_val_t, y_val_t = self._to_tensor(X_val, y_val)
        
        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialisation du modele
        self.model_ = MLPRegressor(
            self.input_dim, self.hidden_layers, self.dropout_rate
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # Reduce LR sur plateau de validation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=7, factor=0.5, verbose=False
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        
        for epoch in range(self.epochs):
            # Phase entrainement
            self.model_.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model_(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Phase validation
            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            self.train_losses_.append(epoch_loss / len(loader))
            self.val_losses_.append(val_loss)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping a l'epoch {epoch+1} (patience={self.patience})")
                    break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1:3d} | Train Loss: {self.train_losses_[-1]:.4f} | Val Loss: {val_loss:.4f}")
        
        # Restauration des meilleurs poids
        if best_weights:
            self.model_.load_state_dict(best_weights)
        
        return self
    
    def predict(self, X):
        self.model_.eval()
        X_t = self._to_tensor(X)
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()
        return preds
