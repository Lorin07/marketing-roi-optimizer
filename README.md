# Marketing ROI Optimizer

Systeme intelligent multi-modeles de prediction des ventes et optimisation du ROI marketing.
Projet de fin de Mastere Data Engineering & AI - EFREI Paris 2026.

## Resultats des modeles

| Modele           | CV R2  | Test R2 | Test RMSE |
|------------------|--------|---------|-----------|
| GradientBoosting | 0.9972 | 0.9989  | 3.12      |
| XGBoost          | 0.9967 | 0.9988  | 3.20      |
| RandomForest     | 0.9972 | 0.9986  | 3.47      |
| MLP PyTorch      | -      | 0.9962  | 5.73      |
| LinearRegression | 0.9940 | 0.9956  | 6.16      |

## Stack technique

- Python 3.10 | scikit-learn 1.4 | XGBoost 2.0 | LightGBM 4.3
- PyTorch 2.2 (MPS Apple Silicon)
- FastAPI 0.111 | Uvicorn
- Streamlit 1.35
- SHAP 0.44

## Structure
```
projdata/
├── data/raw/          # Dataset brut (Kaggle)
├── data/processed/    # Features engineerees
├── src/
│   ├── data/          # Loader + EDA
│   ├── features/      # Preprocessing pipeline
│   ├── models/        # Trainer + MLP PyTorch
│   └── evaluation/    # Metriques + SHAP
├── api/               # FastAPI REST
├── dashboard/         # Streamlit
├── outputs/
│   ├── models/        # Modeles joblib
│   ├── figures/       # 13 graphiques
│   └── reports/       # CSV comparaison
├── scripts/           # Scripts d'execution
├── tests/             # Tests unitaires pytest
└── config/            # config.yaml
```

## Lancement rapide
```bash
# Environnement
source venv/bin/activate

# Preprocessing
python scripts/run_preprocessing.py

# EDA
python scripts/run_eda.py

# Entrainement
python scripts/run_training.py

# Deep Learning
python scripts/run_deep_learning.py

# Interpretabilite
python scripts/run_interpretability.py

# API (Terminal 1)
uvicorn api.main:app --reload --port 8000

# Dashboard (Terminal 2)
streamlit run dashboard/app.py --server.port 8501

# Tests
pytest tests/ -v
```

## Insights business (SHAP)

- **TV = 97% de l'importance** - Levier dominant sur les ventes
- **Radio = 2%** - Second levier significatif
- **Influencer = negligeable** - Type d'influenceur sans impact mesurable
