# Marketing ROI Optimizer

Systeme intelligent multi-modeles d'optimisation du ROI marketing.

## Stack technique
- Python 3.11+
- scikit-learn, XGBoost, LightGBM, PyTorch
- FastAPI (API REST)
- Streamlit (Dashboard)
- SHAP (Interpretabilite)

## Dataset
Dummy Marketing and Sales Data (Kaggle) - 4572 lignes, 5 colonnes.

## Structure du projet
```
projdata/
├── data/          # Donnees brutes et traitees
├── src/           # Code source (data, features, models, evaluation)
├── api/           # API FastAPI
├── dashboard/     # Dashboard Streamlit
├── config/        # Fichiers de configuration
├── outputs/       # Modeles, figures, rapports
├── tests/         # Tests unitaires
└── docs/          # Documentation
```

## Lancement
```bash
source venv/bin/activate
# API
uvicorn api.main:app --reload
# Dashboard
streamlit run dashboard/app.py
```
