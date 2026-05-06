#!/bin/bash

# Lancement automatique du projet Marketing ROI Optimizer
# Usage : bash launch.sh

set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"
source venv/bin/activate

echo ""
echo "=================================================="
echo "   MARKETING ROI OPTIMIZER - Demarrage"
echo "=================================================="

# Arreter les processus existants proprement
echo "[1/5] Nettoyage des processus existants..."
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "streamlit run dashboard" 2>/dev/null || true
sleep 1

# Verifier que les modeles existent
echo "[2/5] Verification des modeles..."
if [ ! -f "outputs/models/GradientBoosting.joblib" ]; then
    echo "     Modeles absents - lancement du pipeline d'entrainement..."
    python scripts/run_preprocessing.py
    python scripts/run_training.py
    python scripts/run_deep_learning.py
    python scripts/run_interpretability.py
else
    echo "     Modeles detectes - OK"
fi

# Lancer l'API en arriere-plan
echo "[3/5] Demarrage de l'API FastAPI (port 8000)..."
nohup uvicorn api.main:app --port 8000 --host 0.0.0.0 > logs/api.log 2>&1 &
API_PID=$!
echo "     PID API : $API_PID"

# Attendre que l'API soit prete
echo "[4/5] Attente disponibilite API..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "     API prete !"
        break
    fi
    sleep 1
    echo "     Tentative $i/30..."
done

# Lancer le Dashboard
echo "[5/5] Demarrage du Dashboard Streamlit (port 8501)..."
nohup streamlit run dashboard/app.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "     PID Dashboard : $DASH_PID"

sleep 3

echo ""
echo "=================================================="
echo "   PROJET DEMARRE AVEC SUCCES"
echo "=================================================="
echo ""
echo "   Dashboard  : http://localhost:8501"
echo "   API REST   : http://localhost:8000"
echo "   Swagger UI : http://localhost:8000/docs"
echo ""
echo "   Pour arreter : bash stop.sh"
echo "   Logs API      : tail -f logs/api.log"
echo "   Logs Dashboard: tail -f logs/dashboard.log"
echo "=================================================="
echo ""

# Ouvrir le dashboard dans le navigateur automatiquement
sleep 2
open http://localhost:8501
