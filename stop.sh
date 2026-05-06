#!/bin/bash
echo "Arret des services..."
pkill -f "uvicorn api.main:app" 2>/dev/null && echo "API arretee" || echo "API deja arretee"
pkill -f "streamlit run dashboard" 2>/dev/null && echo "Dashboard arrete" || echo "Dashboard deja arrete"
echo "Tous les services sont arretes."
