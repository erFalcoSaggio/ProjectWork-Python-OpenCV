#!/usr/bin/env bash
# Avvia Falcari Cam. Al primo lancio crea un virtualenv e installa le
# dipendenze; nei lanci successivi riusa quello esistente.
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Setup iniziale: creo il virtualenv..."
    python3 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "Installo le dipendenze..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

python3 main.py
