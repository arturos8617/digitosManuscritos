# MLP de Dígitos — Cliente/Servidor

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Entrenamiento
python -m src.mlp.train --epochs 10 --hidden 128 --lr 0.1 --l2 0.0001

## Evaluación
python -m src.mlp.eval --weights weights.npz

## Servidor (API)
uvicorn src.server.server:app --reload --port 8000

## Cliente
Abre src/client/index.html en tu navegador (sirve con un HTTP server si tu navegador bloquea archivos locales).
