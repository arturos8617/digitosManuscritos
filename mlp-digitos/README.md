# MLP de Dígitos — Cliente/Servidor

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Entrenamiento
python -m src.mlp.train --epochs 15 --hidden 256 --lr 0.1

Esto generará el archivo:
weights.npz

## Generar plantillas de referencia
python - <<'PY'
from src.mlp.templates import build_templates
build_templates("templates.npz")
print("Plantillas generadas")
PY

Esto generará:
templates.npz

## Evaluación
python -m src.mlp.eval --weights weights.npz

## Ejecutar el backend
uvicorn src.server.server:app --reload --port 8000

Verificación:
API activa en: http://localhost:8000
Documentación automática: http://localhost:8000/docs

## Ejecutar el frontend
cd frontend
npm install
npm run dev

Luego abre en el navegador:
http://localhost:5173

## Uso del sistema
El sistema mostrará un dígito objetivo (por ejemplo: “Escribe el número 7”).

El usuario dibuja el dígito en el canvas.

Presiona Predecir.

El sistema muestra:

dígito detectado

nivel de confianza

similitud con la plantilla

retroalimentación básica

Puede presionar Nuevo número para continuar practicand