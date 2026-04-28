# MLP de Dígitos — Cliente/Servidor

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Entrenamiento
python -m src.mlp.train --epochs 15 --hidden 256 --lr 0.1

Esto generará el archivo:
weights.npz

## Generar plantillas de referencia
python - <<'PY2'
from src.mlp.templates import build_templates
build_templates("templates.npz")
print("Plantillas generadas")
PY2

Esto generará:
templates.npz

## Recolectar muestras reales del canvas
1. Ejecuta backend y frontend.
2. Selecciona el modo de ejercicio (números / vocales minúsculas / vocales mayúsculas).
3. Dibuja el símbolo objetivo en la app React.
3. Si el trazo representa bien al objetivo, presiona **Guardar muestra**.
4. Las muestras se guardan en:
   - `canvas_samples/<digito>/*.png` para modo números (compatibilidad con entrenamiento actual).
   - `canvas_samples/<modo>/<etiqueta>/*.png` para vocales.

## Reentrenar la misma MLP con muestras reales
python -m src.mlp.train --epochs 15 --hidden 256 --lr 0.1 --canvas-dir canvas_samples --canvas-repeat 3

- `--canvas-dir`: directorio con muestras reales por carpeta (`0` a `9`).
- `--canvas-repeat`: cuánto peso tendrán esas muestras al mezclarlas con MNIST.

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
El sistema mostrará un símbolo objetivo según el modo de práctica (por ejemplo: “Escribe el número 7” o “Escribe la vocal a”).

El usuario dibuja el dígito en el canvas.

Presiona Predecir.

El sistema muestra:

dígito detectado

nivel de confianza

similitud con la plantilla

retroalimentación básica

Puede presionar Nuevo número para continuar practicando.

## Documentación
- [01 Alcance](docs/01_alcance.md)
- [02 Modelo matemático](docs/02_modelo_matematico.md)
- [03 Arquitectura](docs/03_arquitectura.md)
- [04 UMLs](docs/04_umls.md)
- [05 Resultados](docs/05_resultados.md)
- [06 Entrenamiento](docs/06_entrenamiento.md)
- [07 Compatibilidad](docs/07_compatibilidad.md)
- [08 UX](docs/08_ux.md)
- [10 Mantenimiento y soporte](docs/10_mantenimiento_soporte.md)
