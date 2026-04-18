# 06. Flujo real de entrenamiento (`src/mlp/train.py`)

## 1) Resumen del pipeline real
El script `src/mlp/train.py` implementa un entrenamiento supervisado de un MLP en cuatro fases:

1. **Parseo de argumentos CLI** (hiperparámetros y opciones de datos reales).
2. **Carga de datos** con `load_mnist()` y división en train/val/test.
3. **Bucle por épocas**:
   - entrenamiento por mini-batches,
   - validación con `val_acc` al final de cada época.
4. **Evaluación final en test**, guardado de pesos y salida por consola.

---

## 2) División exacta de datos (`src/mlp/data.py`)
La función `load_mnist()` realiza este flujo:

- Carga MNIST completo (OpenML o fallback local `mnist.npz`).
- Normaliza píxeles en \\( [0, 1] \\) dividiendo entre 255.
- Mezcla aleatoriamente con `rng.permutation(seed=42)`.
- Divide en proporciones fijas:
  - **train = 80%**
  - **val = 10%**
  - **test = 10%**

Con MNIST completo (70,000 muestras), esto queda en:

- **56,000** entrenamiento,
- **7,000** validación,
- **7,000** prueba.

> Importante: la división 80/10/10 ocurre **después** de la mezcla aleatoria, por lo que no se conserva el orden original del dataset.

---

## 3) Hiperparámetros de CLI y su impacto
El script define estos argumentos:

- `--epochs` (int, default `10`): número de pasadas completas sobre el set de entrenamiento.
  - Más épocas suelen mejorar la convergencia hasta cierto punto.
  - Demasiadas épocas pueden sobreajustar si no hay regularización.

- `--hidden` (int, default `128`): cantidad de neuronas en la capa oculta del MLP.
  - Mayor valor = más capacidad representacional.
  - También incrementa costo computacional y riesgo de sobreajuste.

- `--lr` (float, default `0.1`): tasa de aprendizaje en el paso de actualización.
  - Alta: converge más rápido, pero puede desestabilizar.
  - Baja: más estable, pero más lenta y puede quedar subentrenado con pocas épocas.

- `--l2` (float, default `0.0`): factor de regularización L2 en el modelo.
  - Penaliza pesos grandes para mejorar generalización.
  - Si es muy alto, puede reducir demasiado la capacidad de ajuste.

- `--batch` (int, default `128`): tamaño de mini-batch.
  - Batch pequeño: actualizaciones más ruidosas, posible mejor generalización.
  - Batch grande: gradiente más estable, menos actualizaciones por época.

- `--canvas-dir` (str, default `None`): ruta de muestras reales etiquetadas (`0..9/*.png`) para mezclar con train.
  - Permite acercar el dominio de entrenamiento a dibujos reales del sistema.

- `--canvas-repeat` (int, default `3`): número de repeticiones de las muestras de `canvas-dir`.
  - Repondera datos reales dentro del entrenamiento.
  - Puede mejorar desempeño en dominio real, pero sesgar si hay pocas muestras o mala calidad.

---

## 4) Ciclo por época (implementación real)
Por cada época `epoch` en `1..epochs`:

1. **Entrenamiento por mini-batches**:
   - Se generan lotes con `iterate_minibatches(X_train, y_train, batch_size=args.batch, seed=42+epoch)`.
   - En cada lote:
     - `forward(Xb)` para obtener probabilidades,
     - cálculo de pérdida con `loss(probs, one_hot(yb))`,
     - `backward(cache, one_hot(yb))` para gradientes,
     - `step(grads, lr=args.lr)` para actualizar pesos.

2. **Validación de época**:
   - Se evalúa el modelo completo en `X_val`.
   - Se calcula `val_pred = argmax(val_probs)` y luego `val_acc = mean(val_pred == y_val)`.
   - Se imprime: `Epoch XX | val_acc=...`.

Al terminar todas las épocas:

3. **Evaluación final en test**:
   - Se ejecuta inferencia sobre `X_test`.
   - Se calcula `test_acc` de forma análoga a validación.
   - Se imprime `Test acc=...`.

4. **Persistencia**:
   - Se guarda `weights.npz` con `W1`, `b1`, `W2`, `b2`.

---

## 5) Experimentos (configuración vs métricas)
Se ejecutaron experimentos con el mismo pipeline y división 80/10/10, variando hiperparámetros.

| Experimento | epochs | hidden | lr   | l2  | batch | val_acc (época final) | test_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (base) | 5 | 128 | 0.10 | 0.0 | 128 | 0.9437 | 0.9390 |
| B | 5 | 256 | 0.10 | 0.0 | 128 | 0.9459 | 0.9417 |
| C | 5 | 128 | 0.05 | 0.0 | 128 | 0.9259 | 0.9183 |

Fuente de métricas: logs en `docs/figures/experiments/*.log`.

### Configuración elegida
Se propone como configuración recomendada de compromiso:

- `--epochs 5 --hidden 256 --lr 0.1 --l2 0.0 --batch 128`

Justificación:
- Logró el mejor `val_acc` y `test_acc` dentro de las corridas comparadas.
- Mantiene un tiempo de entrenamiento similar al baseline.
- Aumentar capacidad (`hidden`) mejoró desempeño sin cambios adicionales de complejidad del pipeline.

> Nota: si se incorpora `--canvas-dir`, conviene reevaluar esta selección porque cambia la distribución de entrenamiento.

---

## 6) Comandos de referencia
Ejemplos usados para reproducir los experimentos:

```bash
PYTHONPATH=src python -m mlp.train --epochs 5 --hidden 128 --lr 0.10 --l2 0.0 --batch 128
PYTHONPATH=src python -m mlp.train --epochs 5 --hidden 256 --lr 0.10 --l2 0.0 --batch 128
PYTHONPATH=src python -m mlp.train --epochs 5 --hidden 128 --lr 0.05 --l2 0.0 --batch 128
```
