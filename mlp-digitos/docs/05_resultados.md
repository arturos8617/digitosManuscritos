# 05. Resultados y Evaluación

## 1. Configuración de entrenamiento
- Épocas: 5  
- Neuronas ocultas: 128  
- Tasa de aprendizaje: 0.1  
- Regularización L2: 0.0  
- Conjunto: MNIST (train/val/test)

## 2. Métricas globales
Fuente: `docs/figures/metrics.txt` y validación regenerada con `weights.npz`.

- **Accuracy (test):** `0.9680`
- **Error de clasificación (test = 1 - accuracy):** `0.0320` (3.20%)
- **Accuracy final de validación:** `0.9711`
  - Métrica regenerada cargando `weights.npz` y evaluando sobre el split de validación (`load_mnist`), debido a que no se encontró log de entrenamiento versionado.

## 3. Matriz de confusión
Se adjunta en `docs/figures/confusion_matrix.csv`.

Observaciones principales (confusiones más frecuentes):
- **8 → 1**: 6 casos.
- **0 → 8**: 6 casos.
- **6 → 5**: 6 casos.
- También destacan confusiones de 3 con 8 (8 casos) y de 9 con 7 (7 casos).

## 4. Clasification report
Extraído de `docs/figures/classification_report.txt` (precision, recall, F1 por clase).

Resumen por clase (F1 más bajos):
- **Clase 8**: F1 = **0.9485** (el menor). También tiene la menor precisión (0.9367), consistente con varias falsas alarmas hacia otras clases visualmente similares.
- **Clase 5**: F1 = **0.9556**. Su recall (0.9469) sugiere más falsos negativos que el promedio.
- **Clase 3**: F1 = **0.9669** (tercer valor más bajo), con confusiones visibles hacia clases de trazos curvos/cerrados.

Posibles causas:
- Superposición morfológica entre dígitos (por ejemplo, bucles en 8 y formas cercanas en 3/5/9).
- Variabilidad del trazo manuscrito y grosor de línea.
- Capacidad limitada del MLP frente a patrones espaciales finos (sin convoluciones).

## 5. Observaciones
- El desempeño mejora con más épocas o más neuronas (p.ej., 10 épocas, 256 neuronas).
- Invertir el color de la imagen de entrada alineó el dominio del cliente con MNIST.
- Limitaciones: sin data augmentation ni normalización por dígito; modelo MLP simple.

## 6. Conclusión
El sistema cumple el objetivo de clasificar dígitos manuscritos con un MLP implementado desde cero, alcanzando una precisión adecuada para la demostración y la defensa.
