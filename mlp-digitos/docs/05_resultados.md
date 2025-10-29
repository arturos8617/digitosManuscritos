# 05. Resultados y Evaluación

## 1. Configuración de entrenamiento
- Épocas: 5  
- Neuronas ocultas: 128  
- Tasa de aprendizaje: 0.1  
- Regularización L2: 0.0  
- Conjunto: MNIST (train/val/test)

## 2. Métricas globales
Copiar desde `docs/figures/metrics.txt`:

- **Accuracy (test):** `X.XXXX`

## 3. Matriz de confusión
Se adjunta en `docs/figures/confusion_matrix.csv`.

> (Opcional) Si deseas pegarla como tabla, conviértela a Markdown o inserta una imagen generada.

## 4. Clasification report
Extraído de `docs/figures/classification_report.txt` (precision, recall, F1 por clase).

## 5. Observaciones
- El desempeño mejora con más épocas o más neuronas (p.ej., 10 épocas, 256 neuronas).
- Invertir el color de la imagen de entrada alineó el dominio del cliente con MNIST.
- Limitaciones: sin data augmentation ni normalización por dígito; modelo MLP simple.

## 6. Conclusión
El sistema cumple el objetivo de clasificar dígitos manuscritos con un MLP implementado desde cero, alcanzando una precisión adecuada para la demostración y la defensa.
