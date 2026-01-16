# 02. Modelo Matemático del Perceptrón Multicapa

## 1. Introducción
Este modelo se utiliza para clasificar imágenes de dígitos manuscritos (0-9) de 28×28 píxeles.  
Se implementa un perceptrón multicapa (MLP) desde cero, entrenado con el conjunto MNIST.

---

## 2. Representación de datos
Cada imagen \( x \in \mathbb{R}^{784} \) corresponde a una matriz 28×28 aplanada a un vector columna.  
La etiqueta \( y \in \{0,1,\dots,9\} \) se codifica en *one-hot*:

\[
y = [0,0,\dots,1,\dots,0]^T \in \mathbb{R}^{10}
\]

---

## 3. Propagación hacia adelante
El MLP consta de:
- Capa oculta con \( H \) neuronas y función de activación ReLU.
- Capa de salida con 10 neuronas y activación Softmax.

\[
z^{(1)} = W^{(1)}x + b^{(1)}, \quad a^{(1)} = \text{ReLU}(z^{(1)})
\]
\[
z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}, \quad \hat{y} = \text{Softmax}(z^{(2)})
\]

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

---

## 4. Función de pérdida
Se utiliza entropía cruzada:

\[
L = -\frac{1}{m}\sum_{i=1}^{m} \sum_{k=1}^{10} y_{k}^{(i)} \log(\hat{y}_{k}^{(i)})
\]

---

## 5. Retropropagación
Se derivan los gradientes para cada capa:

\[
\delta^{(2)} = \hat{y} - y
\]
\[
\delta^{(1)} = (W^{(2)})^T \delta^{(2)} \odot \text{ReLU}'(z^{(1)})
\]
\[
\frac{\partial L}{\partial W^{(2)}} = \frac{1}{m}\delta^{(2)}(a^{(1)})^T, \quad
\frac{\partial L}{\partial b^{(2)}} = \frac{1}{m}\sum_i \delta^{(2)}_i
\]
\[
\frac{\partial L}{\partial W^{(1)}} = \frac{1}{m}\delta^{(1)}x^T, \quad
\frac{\partial L}{\partial b^{(1)}} = \frac{1}{m}\sum_i \delta^{(1)}_i
\]

---

## 6. Actualización de pesos
\[
W := W - \eta \frac{\partial L}{\partial W}, \quad
b := b - \eta \frac{\partial L}{\partial b}
\]

Donde \( \eta \) es la tasa de aprendizaje.

---

## 7. Resultados experimentales
- **Épocas:** 5  
- **Neuronas ocultas:** 128  
- **Precisión (test):** 93.7 %  
- **Dataset:** MNIST (60 000 entrenamiento / 10 000 prueba)  

---

## 8. Conclusión
El modelo alcanza un rendimiento satisfactorio usando solo NumPy, demostrando comprensión de redes neuronales, funciones de activación y optimización numérica sin librerías de alto nivel.
