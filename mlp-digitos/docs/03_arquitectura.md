# 03. Arquitectura del Sistema

## 1. Descripción general
El sistema sigue una arquitectura **cliente-servidor** con tres componentes principales:

1. **Cliente web:** interfaz gráfica con canvas HTML5 para capturar el dígito.
2. **Servidor FastAPI:** recibe la imagen codificada en Base64, la procesa y devuelve la predicción.
3. **Modelo MLP:** clasificador entrenado con MNIST y cargado desde `weights.npz`.

---

## 2. Diagrama de arquitectura
[Usuario]
│
▼
[Navegador Web]
│ (HTTP POST /predict)
▼
[Servidor FastAPI] ──► [MLP Inference]
│ │
▼ ▼
[Base de datos logs.db] [weights.npz]

---

## 3. Flujo de comunicación
1. El usuario dibuja un dígito → se genera imagen 28×28 → se codifica en Base64.  
2. El cliente envía `POST /predict` con `{image_b64: ...}`.  
3. El servidor decodifica, normaliza (e invierte colores), y ejecuta `MLP.forward`.  
4. Devuelve un JSON con `{digit, confidence, latency_ms}`.  
5. La predicción se almacena en `logs.db`.

---

## 4. Tecnologías
- **Frontend:** HTML5, JavaScript, Canvas API, Fetch API.  
- **Backend:** Python 3.10, FastAPI, NumPy, Pillow, SQLite3.  
- **Entrenamiento:** NumPy puro, scikit-learn (descarga de MNIST).

---

## 5. Requisitos no funcionales
- Latencia promedio < 100 ms por predicción.  
- Independencia de frameworks de alto nivel.  
- Código multiplataforma (Linux/Windows/macOS).  
- Comunicación mediante HTTP REST (JSON).

---

## 6. Seguridad y validación
- Límite de tamaño de imagen ≤ 50 KB.  
- Validación de formato Base64.  
- Manejo de errores y logging básico.

---

## 7. Futuras mejoras
- Integración con TensorFlow/PyTorch para comparar precisión.  
- Servidor remoto en nube (Render, Railway, etc.).  
- Persistencia avanzada con registros de usuario.