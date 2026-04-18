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

---

## 8. Flujo detallado por endpoint

### 8.1 `POST /predict`

#### Entrada esperada
`/predict` recibe un JSON con este contrato:

- `image_b64` (**string**, obligatorio): imagen codificada en Base64.
- `target_digit` (**int**, opcional): dígito objetivo para calcular similitud y feedback.

Modelo Pydantic en backend (`PredictRequest`):

```json
{
  "image_b64": "<base64>",
  "target_digit": 3
}
```

#### Pipeline interno (paso a paso)
1. **Recepción HTTP (FastAPI):**
   - El endpoint asíncrono `predict(req: PredictRequest)` valida tipos con Pydantic.
   - Se registra tiempo inicial para latencia.

2. **Preprocesado e inferencia (`src/server/infer.py` + `src/mlp/data.py`):**
   - `Inference.predict_from_base64` decodifica Base64 (`base64.b64decode`) y abre la imagen con PIL.
   - `preprocess_pil_for_mlp` aplica la misma transformación de entrenamiento/inferencia:
     - conversión a escala de grises (`L`),
     - redimensión a `28x28`,
     - normalización `0..1`,
     - inversión de colores (`1 - x`) para alinear con MNIST,
     - ajuste suave de contraste (`x ** 0.7`),
     - umbral suave (`(x > 0.2) * x`),
     - renormalización por máximo,
     - reshape a vector de `784` features.

3. **Inferencia MLP:**
   - `MLP.forward(x)` produce probabilidades por clase.
   - `digit = argmax(probs)` y `confidence = probs[digit]`.
   - Si `target_digit` existe, se calcula `similarity_score` por similitud coseno contra plantilla del dígito objetivo y se escala a `0..100`.

4. **Lógica de feedback (MVP):**
   - Se evalúa `match` cuando hay `target_digit`.
   - Se genera `feedback` textual con base en:
     - coincidencia `digit == target_digit`,
     - `similarity_score` (rangos >=85, >=70 o menor).

5. **Persistencia en SQLite (`src/server/store.py`):**
   - Se calcula `latency_ms`.
   - Se llama `store.insert(digit, conf, latency)` para insertar en `logs.db`.
   - Tabla `logs`:
     - `id` (PK autoincrement),
     - `ts` (epoch segundos),
     - `digit`,
     - `confidence`,
     - `latency_ms`.

#### Estructura exacta de respuesta JSON
La respuesta del endpoint devuelve exactamente:

```json
{
  "digit": 8,
  "confidence": 0.973421,
  "latency_ms": 12.34,
  "target_digit": 3,
  "match": false,
  "similarity_score": 64.1,
  "feedback": "Incorrecto. Intenta de nuevo y haz el dígito más grande y centrado."
}
```

Notas:
- `target_digit` puede ser `null` si no se envía.
- `match`, `similarity_score` y `feedback` pueden ser `null` cuando no hay `target_digit`.
- En error de decodificación/procesamiento se responde `HTTP 400`.

### 8.2 `POST /samples/save`

#### Entrada y validación
`/samples/save` recibe:

```json
{
  "image_b64": "<base64>",
  "label": 5
}
```

Reglas:
- `label` debe estar entre `0` y `9`; fuera de rango devuelve `HTTP 400` con mensaje `"label debe estar entre 0 y 9"`.

#### Guardado por carpeta y formato
1. Se crea (si no existe) la carpeta por etiqueta: `canvas_samples/<label>/`.
2. Se decodifica `image_b64`.
3. La imagen se transforma a:
   - escala de grises `L`,
   - tamaño `28x28`.
4. Se guarda en **PNG** con nombre:
   - `<timestamp>_<uuid8>.png`
   - Ejemplo: `canvas_samples/5/1713410000_a1b2c3d4.png`.

Respuesta:

```json
{
  "ok": true,
  "label": 5,
  "saved_path": "canvas_samples/5/1713410000_a1b2c3d4.png"
}
```

### 8.3 Nota técnica de concurrencia y robustez

#### Estado actual
- El endpoint `predict` está definido como `async`, pero la inferencia NumPy/PIL y SQLite son operaciones CPU/IO locales que se ejecutan de forma síncrona dentro del handler.
- SQLite se abre con `check_same_thread=False`, permitiendo uso desde distintos hilos sobre la misma conexión.

#### Limitaciones y riesgos bajo alta carga
- **Contención de escritura en SQLite:** múltiples escrituras simultáneas pueden producir bloqueos (`database is locked`) o aumentar latencia.
- **Conexión compartida única:** una sola conexión global simplifica el código, pero incrementa riesgo de condiciones de carrera/serialización implícita.
- **Handler async con trabajo bloqueante:** bajo picos de tráfico, tareas bloqueantes reducen el beneficio del modelo async y pueden degradar throughput.

#### Mejora futura mínima recomendada
Estrategia incremental (sin rediseño completo):
1. **Conexión por request (o pequeño pool):**
   - abrir/cerrar conexión SQLite por operación, o
   - usar un pool ligero (si se migra a driver compatible).
2. **Cola de escritura (writer único):**
   - encolar eventos de predicción,
   - persistirlos desde un worker dedicado para evitar contención.

Con cualquiera de estas dos opciones se reduce la probabilidad de bloqueos y mejora la estabilidad cuando sube la concurrencia.
