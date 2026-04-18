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

## 7. Futuras mejoras (roadmap por fases)

### 7.1 Corto plazo (0-2 sprints)

**Objetivo técnico**  
Mejorar observabilidad, trazabilidad y experiencia de uso sin alterar la arquitectura base MLP + FastAPI.

**Líneas de trabajo**
- **Documentación operativa y técnica**
  - Añadir guía de troubleshooting y decisiones de diseño (ADRs ligeros).
  - Estandarizar contratos de API con ejemplos de error y éxito por endpoint.
- **Métricas y monitoreo**
  - Versionar métricas base por release: accuracy offline, confidence promedio, p50/p95 de latencia.
  - Crear tablero mínimo (CSV/SQLite + script) para comparar tendencias entre versiones.
- **UX de inferencia**
  - Mejorar feedback en frontend (mensajes por nivel de confianza y sugerencias de centrado/grosor).
  - Añadir estados visuales de carga/error y validación temprana del canvas.

**Cambio esperado en métricas**
- Reducir tickets/incidencias de uso por ambigüedad de interfaz y errores de entrada.
- Mejorar estabilidad percibida: menor variabilidad de latencia reportada por usuario.
- Incrementar confiabilidad de análisis al disponer de línea base consistente entre releases.

**Esfuerzo estimado**  
**Bajo a medio** (1-2 personas, 1-2 sprints).

**Riesgo principal**  
Optimizar UX y documentación sin disciplina de medición puede generar mejoras cosméticas no reflejadas en KPIs.

---

### 7.2 Mediano plazo (2-6 sprints)

**Objetivo técnico**  
Elevar robustez del MLP frente a variaciones reales de escritura y validar comportamiento en entornos heterogéneos.

**Líneas de trabajo**
- **Data augmentation controlado**
  - Aplicar rotaciones suaves, traslaciones, variación de grosor/ruido y deformaciones leves.
  - Mantener experimento reproducible con semillas y configuración versionada.
- **Calibración de probabilidades**
  - Evaluar temperatura o métodos de calibración para alinear confidence con probabilidad real.
  - Ajustar umbrales de decisión para feedback y detección de baja confianza.
- **Pruebas multi-dispositivo**
  - Validar captura/inferencia en desktop y móviles (distintas resoluciones y densidades de píxel).
  - Medir diferencias de latencia, centrado de trazo y tasa de error por tipo de dispositivo.

**Cambio esperado en métricas**
- Aumento de accuracy/F1 en datos fuera de distribución MNIST-like.
- Disminución de falsos positivos con confidence alta mal calibrada.
- Menor brecha de desempeño entre escritorio y móvil (latencia y tasa de acierto).

**Esfuerzo estimado**  
**Medio** (2-3 personas, 2-4 sprints).

**Riesgo principal**  
Sobreajuste a aumentaciones sintéticas que no representen escritura real de usuarios.

---

### 7.3 Largo plazo (6+ sprints)

**Objetivo técnico**  
Evolucionar de prototipo local a plataforma escalable con mejor capacidad predictiva y operación en producción.

**Líneas de trabajo**
- **Migración/benchmark a CNN**
  - Diseñar experimento comparativo MLP vs CNN con mismo protocolo de evaluación.
  - Mantener fallback a MLP durante transición para reducir riesgo operativo.
- **Despliegue en nube**
  - Contenerización, CI/CD y entorno administrado (por ejemplo, servicio web + almacenamiento gestionado).
  - Instrumentar observabilidad de producción (logs estructurados, métricas, alertas).
- **Cuentas de usuario y persistencia avanzada**
  - Autenticación básica, historial de práctica y trazas por usuario.
  - Modelo de datos preparado para analítica de progreso y personalización de feedback.

**Cambio esperado en métricas**
- Mejora significativa de accuracy global y especialmente en trazos complejos/ruidosos.
- Mayor disponibilidad y menor tiempo de recuperación ante fallas en operación remota.
- Mayor retención de usuarios gracias a historial y personalización.

**Esfuerzo estimado**  
**Alto** (3-5 personas, 6+ sprints).

**Riesgo principal**  
Incremento de complejidad técnica y de costos de operación sin una base de datos real suficientemente representativa.

---

### 7.4 Dependencias y orden recomendado

1. **Primero: dataset real etiquetado y pipeline de calidad de datos**  
   Sin datos reales (capturas del canvas correctamente etiquetadas), cualquier mejora de modelo tendrá baja validez externa.
2. **Segundo: baseline robusta del MLP (métricas + calibración + multi-dispositivo)**  
   Permite medir el punto de partida real y evita comparar arquitecturas con protocolos inconsistentes.
3. **Tercero: comparación formal MLP vs CNN**  
   Solo cuando exista dataset real estable y baseline del MLP, para estimar ganancia real de migrar a CNN.
4. **Cuarto: despliegue en nube y cuentas de usuario**  
   Recomendado después de validar calidad predictiva y operación básica, para escalar con menor riesgo.

**Dependencia crítica explícita:** primero consolidar **dataset real etiquetado**, luego ejecutar comparación **MLP vs CNN** bajo el mismo protocolo experimental.

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
