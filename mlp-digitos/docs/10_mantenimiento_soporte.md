# 10. Mantenimiento, soporte y ciclo de mejora continua

## 1) Objetivo
Este documento define el plan operativo para mantener estable el sistema de reconocimiento de dígitos, asegurar trazabilidad del modelo en producción y establecer un ciclo disciplinado de mejora basado en casos reales.

Aplica a:
- `logs.db` (registro operativo del backend),
- `weights.npz` (pesos de inferencia),
- `canvas_samples/` (muestras reales recolectadas desde el canvas),
- flujo de soporte para incidentes y solicitudes.

---

## 2) Tareas periódicas de mantenimiento

### 2.1 Respaldo y limpieza de `logs.db`
**Objetivo:** evitar crecimiento descontrolado, reducir riesgo de pérdida de datos operativos y mantener auditoría mínima útil.

#### Frecuencia recomendada
- **Diaria (automática):** respaldo incremental.
- **Semanal (automática):** respaldo completo comprimido.
- **Mensual (manual/automática):** limpieza y compactación.

#### Política sugerida
1. **Respaldar antes de limpiar.**
2. Conservar al menos:
   - respaldos diarios de los últimos **7 días**,
   - respaldos semanales de las últimas **4 semanas**,
   - respaldos mensuales de los últimos **6 meses**.
3. Eliminar registros antiguos según necesidad operativa (por ejemplo, mayores a 90 o 180 días).
4. Ejecutar compactación posterior (equivalente a `VACUUM` en SQLite) durante ventana de bajo tráfico.

#### Checklist operativo
- Verificar que el backup sea legible/restaurable.
- Validar espacio libre en disco antes y después de limpieza.
- Registrar fecha, operador, tamaño de `logs.db` antes/después y cantidad de filas eliminadas.

---

### 2.2 Versionado de `weights.npz` y trazabilidad de entrenamientos
**Objetivo:** poder responder con precisión qué modelo está desplegado, con qué datos se entrenó y qué desempeño tenía al momento del release.

#### Reglas de versionado
- No sobrescribir sin trazabilidad. Publicar cada modelo con etiqueta de versión, por ejemplo:
  - `weights_vYYYYMMDD_HHMM.npz` o
  - `weights_<git_sha>.npz`.
- Mantener `weights.npz` solo como alias del modelo activo en producción.

#### Metadatos mínimos por entrenamiento
Guardar en un registro (archivo `.md`, `.json` o tabla interna) al menos:
- fecha/hora de entrenamiento,
- commit Git del código,
- hiperparámetros (`--epochs`, `--hidden`, `--lr`, `--batch`, `--l2`),
- uso de muestras reales (`--canvas-dir`, `--canvas-repeat`),
- tamaño de datos por split,
- métricas de validación y prueba,
- ruta del artefacto de pesos publicado.

#### Criterio de promoción a producción
- Solo promover modelos con evidencia de mejora o estabilidad frente al baseline vigente.
- Toda promoción debe dejar constancia de:
  - versión anterior,
  - versión nueva,
  - motivo del cambio,
  - responsable,
  - fecha/hora de despliegue.

---

### 2.3 Revisión de muestras en `canvas_samples/`
**Objetivo:** controlar calidad de los datos reales para que el reentrenamiento mejore y no degrade el modelo.

#### Frecuencia recomendada
- **Semanal:** revisión rápida de calidad.
- **Quincenal o mensual:** depuración más profunda.

#### Criterios de calidad
Marcar para exclusión o revisión manual muestras con:
- etiqueta dudosa (archivo en clase incorrecta),
- trazo incompleto o casi vacío,
- ruido extremo o dibujo no numérico,
- duplicados masivos que sesguen una clase.

#### Control de balance
- Monitorear cantidad de muestras por clase (`0..9`).
- Si una clase está subrepresentada, priorizar recolección dirigida.
- Evitar que una sola clase domine el conjunto real.

---

## 3) Flujo de soporte

### 3.1 Canal de reporte de errores
Cada incidente debe registrarse en un ticket con estos campos mínimos:
- ID del ticket,
- fecha/hora del reporte,
- entorno (dev/staging/prod),
- pasos para reproducir,
- resultado esperado vs observado,
- evidencia (captura, log, payload),
- versión de modelo (`weights`) y versión de backend/frontend.

### 3.2 Severidad y prioridad
Usar una matriz simple (Severidad x Prioridad):

- **Severidad S1 (Crítica):** servicio caído, errores generalizados, pérdida de funcionalidad esencial.
- **Severidad S2 (Alta):** función principal degradada, alto impacto en usuarios.
- **Severidad S3 (Media):** impacto parcial con workaround.
- **Severidad S4 (Baja):** problemas cosméticos o de baja frecuencia.

Prioridad sugerida:
- **P1:** atención inmediata.
- **P2:** siguiente ventana operativa.
- **P3:** backlog cercano.
- **P4:** mejora planificada.

### 3.3 Tiempo objetivo de respuesta (SLA interno)
Objetivos para **primera respuesta** y **contención inicial**:

- **S1 / P1:** primera respuesta ≤ **30 min**, contención ≤ **4 h**.
- **S2 / P2:** primera respuesta ≤ **4 h hábiles**, contención ≤ **1 día hábil**.
- **S3 / P3:** primera respuesta ≤ **1 día hábil**, resolución planificada ≤ **5 días hábiles**.
- **S4 / P4:** primera respuesta ≤ **2 días hábiles**, resolución según roadmap.

> Nota: estos tiempos son objetivos operativos y deben revisarse trimestralmente con base en capacidad real del equipo.

---

## 4) Ciclo de realimentación del modelo

### 4.1 Recolectar casos fallidos
Fuentes prioritarias:
- predicciones con baja confianza,
- errores reportados por usuarios,
- discrepancias repetidas en una misma clase.

Para cada caso fallido, conservar:
- imagen de entrada,
- etiqueta esperada,
- predicción del modelo,
- confianza,
- contexto mínimo (fecha, versión de modelo, canal).

### 4.2 Etiquetado y curación
- Revisar manualmente los casos fallidos.
- Confirmar/corregir etiqueta.
- Separar muestras inválidas (ruido extremo, no-dígito) para no contaminar entrenamiento.
- Integrar muestras válidas en `canvas_samples/<clase>/`.

### 4.3 Reentrenamiento con datos reales
Ejecutar reentrenamiento incorporando muestras de canvas:

```bash
python -m src.mlp.train --epochs 15 --hidden 256 --lr 0.1 --canvas-dir canvas_samples --canvas-repeat 3
```

Recomendaciones:
- Probar varios valores de `--canvas-repeat` (por ejemplo 2, 3, 5) y comparar.
- Mantener registro explícito de la configuración utilizada en cada corrida.

### 4.4 Reevaluación antes de desplegar
Antes de promover nuevo `weights`:
1. Evaluar métricas de validación y prueba (accuracy global y por clase).
2. Comparar contra baseline actual en los mismos criterios.
3. Revisar específicamente el subconjunto de casos fallidos históricos.
4. Desplegar solo si no hay regresiones relevantes o si existe justificación documentada.

### 4.5 Cierre del ciclo
- Publicar resumen del ciclo (hallazgos, mejoras, regresiones, decisión final).
- Actualizar documentación de métricas y versión desplegada.
- Programar próxima iteración (semanal, quincenal o mensual según volumen de incidencias).

---

## 5) Indicadores recomendados
Para seguimiento de mantenimiento y soporte:
- crecimiento mensual de `logs.db`,
- porcentaje de tickets por severidad,
- cumplimiento de tiempos de primera respuesta,
- tasa de error de inferencia reportada,
- mejora de accuracy en casos reales de `canvas_samples`.

Estos indicadores permiten decidir cuándo priorizar deuda técnica, recolección de datos o reentrenamiento.
