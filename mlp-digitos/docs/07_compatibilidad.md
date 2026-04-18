# 07 — Compatibilidad

## Objetivo
Registrar el estado de compatibilidad del flujo principal de la app:

1. **Dibujar** en canvas.
2. **Predecir** dígito vía API.
3. **Guardar muestra** en almacenamiento local del backend.

## Alcance de verificación

- **Fecha de verificación:** 2026-04-18.
- **Entorno ejecutado directamente:** Linux (contenedor), validación técnica de backend y rutas de inferencia/guardado mediante `pytest`.
- **Combinaciones prioritarias:** escritorio moderno (Chrome/Firefox/Edge/Safari) y móvil táctil (Chrome Android / Safari iOS).

> Nota: en este entorno no se dispone de granja real de dispositivos/navegadores; por ello, los casos fuera de Linux/desktop quedan marcados como **parcial** hasta validar en hardware objetivo.

## Matriz de compatibilidad (combinaciones prioritarias)

| SO | Navegador | Tipo de entrada | Resultado | Observaciones |
|---|---|---|---|---|
| Linux | Chrome (Chromium) | Mouse | **OK** | Flujo completo validado técnicamente (dibujar/payload canvas, predecir y persistir muestra) con pruebas automatizadas de API e inferencia. |
| Linux | Firefox | Mouse | **Parcial** | Backend compatible; falta corrida E2E manual en navegador Firefox para confirmar eventos de puntero y UX final. |
| Linux | Edge | Mouse | **Parcial** | Motor Chromium sugiere comportamiento equivalente a Chrome, pendiente confirmación manual del flujo UI. |
| macOS | Chrome | Mouse/Trackpad | **Parcial** | Sin validación manual en equipo macOS; no se anticipan bloqueos en API. |
| macOS | Safari | Mouse/Trackpad | **Parcial** | Riesgo moderado por diferencias de eventos de canvas/touch en Safari; requiere prueba manual dedicada. |
| Windows | Chrome | Mouse | **Parcial** | Pendiente validación manual de instalación local y permisos de escritura para guardado de muestras. |
| Windows | Edge | Mouse | **Parcial** | Esperable similar a Chrome; faltan pruebas manuales E2E. |
| Android | Chrome | Touch | **Parcial** | Pendiente validar precisión de trazo táctil y escalado del canvas en pantallas pequeñas. |
| iOS | Safari | Touch | **Parcial** | Caso prioritario pendiente: Safari iOS puede requerir ajuste fino de eventos táctiles y prevención de scroll accidental. |

## Evidencia de verificación ejecutada

Se ejecutaron pruebas automatizadas orientadas al flujo completo:

- `tests/test_api.py`: contrato de endpoints de API de predicción.
- `tests/test_infer_paths.py`: rutas de inferencia/modelo.
- `tests/test_store_samples.py`: guardado de muestras del canvas.
- `tests/test_canvas_data.py`: lectura y normalización de datos provenientes del canvas.

Resultado en esta fecha: **4 passed**.

## Limitaciones detectadas

1. **Cobertura real de navegadores/dispositivos incompleta**
   - Solo hubo validación ejecutable en entorno Linux de desarrollo.
   - El resto de combinaciones queda en estado parcial hasta prueba manual en hardware real.

2. **Touch en móviles (riesgo funcional)**
   - No se confirmó aún la ergonomía de trazo ni la consistencia de eventos táctiles (inicio/movimiento/fin) en Android e iOS.
   - Posible interferencia por scroll/gestos del navegador si no se gestiona correctamente el área táctil del canvas.

3. **Rendimiento en móviles (riesgo de UX)**
   - No hay medición formal de latencia en equipos de gama media/baja.
   - Se recomienda medir tiempo de respuesta de predicción y frame rate de dibujo bajo red y CPU limitadas.

## Recomendaciones para cierre de compatibilidad

1. Ejecutar una pasada manual de humo por combinación prioritaria con checklist único:
   - Dibujar dígito legible.
   - Predecir y revisar confianza.
   - Guardar muestra y verificar archivo generado.
2. Registrar evidencia (captura + resultado) por combinación.
3. Promover a **OK** solo combinaciones con flujo completo comprobado en dispositivo real.
