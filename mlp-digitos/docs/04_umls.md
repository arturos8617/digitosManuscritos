# 04. Diagramas UML

## 1. Caso de uso principal
**Actor:** Usuario final  
**Caso:** Clasificar dígito manuscrito

**Flujo principal:**
1. El usuario abre la interfaz web.
2. Dibuja un dígito en el canvas.
3. Presiona *Predecir*.
4. El sistema envía la imagen al servidor.
5. El servidor responde con el dígito y la confianza.
6. El usuario observa el resultado.

---

## 2. Diagrama de clases (texto descriptivo)

- **MLP**
  - Atributos: `W1, b1, W2, b2`
  - Métodos: `forward(x)`, `backward()`, `loss()`

- **Inference**
  - Atributos: `model`
  - Métodos: `predict_from_base64(image_b64)`

- **Store**
  - Atributos: `db_path`
  - Métodos: `insert(digit, conf, latency)`

- **Server (FastAPI)**
  - Métodos: `/predict`, `/static`

---

## 3. Diagrama de secuencia (resumen)
Usuario → Cliente → Servidor → Inference → MLP
│ │ │ │
│ Dibujar│ │ │
│ click▶ │ POST /predict │
│ │────────────────────► │
│ │ │ inferir()│
│ │ │────────► │
│ │ │ resultado│
│ │◄──────────────────── │
│ │ mostrar resultado │


---

## 4. Consideraciones
Los diagramas UML se elaborarán con **PlantUML** o **draw.io** para presentación.  
El nivel de detalle se mantendrá simple, suficiente para ilustrar las interacciones y responsabilidades.
