import { useEffect, useRef, useState } from "react";

const CANVAS_W = 224;
const CANVAS_H = 224;

const OUT_W = 28;
const OUT_H = 28;

export default function DigitCanvas({ onStatus, onResult, targetDigit }) {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "black";
  }, []);

  function getPos(evt) {
    const c = canvasRef.current;
    const rect = c.getBoundingClientRect();
    if (evt.touches && evt.touches[0]) {
      return {
        x: evt.touches[0].clientX - rect.left,
        y: evt.touches[0].clientY - rect.top,
      };
    }
    return {
      x: evt.clientX - rect.left,
      y: evt.clientY - rect.top,
    };
  }

  function startDraw(evt) {
    evt.preventDefault?.();
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    const p = getPos(evt);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    setDrawing(true);
  }

  function moveDraw(evt) {
    evt.preventDefault?.();
    if (!drawing) return;
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    const p = getPos(evt);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }

  function endDraw(evt) {
    evt.preventDefault?.();
    setDrawing(false);
  }

  function clear() {
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "black";
    onResult(null);
    onStatus?.("Listo");
  }

  function getProcessedB64() {
    const c = canvasRef.current;
  
    const tmp = document.createElement("canvas");
    tmp.width = CANVAS_W;
    tmp.height = CANVAS_H;
    const tctx = tmp.getContext("2d", { willReadFrequently: true });
    tctx.drawImage(c, 0, 0);
  
    const imgData = tctx.getImageData(0, 0, CANVAS_W, CANVAS_H);
    const data = imgData.data;
  
    let minX = CANVAS_W, minY = CANVAS_H, maxX = 0, maxY = 0;
    let found = false;
  
    // 1) Bounding box con umbral más estricto (evita ruido)
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  
      if (lum < 220) { // antes 240: demasiado permisivo
        found = true;
        const idx = i / 4;
        const x = idx % CANVAS_W;
        const y = Math.floor(idx / CANVAS_W);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  
    if (!found || maxX <= minX || maxY <= minY) return null;
  
    // 2) Padding ligeramente mayor (ayuda con 1,7,9)
    const pad = 10;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(CANVAS_W - 1, maxX + pad);
    maxY = Math.min(CANVAS_H - 1, maxY + pad);
  
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;
  
    // Crop
    const crop = document.createElement("canvas");
    crop.width = cropW;
    crop.height = cropH;
    const cctx = crop.getContext("2d");
    cctx.drawImage(c, minX, minY, cropW, cropH, 0, 0, cropW, cropH);
  
    // Center in square
    const size = Math.max(cropW, cropH);
    const square = document.createElement("canvas");
    square.width = size;
    square.height = size;
    const sctx = square.getContext("2d");
    sctx.fillStyle = "white";
    sctx.fillRect(0, 0, size, size);
    sctx.drawImage(crop, (size - cropW) / 2, (size - cropH) / 2);
  
    // Scale to 28x28
    const out = document.createElement("canvas");
    out.width = OUT_W;
    out.height = OUT_H;
    const octx = out.getContext("2d", { willReadFrequently: true });
    octx.imageSmoothingEnabled = true;
    octx.drawImage(square, 0, 0, OUT_W, OUT_H);
  
    // 3) Post-procesado en cliente: aumentar contraste rápido
    // (muy parecido a lo que hace tu infer.py)
    const od = octx.getImageData(0, 0, OUT_W, OUT_H);
    const px = od.data;
    for (let i = 0; i < px.length; i += 4) {
      const r = px[i], g = px[i + 1], b = px[i + 2];
      // luminancia (0=negro, 255=blanco)
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      // convertimos a "tinta" (1=trazo, 0=fondo) aproximado
      let ink = 1.0 - (lum / 255.0);
      // contraste suave
      ink = Math.pow(Math.max(0, Math.min(1, ink)), 0.7);
      // umbral suave
      ink = (ink > 0.2) ? ink : 0.0;
      // regresamos a gris invertido para PNG (negro fondo / blanco tinta no importa,
      // tu servidor vuelve a invertir; lo importante es que el trazo quede fuerte)
      const outLum = 255 * (1.0 - ink);
      px[i] = px[i + 1] = px[i + 2] = outLum;
      px[i + 3] = 255;
    }
    octx.putImageData(od, 0, 0);
  
    return out.toDataURL("image/png").split(",")[1];
  }
  

  async function predict() {
    onStatus?.("Procesando...");
    const b64 = getProcessedB64();
    if (!b64) {
      onStatus?.("No se detectó trazo");
      onResult({ digit: "-", confidence: 0, latency_ms: 0 });
      return;
    }

    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64: b64, target_digit: targetDigit }),
    });

    const json = await resp.json();
    onResult(json);
    onStatus?.("Listo");
  }

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{ border: "2px solid #333", borderRadius: 10, touchAction: "none" }}
        onMouseDown={startDraw}
        onMouseMove={moveDraw}
        onMouseUp={endDraw}
        onMouseLeave={endDraw}
        onTouchStart={startDraw}
        onTouchMove={moveDraw}
        onTouchEnd={endDraw}
      />

      <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
        <button onClick={clear} style={btnStyle}>Limpiar</button>
        <button onClick={predict} style={btnStylePrimary}>Predecir</button>
      </div>
    </div>
  );
}

const btnStyle = {
  padding: "10px 14px",
  borderRadius: 10,
  border: "1px solid #ccc",
  background: "white",
  cursor: "pointer",
};

const btnStylePrimary = {
  ...btnStyle,
  border: "1px solid #222",
};
