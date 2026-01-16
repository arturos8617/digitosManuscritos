const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Inicializar canvas en blanco
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Ajustes del lápiz
ctx.lineWidth = 22;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'black';

let drawing = false;

// Obtener posición del mouse o touch
function pos(evt) {
  const rect = canvas.getBoundingClientRect();
  if (evt.touches && evt.touches[0]) {
    return {
      x: evt.touches[0].clientX - rect.left,
      y: evt.touches[0].clientY - rect.top
    };
  }
  return { x: evt.offsetX, y: evt.offsetY };
}

// Eventos para dibujar
canvas.addEventListener('mousedown', e => {
  drawing = true;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
});

canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = pos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});

canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseleave', () => drawing = false);

canvas.addEventListener('touchstart', e => {
  drawing = true;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
});

canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (!drawing) return;
  const p = pos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});

canvas.addEventListener('touchend', () => drawing = false);

// Limpiar canvas
document.getElementById('clear').onclick = () => {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'black';
};


// -----------------------------------------------------
// 🔥 FUNCIÓN PRINCIPAL DE PROCESAMIENTO
// Detectar, recortar, centrar y escalar el dígito
// -----------------------------------------------------

function getProcessedImage28() {
  const temp = document.createElement('canvas');
  const tctx = temp.getContext('2d');
  temp.width = canvas.width;
  temp.height = canvas.height;

  // Copiar contenido del canvas
  tctx.drawImage(canvas, 0, 0);

  // Obtener pixeles
  const imgData = tctx.getImageData(0, 0, temp.width, temp.height);
  const data = imgData.data;

  // Buscar bounding box del trazo
  let minX = temp.width, minY = temp.height;
  let maxX = 0, maxY = 0;

  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    const r = data[i], g = data[i + 1], b = data[i + 2];

    // Consideramos "negro" si la luminancia es baja
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;

    if (lum < 240) {  // threshold
      let index = i / 4;
      let x = index % temp.width;
      let y = Math.floor(index / temp.width);

      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  // Si no se dibujó nada, regresar null
  if (maxX <= minX || maxY <= minY) {
    return null;
  }

  // Recortar el área donde está el dígito
  const cropWidth = maxX - minX;
  const cropHeight = maxY - minY;

  const cropped = document.createElement('canvas');
  const cctx = cropped.getContext('2d');
  cropped.width = cropWidth;
  cropped.height = cropHeight;

  cctx.drawImage(
    canvas,
    minX, minY, cropWidth, cropHeight,
    0, 0, cropWidth, cropHeight
  );

  // Crear un canvas cuadrado con padding para centrar
  const size = Math.max(cropWidth, cropHeight);
  const square = document.createElement('canvas');
  const sctx = square.getContext('2d');
  square.width = size;
  square.height = size;

  // Fondo blanco
  sctx.fillStyle = "white";
  sctx.fillRect(0, 0, size, size);

  const offsetX = (size - cropWidth) / 2;
  const offsetY = (size - cropHeight) / 2;
  sctx.drawImage(cropped, offsetX, offsetY);

  // Escalar a 28x28
  const out = document.createElement("canvas");
  out.width = 28;
  out.height = 28;
  const octx = out.getContext("2d");

  octx.drawImage(square, 0, 0, 28, 28);

  // Convertir a base64 PNG
  return out.toDataURL("image/png").split(",")[1];
}


// -----------------------------------------------------
// 🔥 BOTÓN DE PREDICCIÓN
// -----------------------------------------------------

document.getElementById('predict').onclick = async () => {
  const out = document.getElementById('out');
  out.textContent = 'Procesando...';

  const processedB64 = getProcessedImage28();
  if (!processedB64) {
    out.textContent = "No se detectó ningún trazo.";
    return;
  }

  const resp = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: processedB64 })
  });

  const json = await resp.json();

  out.textContent =
    `Predicción: ${json.digit}  ` +
    `(confianza: ${(json.confidence * 100).toFixed(1)}%),  ` +
    `latencia: ${json.latency_ms.toFixed(1)} ms`;
};
