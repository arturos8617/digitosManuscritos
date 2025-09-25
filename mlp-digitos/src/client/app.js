const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'white';
ctx.fillRect(0,0,canvas.width,canvas.height);
ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.strokeStyle = 'black';
let drawing = false;

function pos(evt){
  if (evt.touches && evt.touches[0]){
    const rect = canvas.getBoundingClientRect();
    return {x: evt.touches[0].clientX - rect.left, y: evt.touches[0].clientY - rect.top};
  }
  return {x: evt.offsetX, y: evt.offsetY};
}

canvas.addEventListener('mousedown', e=>{ drawing=true; const p=pos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); });
canvas.addEventListener('mousemove', e=>{ if(!drawing) return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
canvas.addEventListener('mouseup', ()=> drawing=false);
canvas.addEventListener('mouseleave', ()=> drawing=false);
canvas.addEventListener('touchstart', e=>{ drawing=true; const p=pos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); });
canvas.addEventListener('touchmove', e=>{ e.preventDefault(); if(!drawing) return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
canvas.addEventListener('touchend', ()=> drawing=false);

document.getElementById('clear').onclick = ()=>{ ctx.fillStyle='white'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.fillStyle='black'; };

document.getElementById('predict').onclick = async ()=>{
  const out = document.getElementById('out');
  const dataUrl = canvas.toDataURL('image/png');
  const b64 = dataUrl.split(',')[1];
  out.textContent = '...';
  const resp = await fetch('http://localhost:8000/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: b64 })
  });
  const json = await resp.json();
  out.textContent = `Predicción: ${json.digit} (confianza: ${(json.confidence*100).toFixed(1)}%), latencia: ${json.latency_ms.toFixed(1)} ms`;
};
