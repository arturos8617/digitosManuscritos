# FastAPI: /predict API + serve ./src/client as static
import time, pathlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .infer import Inference
from .store import Store
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../src
CLIENT_DIR = ROOT / "client"
WEIGHTS_PATH = ROOT.parent / "weights.npz"  # repo root/weights.npz
DB_PATH = ROOT.parent / "logs.db"

app = FastAPI()

# (Optional) Allow cross-origin (harmless even if serving same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

inf = Inference(str(WEIGHTS_PATH))
store = Store(str(DB_PATH))

class PredictRequest(BaseModel):
    image_b64: str
    target_digit: Optional[int] = None

@app.post('/predict')
async def predict(req: PredictRequest):
    t0 = time.time()
    try:
        digit, conf, score = inf.predict_from_base64(req.image_b64, req.target_digit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency = (time.time() - t0) * 1000.0
    store.insert(digit, conf, latency)

    # Feedback mínimo (MVP) basado en target + score + conf
    feedback = None
    match = None
    if req.target_digit is not None:
        match = (digit == req.target_digit)
        if not match:
            feedback = "Incorrecto. Intenta de nuevo y haz el dígito más grande y centrado."
        else:
            # score suele reflejar “qué tan parecido” al estilo plantilla
            if score is not None and score >= 85:
                feedback = "¡Muy bien! Se parece mucho."
            elif score is not None and score >= 70:
                feedback = "Bien, pero puedes mejorar. Intenta hacerlo más centrado."
            else:
                feedback = "Correcto, pero la forma puede mejorar. Hazlo más claro y completo."

    return {
        "digit": digit,
        "confidence": conf,
        "latency_ms": latency,
        "target_digit": req.target_digit,
        "match": match,
        "similarity_score": score,
        "feedback": feedback,
    }

# Mount /static -> serve client assets
app.mount("/static", StaticFiles(directory=str(CLIENT_DIR)), name="static")

# Serve index.html at root
@app.get("/")
async def index():
    return FileResponse(str(CLIENT_DIR / "index.html"))
