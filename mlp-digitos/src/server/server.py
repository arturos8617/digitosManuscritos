# FastAPI: /predict API + serve ./src/client as static
import time, pathlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .infer import Inference
from .store import Store
from typing import Optional, Literal

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../src
CLIENT_DIR = ROOT / "client"
WEIGHTS_PATH = ROOT.parent / "weights.npz"  # repo root/weights.npz
DB_PATH = ROOT.parent / "logs.db"
SAMPLES_DIR = ROOT.parent / "canvas_samples"
WEIGHTS_PATHS = {
    "digits": ROOT.parent / "weights.npz",
    "vowels_lower": ROOT.parent / "weights_vowels_lower.npz",
    "vowels_upper": ROOT.parent / "weights_vowels_upper.npz",
}
TEMPLATES_PATHS = {
    "digits": ROOT.parent / "templates.npz",
    "vowels_lower": ROOT.parent / "templates_vowels_lower.npz",
    "vowels_upper": ROOT.parent / "templates_vowels_upper.npz",
}
MODE_LABELS = {
    "digits": [str(i) for i in range(10)],
    "vowels_lower": ["a", "e", "i", "o", "u"],
    "vowels_upper": ["A", "E", "I", "O", "U"],
}

app = FastAPI()

# (Optional) Allow cross-origin (harmless even if serving same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

infer_by_mode = {}
for mode, path in WEIGHTS_PATHS.items():
    if path.exists():
        infer_by_mode[mode] = Inference(
            str(path),
            str(TEMPLATES_PATHS[mode]),
            labels=MODE_LABELS[mode],
        )
store = Store(str(DB_PATH), str(SAMPLES_DIR))

class PredictRequest(BaseModel):
    image_b64: str
    mode: Literal["digits", "vowels_lower", "vowels_upper"] = "digits"
    target_symbol: Optional[str] = None
    target_digit: Optional[int] = None


class SaveSampleRequest(BaseModel):
    image_b64: str
    mode: Literal["digits", "vowels_lower", "vowels_upper"] = "digits"
    label: str | int


@app.post('/predict')
async def predict(req: PredictRequest):
    t0 = time.time()
    print("\n === Nueva Peticion ===")
    print("Mode:", req.mode)
    print("Target symbol: ", req.target_symbol)
    print("Target digit: ", req.target_digit)
    print("Longitud imagen base64:", len(req.image_b64))

    try:
        inf = infer_by_mode.get(req.mode)
        if inf is None:
            raise HTTPException(
                status_code=503,
                detail=f"No hay pesos entrenados para el modo '{req.mode}'.",
            )
        target_symbol = req.target_symbol
        if target_symbol is None and req.target_digit is not None and req.mode == "digits":
            target_symbol = str(req.target_digit)
        symbol, conf, score = inf.predict_from_base64(req.image_b64, target_symbol)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency = (time.time() - t0) * 1000.0
    print("Prediccion: ", symbol)
    print("Prediccion: ", conf)
    print("Prediccion: ", score)
    print("Prediccion: ", latency)
    print("\n === FIN Peticion ===")

    store.insert(req.mode, symbol, conf, latency)

    # Feedback mínimo (MVP) basado en target + score + conf
    feedback = None
    match = None
    expected = target_symbol
    if expected is not None:
        match = (symbol == expected)
        if not match:
            feedback = "Incorrecto. Intenta de nuevo y haz el trazo más grande y centrado."
        else:
            # score suele reflejar “qué tan parecido” al estilo plantilla
            if score is not None and score >= 85:
                feedback = "¡Muy bien! Se parece mucho."
            elif score is not None and score >= 70:
                feedback = "Bien, pero puedes mejorar. Intenta hacerlo más centrado."
            else:
                feedback = "Correcto, pero la forma puede mejorar. Hazlo más claro y completo."

    return {
        "digit": int(symbol) if req.mode == "digits" and symbol.isdigit() else None,
        "symbol": symbol,
        "confidence": conf,
        "latency_ms": latency,
        "mode": req.mode,
        "target_symbol": expected,
        "target_digit": req.target_digit,
        "match": match,
        "similarity_score": score,
        "feedback": feedback,
    }


@app.post('/samples/save')
async def save_sample(req: SaveSampleRequest):
    allowed = set(MODE_LABELS[req.mode])
    label = str(req.label)
    if label not in allowed:
        raise HTTPException(status_code=400, detail=f"label inválida para {req.mode}. Permitidas: {sorted(allowed)}")

    try:
        saved_path = store.save_sample(req.image_b64, req.mode, label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "mode": req.mode,
        "label": label,
        "saved_path": saved_path,
    }


# Mount /static -> serve client assets
app.mount("/static", StaticFiles(directory=str(CLIENT_DIR)), name="static")

# Serve index.html at root
@app.get("/")
async def index():
    return FileResponse(str(CLIENT_DIR / "index.html"))
