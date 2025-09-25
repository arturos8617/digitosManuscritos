# FastAPI con endpoint /predict que recibe imagen base64 y responde JSON.
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .infer import Inference
from .store import Store

app = FastAPI()
inf = Inference('weights.npz')
store = Store('logs.db')

class PredictRequest(BaseModel):
    image_b64: str

@app.post('/predict')
async def predict(req: PredictRequest):
    t0 = time.time()
    try:
        digit, conf = inf.predict_from_base64(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    latency = (time.time() - t0) * 1000.0
    store.insert(digit, conf, latency)
    return {"digit": digit, "confidence": conf, "latency_ms": latency}
