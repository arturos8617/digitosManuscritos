import base64, io, numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.server.server import app

client = TestClient(app)

def test_predict_endpoint():
    # Imagen blanca con un punto negro (no tiene sentido semántico, solo smoke test)
    img = Image.new('L', (28,28), 255)
    img.putpixel((14,14), 0)
    buf = io.BytesIO(); img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    r = client.post('/predict', json={'image_b64': b64})
    assert r.status_code == 200
    j = r.json()
    assert 'digit' in j and 'confidence' in j and 'latency_ms' in j
