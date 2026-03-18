# Registro mínimo en SQLite: predicción, confianza, timestamp.
import base64
import io
import sqlite3
import time
import uuid
from pathlib import Path
from PIL import Image

SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER,
  digit INTEGER,
  confidence REAL,
  latency_ms REAL
);
"""


class Store:
    def __init__(self, path='logs.db', samples_root='canvas_samples'):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()
        self.samples_root = Path(samples_root)
        self.samples_root.mkdir(parents=True, exist_ok=True)

    def insert(self, digit: int, conf: float, latency_ms: float):
        self.conn.execute(
            'INSERT INTO logs (ts, digit, confidence, latency_ms) VALUES (?, ?, ?, ?)',
            (int(time.time()), digit, conf, latency_ms)
        )
        self.conn.commit()

    def save_sample(self, image_b64: str, label: int) -> str:
        label_dir = self.samples_root / str(int(label))
        label_dir.mkdir(parents=True, exist_ok=True)

        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert('L').resize((28, 28))

        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        out_path = label_dir / filename
        img.save(out_path, format='PNG')
        return str(out_path)
