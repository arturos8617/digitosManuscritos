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
  mode TEXT,
  symbol TEXT,
  confidence REAL,
  latency_ms REAL
);
"""


class Store:
    def __init__(self, path='logs.db', samples_root='canvas_samples'):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self._migrate_logs_schema()
        self.conn.commit()
        self.samples_root = Path(samples_root)
        self.samples_root.mkdir(parents=True, exist_ok=True)

    def _migrate_logs_schema(self):
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(logs)")}
        if "mode" not in cols:
            self.conn.execute("ALTER TABLE logs ADD COLUMN mode TEXT DEFAULT 'digits'")
        if "symbol" not in cols:
            self.conn.execute("ALTER TABLE logs ADD COLUMN symbol TEXT")

    def insert(self, mode: str, symbol: str, conf: float, latency_ms: float):
        self.conn.execute(
            'INSERT INTO logs (ts, mode, symbol, confidence, latency_ms) VALUES (?, ?, ?, ?, ?)',
            (int(time.time()), mode, str(symbol), conf, latency_ms)
        )
        self.conn.commit()

    def save_sample(self, image_b64: str, mode: str | int, label: str | None = None) -> str:
        # compatibilidad hacia atrás: save_sample(image_b64, label_int)
        if label is None:
            label = str(mode)
            mode = "digits"
        else:
            label = str(label)
            mode = str(mode)

        if mode == "digits":
            # compatibilidad con flujo actual de entrenamiento (canvas_samples/0..9)
            label_dir = self.samples_root / str(label)
        else:
            label_dir = self.samples_root / mode / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert('L').resize((28, 28))

        filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        out_path = label_dir / filename
        img.save(out_path, format='PNG')
        return str(out_path)
