# Registro mínimo en SQLite: predicción, confianza, timestamp.
import sqlite3, time

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
    def __init__(self, path='logs.db'):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def insert(self, digit: int, conf: float, latency_ms: float):
        self.conn.execute(
            'INSERT INTO logs (ts, digit, confidence, latency_ms) VALUES (?, ?, ?, ?)',
            (int(time.time()), digit, conf, latency_ms)
        )
        self.conn.commit()
