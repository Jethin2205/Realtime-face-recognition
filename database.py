import sqlite3, numpy as np, csv, io
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/face_recognition.db")

def _enc(arr):
    b = io.BytesIO(); np.save(b, arr); return b.getvalue()

def _dec(blob):
    return np.load(io.BytesIO(blob))

class DatabaseManager:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _conn(self):
        c = sqlite3.connect(str(self.path))
        c.row_factory = sqlite3.Row
        return c

    def _setup(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age TEXT, gender TEXT,
                    face_encoding BLOB NOT NULL,
                    registered_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    timestamp TEXT NOT NULL,
                    emotion TEXT, confidence REAL,
                    FOREIGN KEY (person_id) REFERENCES persons(id)
                );
            """)

    def add_person(self, name, age, gender, encoding):
        with self._conn() as c:
            return c.execute(
                "INSERT INTO persons (name,age,gender,face_encoding) VALUES (?,?,?,?)",
                (name, age, gender, _enc(encoding))
            ).lastrowid

    def get_all_persons(self):
        with self._conn() as c:
            rows = c.execute("SELECT id,name,age,gender,face_encoding FROM persons").fetchall()
        return [{"id":r["id"],"name":r["name"],"age":r["age"],
                 "gender":r["gender"],"face_encoding":_dec(r["face_encoding"])} for r in rows]

    def add_log(self, person_id, emotion, confidence=0.0):
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._conn() as c:
            return c.execute(
                "INSERT INTO logs (person_id,timestamp,emotion,confidence) VALUES (?,?,?,?)",
                (person_id, ts, emotion, confidence)
            ).lastrowid

    def get_logs(self, limit=100):
        with self._conn() as c:
            rows = c.execute("""
                SELECT l.log_id, COALESCE(p.name,'Unknown') AS name,
                       l.timestamp, l.emotion, l.confidence
                FROM logs l LEFT JOIN persons p ON l.person_id=p.id
                ORDER BY l.log_id DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, path="data/logs_export.csv"):
        logs = self.get_logs(100000)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["log_id","name","timestamp","emotion","confidence"])
            w.writeheader(); w.writerows(logs)
        print(f"Exported {len(logs)} rows → {path}")
        return path

    def stats(self):
        with self._conn() as c:
            p = c.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            l = c.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        return {"persons": p, "logs": l}
