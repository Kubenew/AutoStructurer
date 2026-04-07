import sqlite3
import json

class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            source TEXT,
            schema TEXT,
            t_start REAL,
            t_end REAL,
            text TEXT,
            entities_json TEXT,
            topic TEXT,
            contradiction REAL,
            confidence REAL,
            ref_path TEXT,
            phash TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            chunk_id TEXT,
            kind TEXT,
            dim INTEGER,
            packed BLOB,
            scale REAL,
            zero REAL,
            PRIMARY KEY(chunk_id, kind)
        )
        """)

        conn.commit()
        conn.close()

    def insert_chunk(self, chunk):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT OR REPLACE INTO chunks(
                chunk_id, doc_id, source, schema,
                t_start, t_end, text,
                entities_json, topic,
                contradiction, confidence, ref_path, phash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["chunk_id"],
            chunk["doc_id"],
            chunk["source"],
            chunk["schema"],
            float(chunk.get("t_start", 0.0)),
            float(chunk.get("t_end", 0.0)),
            chunk["text"],
            json.dumps(chunk["entities"], ensure_ascii=False),
            chunk["topic"],
            float(chunk["contradiction"]),
            float(chunk["confidence"]),
            chunk.get("ref_path"),
            chunk.get("phash")
        ))

        conn.commit()
        conn.close()

    def insert_vector(self, chunk_id, kind, dim, packed, scale, zero):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT OR REPLACE INTO vectors(chunk_id, kind, dim, packed, scale, zero)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (chunk_id, kind, int(dim), packed, float(scale), float(zero)))

        conn.commit()
        conn.close()

    def fetch_vectors(self, kind="text"):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, dim, packed, scale, zero FROM vectors WHERE kind=?", (kind,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def fetch_chunk(self, chunk_id):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_id, doc_id, source, schema, t_start, t_end, text,
                   entities_json, topic, contradiction, confidence, ref_path, phash
            FROM chunks WHERE chunk_id=?
        """, (chunk_id,))
        row = cur.fetchone()
        conn.close()
        return row
