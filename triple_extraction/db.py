"""SQLite job store: schema + CRUD + restart recovery

WAL + busy_timeout กัน lock ระหว่าง worker thread ↔ API thread
(check_same_thread=False + lock ต่อ connection เพราะ uvicorn/worker คนละ thread)
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'queued',
    text       TEXT NOT NULL,
    meta       TEXT,
    triples    TEXT,
    timing     TEXT,
    error      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class JobStore:
    """ครอบ connection เดียว + lock — thread-safe พอสำหรับ worker เดียว + API"""

    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def _run(self, fn):
        with self._lock:
            result = fn(self._conn)
            self._conn.commit()
            return result

    def create_job(self, text: str, meta: dict | None = None) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        job = {
            "id": uuid.uuid4().hex,
            "status": "queued",
            "text": text,
            "meta": meta,
        }

        def fn(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO jobs (id, status, text, meta, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (job["id"], job["status"], text, _dump(meta), now, now),
            )
            return job

        return self._run(fn)

    def get_job(self, job_id: str) -> dict | None:
        def fn(conn: sqlite3.Connection):
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return _row_to_job(row) if row else None

        return self._run(fn)

    def complete_job(self, job_id: str, triples: list, timing: dict) -> None:
        self._update(job_id, status="done", triples=_dump(triples), timing=_dump(timing))

    def fail_job(self, job_id: str, error: str) -> None:
        self._update(job_id, status="failed", error=error)

    def _update(self, job_id: str, **fields: str) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())

        def fn(conn: sqlite3.Connection):
            conn.execute(
                f"UPDATE jobs SET {sets}, updated_at = ? WHERE id = ?",
                [*values, datetime.now(timezone.utc).isoformat(), job_id],
            )

        self._run(fn)

    def claim_next_queued(self) -> dict | None:
        """ดึง job queued ตัวแรกและเปลี่ยนเป็น processing (สำหรับ worker)"""

        def fn(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), row[0]),
            )
            return row[0]

        job_id = self._run(fn)
        return self.get_job(job_id) if job_id else None

    def recover_stale_jobs(self) -> list[str]:
        """ตอนบูต: mark job ที่ค้าง processing จากรอบก่อนเป็น failed"""

        def fn(conn: sqlite3.Connection):
            now = datetime.now(timezone.utc).isoformat()
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status = 'processing'"
            ).fetchall()
            ids = [r[0] for r in rows]
            conn.execute(
                "UPDATE jobs SET status = 'failed',"
                " error = 'server restarted while processing',"
                " updated_at = ? WHERE status = 'processing'",
                (now,),
            )
            return ids

        return self._run(fn)


def _dump(value) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def _row_to_job(row: tuple) -> dict:
    keys = [
        "id", "status", "text", "meta", "triples", "timing",
        "error", "created_at", "updated_at",
    ]
    job = dict(zip(keys, row))
    for field in ("meta", "triples", "timing"):
        job[field] = json.loads(job[field]) if job[field] else None
    return job
