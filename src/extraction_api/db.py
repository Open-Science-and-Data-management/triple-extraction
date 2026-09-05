"""SQLite เก็บแค่ queue/track (id + status) — payload อยู่ที่ไฟล์ใน results.py"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  error TEXT,
  callback_url TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobDB:
    """ponytail: check_same_thread=False + lock เดียวพอ — ไม่มี pool/WAL"""

    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    @staticmethod
    def _to_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def enqueue(self, callback_url: str | None = None) -> str:
        import uuid

        job_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, callback_url, created_at) VALUES (?, ?, ?)",
                (job_id, callback_url, _now()),
            )
            self._conn.commit()
        return job_id

    def claim(self) -> dict[str, Any] | None:
        """pending → running — UPDATE มีเงื่อนไข status='pending' กัน claim ซ้ำข้าม thread"""
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE jobs SET status = 'running'
                WHERE id = (SELECT id FROM jobs WHERE status = 'pending'
                            ORDER BY created_at LIMIT 1)
                  AND status = 'pending'
                RETURNING *
                """
            )
            row = cur.fetchone()
            self._conn.commit()
        return self._to_job(row)

    def finish(
        self, job_id: str, status: str, error: str | None = None, finished_at: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, finished_at or _now(), job_id),
            )
            self._conn.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_job(row)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def prunable_done(self, older_than: str) -> list[str]:
        """id ของ done ที่ finished_at เก่ากว่า older_than (ISO string)"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status = 'done' AND finished_at < ?",
                (older_than,),
            ).fetchall()
        return [r["id"] for r in rows]

    def close(self) -> None:
        self._conn.close()
