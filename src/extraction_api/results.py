"""ไฟล์ผลต่อ job ที่ data/results/{job_id}.json — เขียน atomic, อ่าน filter, prune

SQLite เก็บแค่ queue/track — payload อยู่ที่ไฟล์ (ดู spec หัวข้อ "ทำไมไฟล์ ไม่ใช่ DB")
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path


def result_path(results_dir: Path, job_id: str) -> Path:
    return results_dir / f"{job_id}.json"


def write_result(results_dir: Path, job_id: str, payload: dict) -> None:
    """เขียนแบบ atomic (.tmp → os.replace) — ผู้อ่านไม่เจอไฟล์ครึ่ง ๆ"""
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = result_path(results_dir, job_id).with_suffix(".json.tmp")
    tmp.write_text(__import__("json").dumps(payload, ensure_ascii=False))
    os.replace(tmp, result_path(results_dir, job_id))


def read_result(results_dir: Path, job_id: str) -> dict | None:
    path = result_path(results_dir, job_id)
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def filter_triples(payload: dict, threshold: float) -> list[dict]:
    """rate once, slice many — กรองจากไฟล์ผลเดิม ไม่ rerun"""
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise TypeError("payload ไม่มี triples เป็น list")
    return [t for t in triples if t["score"] >= threshold]


def prune(
    results_dir: Path,
    jobs: list[dict],
    now: str,
    retention_days: int,
    max_results_mb: int,
) -> list[str]:
    """ลบไฟล์ผลของ done เท่านั้น — คืน id ที่ลบแล้วให้ caller ไปลบแถวใน DB ต่อ

    TTL: finished_at เก่ากว่า retention_days · ขนาด: เกินเพดานลบ done เก่าสุดก่อน
    (finished_at เป็น ISO string เรียงตรงตัวเลขได้เลย — ไม่ parse)
    """
    done = [j for j in jobs if j.get("status") == "done"]
    removed: set[str] = set()

    if retention_days > 0:
        cutoff = (datetime.fromisoformat(now) - timedelta(days=retention_days)).isoformat()
        for j in done:
            if j.get("finished_at") and j["finished_at"] < cutoff:
                removed.add(j["id"])

    candidates = sorted(
        (j for j in done if j["id"] not in removed),
        key=lambda j: (j.get("finished_at") is None, j.get("finished_at") or ""),
    )
    total = sum(f.stat().st_size for f in results_dir.glob("*.json")) if results_dir.exists() else 0
    # 0 = ปิด cap ขนาด (สม่ำเสมอกับ RETENTION_DAYS=0)
    ceiling = max_results_mb * 1024 * 1024 if max_results_mb > 0 else float("inf")
    for j in candidates:
        if total <= ceiling:
            break
        path = result_path(results_dir, j["id"])
        if path.exists():
            total -= path.stat().st_size
        removed.add(j["id"])

    for job_id in removed:
        result_path(results_dir, job_id).unlink(missing_ok=True)
    return sorted(removed)
