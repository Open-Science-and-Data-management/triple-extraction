"""Worker loop — claim → extract → เขียนผล atomic → done/failed → webhook → prune"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from extraction_api import extractor as ex
from extraction_api.db import JobDB
from extraction_api.results import prune, write_result
from extraction_api.settings import Settings

logger = logging.getLogger(__name__)

ExtractFn = Callable[[list[dict[str, str]], dict[str, Any]], list[dict[str, Any]]]


def send_webhook(
    callback_url: str,
    data: dict[str, str],
    transport: httpx.BaseTransport | None = None,
) -> None:
    """ยิง job_id+status เท่านั้น — ห้ามแนบ raw text; พังแล้วไม่กระทบ status"""
    if transport is not None:  # inject จาก test (MockTransport)
        client = httpx.Client(transport=transport)
    else:
        client = httpx.Client(timeout=Settings().callback_timeout)
    try:
        client.post(callback_url, json=data)
    except Exception:  # noqa: BLE001 — webhook พังได้ทุกแบบ ห้ามกระทบ status ของ job
        logger.warning("webhook พัง (ไม่กระทบ job): %s", callback_url)
    finally:
        client.close()


def _job_schema(job: dict[str, Any]) -> dict[str, Any]:
    """default จาก seed.json — seed_relations ต่อ job ทับ relation_hints ได้"""
    schema = ex.load_seed_schema()
    if job.get("seed_relations"):
        schema = {**schema, "relation_hints": json.loads(job["seed_relations"])}
    return schema


def run_job(
    db: JobDB,
    job: dict[str, Any],
    extract: ExtractFn,
    settings: Settings,
    transport: httpx.BaseTransport | None = None,
) -> None:
    try:
        schema = _job_schema(job)
        documents = json.loads(job["documents"]) if job.get("documents") else []
        triples = extract(documents, schema)
        write_result(
            settings.results_dir,
            job["id"],
            {
                "job_id": job["id"],
                "model_version": ex.MODEL_VERSION,
                "seed_schema_hash": ex.schema_hash(schema),
                "documents": documents,
                "triples": triples,
            },
        )
        db.finish(job["id"], status="done")
        status = "done"
    except Exception as e:  # noqa: BLE001 — job ไหนพัง job นั้น failed แล้ววนต่อ
        db.finish(job["id"], status="failed", error=str(e))
        status = "failed"

    if settings.webhook_enabled and job.get("callback_url"):
        send_webhook(job["callback_url"], {"job_id": job["id"], "status": status}, transport=transport)


def _prune_old(db: JobDB, settings: Settings) -> None:
    """วิ่งหลัง job แต่ละ job เสร็จ (ไม่มี timer thread) — ลบได้เฉพาะ done เสมอ"""
    now = datetime.now(UTC).isoformat()
    removed = prune(
        settings.results_dir,
        db.list_done(),
        now=now,
        retention_days=settings.retention_days,
        max_results_mb=settings.max_results_mb,
    )
    for jid in removed:
        db.delete(jid)


def process_pending(
    db: JobDB,
    extract: ExtractFn,
    settings: Settings,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """ทำ pending จนหมด — คืนจำนวน job ที่ทำ"""
    n = 0
    while (job := db.claim()) is not None:
        run_job(db, job, extract, settings, transport=transport)
        _prune_old(db, settings)
        n += 1
    return n


def start_worker(
    db: JobDB,
    extract: ExtractFn,
    settings: Settings,
    stop: threading.Event,
    poll_interval: float = 0.5,
) -> threading.Thread:
    """daemon thread — pending ค้างจากรอบก่อนถูก claim ต่อเอง (startup resume)"""

    def loop() -> None:
        while not stop.is_set():
            if process_pending(db, extract, settings) == 0:
                stop.wait(poll_interval)

    t = threading.Thread(target=loop, daemon=True, name="extraction-worker")
    t.start()
    return t
