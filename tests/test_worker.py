"""worker — claim→extract→done/failed, webhook on/off, resume, prune หลัง job เสร็จ"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx

from extraction_api.db import JobDB
from extraction_api.results import read_result
from extraction_api.settings import Settings
from extraction_api.worker import process_pending, send_webhook


def make_settings(tmp_path, **kw) -> Settings:
    return Settings(
        db_path=tmp_path / "jobs.db",
        results_dir=tmp_path / "results",
        retention_days=kw.pop("retention_days", 7),
        max_results_mb=kw.pop("max_results_mb", 500),
        **kw,
    )


def fake_extract(documents, schema):
    """stand-in ของ extract_raw — คืน triple ครบ provenance 1 อันต่อ document"""
    return [
        {
            "source_file": i,
            "field": doc["field"],
            "sentence": "LoRA reduces hallucination.",
            "head": "LoRA",
            "head_type": "method",
            "tail": "hallucination",
            "tail_type": "concept",
            "relation": "reduces",
            "score": 0.93,
        }
        for i, doc in enumerate(documents)
    ]


def failing_extract(documents, schema):
    raise RuntimeError("GPU หลุด")


# --- lifecycle ---
def test_process_pending_done_writes_payload_with_provenance(tmp_path):
    settings = make_settings(tmp_path)
    db = JobDB(settings.db_path)
    jid = db.enqueue(
        seed_relations=["reduces"],
        documents=[{"field": "text", "content": "LoRA reduces hallucination."}],
    )
    n = process_pending(db, fake_extract, settings)
    assert n == 1
    row = db.get(jid)
    assert row["status"] == "done" and row["finished_at"]
    payload = read_result(settings.results_dir, jid)
    assert payload["job_id"] == jid
    assert payload["model_version"] == "knowledgator/gliner-relex-multi-v1.0"
    assert payload["seed_schema_hash"].startswith("sha256:")
    t = payload["triples"][0]
    assert t["score"] == 0.93 and t["head_type"] == "method"


def test_failed_path_writes_error_and_no_result_file(tmp_path):
    settings = make_settings(tmp_path)
    db = JobDB(settings.db_path)
    jid = db.enqueue()
    process_pending(db, failing_extract, settings)
    row = db.get(jid)
    assert row["status"] == "failed" and "GPU หลุด" in row["error"]
    assert read_result(settings.results_dir, jid) is None


# --- webhook ---
def test_webhook_enabled_posts_job_id_and_status(tmp_path):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    send_webhook("http://cb.example/hook", {"job_id": "j1", "status": "done"}, transport=transport)
    assert sent == [("http://cb.example/hook", {"job_id": "j1", "status": "done"})]


def test_webhook_error_does_not_break_status(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    # พังก็ไม่ raise — webhook ห้ามกระทบ status ของ job
    send_webhook(
        "http://cb.example/hook", {"job_id": "j1", "status": "done"}, transport=httpx.MockTransport(handler)
    )


def test_worker_fires_webhook_on_done_and_not_when_disabled(tmp_path):
    settings = make_settings(tmp_path)
    db = JobDB(settings.db_path)
    sent = []
    handler_calls = sent

    def handler(request: httpx.Request) -> httpx.Response:
        handler_calls.append(json.loads(request.content))
        return httpx.Response(200)

    jid = db.enqueue(callback_url="http://cb.example/hook")
    process_pending(db, fake_extract, settings, transport=httpx.MockTransport(handler))
    assert handler_calls == [{"job_id": jid, "status": "done"}]

    # disabled → ไม่ยิง
    settings_off = make_settings(tmp_path, webhook_enabled=False)
    db2 = JobDB(tmp_path / "off.db")
    db2.enqueue(callback_url="http://cb.example/hook")
    process_pending(db2, fake_extract, settings_off, transport=httpx.MockTransport(handler))
    assert len(handler_calls) == 1  # ไม่เพิ่ม


# --- resume + prune ---
def test_pending_resumes_after_restart(tmp_path):
    settings = make_settings(tmp_path)
    db = JobDB(settings.db_path)
    jid = db.enqueue()
    db.close()  # จำลอง process ตาย
    db2 = JobDB(settings.db_path)  # worker ใหม่เปิด DB เดิม
    assert process_pending(db2, fake_extract, settings) == 1
    assert db2.get(jid)["status"] == "done"


def test_prune_runs_after_each_job(tmp_path):
    settings = make_settings(tmp_path, retention_days=7)
    db = JobDB(settings.db_path)
    # ผล done เก่าวางไว้ก่อน — TTL หมดอายุแล้ว
    old = db.enqueue()
    db.claim()
    old_finished = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    db.finish(old, status="done", finished_at=old_finished)
    old_path = settings.results_dir / f"{old}.json"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text(json.dumps({"job_id": old, "triples": []}))

    jid = db.enqueue()
    process_pending(db, fake_extract, settings)
    assert not old_path.exists()  # ผลเก่าถูก prune หลัง job ใหม่เสร็จ
    assert db.get(old) is None  # แถวใน DB ถูกลบด้วย
    assert db.get(jid)["status"] == "done"
    assert read_result(settings.results_dir, jid) is not None  # ผลใหม่อยู่ครบ
