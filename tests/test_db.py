"""db.py — jobs table: enqueue/claim/finish/get/delete + claim กันข้าม thread"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from extraction_api.db import JobDB


def test_enqueue_and_get_roundtrip(tmp_path):
    db = JobDB(tmp_path / "jobs.db")
    job_id = db.enqueue(callback_url="http://x/cb")
    row = db.get(job_id)
    assert row["status"] == "pending"
    assert row["callback_url"] == "http://x/cb"
    assert row["error"] is None and row["finished_at"] is None
    assert row["created_at"]


def test_claim_moves_pending_to_running(tmp_path):
    db = JobDB(tmp_path / "jobs.db")
    jid = db.enqueue()
    claimed = db.claim()
    assert claimed["id"] == jid
    assert claimed["status"] == "running"
    assert db.claim() is None  # ไม่มี pending เหลือ


def test_seed_relations_persist_through_claim(tmp_path):
    import json

    db = JobDB(tmp_path / "jobs.db")
    db.enqueue(seed_relations=["reduces", "improves"])
    claimed = db.claim()
    assert json.loads(claimed["seed_relations"]) == ["reduces", "improves"]


def test_claim_from_two_threads_yields_each_job_once(tmp_path):
    db = JobDB(tmp_path / "jobs.db")
    ids = {db.enqueue() for _ in range(6)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: db.claim(), range(20)))
    got = [c["id"] for c in claimed if c]
    # ทุก job ถูก claim คนละครั้งพอดี — ไม่มี claim ซ้ำ ไม่มีหลุด
    assert sorted(got) == sorted(ids)
    assert len(got) == len(set(got))


def test_finish_sets_status_error_finished_at(tmp_path):
    db = JobDB(tmp_path / "jobs.db")
    jid = db.enqueue()
    db.claim()
    db.finish(jid, status="done", finished_at="2026-09-05T00:00:00")
    assert db.get(jid)["status"] == "done"
    db.finish(jid, status="failed", error="boom", finished_at="2026-09-05T01:00:00")
    row = db.get(jid)
    assert row["status"] == "failed" and row["error"] == "boom"


def test_delete(tmp_path):
    db = JobDB(tmp_path / "jobs.db")
    jid = db.enqueue()
    db.delete(jid)
    assert db.get(jid) is None


def test_restart_keeps_pending(tmp_path):
    path = tmp_path / "jobs.db"
    db = JobDB(path)
    jid = db.enqueue()
    db.close()
    db2 = JobDB(path)  # เปิด connection ใหม่ = restart
    assert db2.get(jid)["status"] == "pending"
    assert db2.claim()["id"] == jid  # resume ต่อได้


def test_prunable_done_returns_old_done_only(tmp_path):
    db = JobDB(tmp_path / "jobs.db")
    old = db.enqueue()
    db.claim()
    db.finish(old, status="done", finished_at="2026-08-01T00:00:00")
    fresh = db.enqueue()
    db.claim()
    db.finish(fresh, status="done", finished_at="2026-09-05T00:00:00")
    failed = db.enqueue()
    db.claim()
    db.finish(failed, status="failed", finished_at="2026-07-01T00:00:00")
    assert db.prunable_done("2026-09-01T00:00:00") == [old]
