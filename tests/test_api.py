"""API endpoints — validation ที่ trust boundary + lifecycle status + threshold slice"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from extraction_api.db import JobDB
from extraction_api.main import create_app
from extraction_api.results import write_result
from extraction_api.settings import Settings


def make_settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "jobs.db",
        results_dir=tmp_path / "results",
        max_files=2,
        max_bytes=1000,
        default_threshold=0.9,
    )


def body(doc=None, **kw):
    return {"documents": doc or [{"field": "text", "content": "LoRA reduces hallucination."}], **kw}


@pytest.fixture()
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, spawn_worker=False)  # model-free — worker ไม่ spawn
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, app, settings


def seed_done(settings, triples=None):
    """วาง job done + ไฟล์ผลไว้ใน storage ตรง ๆ (inference เป็นเรื่องของ worker)"""
    app_db = JobDB(settings.db_path)
    jid = app_db.enqueue()
    app_db.claim()
    payload = {
        "job_id": jid,
        "model_version": "knowledgator/gliner-relex-multi-v1.0",
        "seed_schema_hash": "sha256:abc",
        "documents": [],
        "triples": triples
        or [
            {
                "source_file": 0,
                "field": "text",
                "sentence": "s",
                "head": "a",
                "head_type": "m",
                "tail": "b",
                "tail_type": "c",
                "relation": "r",
                "score": 0.95,
            },
            {
                "source_file": 0,
                "field": "text",
                "sentence": "s",
                "head": "c",
                "head_type": "m",
                "tail": "d",
                "tail_type": "c",
                "relation": "r",
                "score": 0.55,
            },
        ],
    }
    write_result(settings.results_dir, jid, payload)
    app_db.finish(jid, status="done", finished_at="2026-09-05T00:00:00")
    app_db.close()
    return jid


# --- POST /jobs ---
def test_post_valid_returns_job_id_immediately(client):
    c, _app, settings = client
    r = c.post("/jobs", json=body(callback_url="http://x/cb"))
    assert r.status_code in (200, 201)
    jid = r.json()["job_id"]
    assert JobDB(settings.db_path).get(jid)["status"] == "pending"


def test_post_invalid_field_is_422(client):
    c, _, _ = client
    r = c.post("/jobs", json=body(doc=[{"field": "table", "content": "x"}]))
    assert r.status_code == 422


def test_post_empty_content_is_422(client):
    c, _, _ = client
    r = c.post("/jobs", json=body(doc=[{"field": "text", "content": ""}]))
    assert r.status_code == 422


def test_post_over_limits_is_422_with_reason(client):
    c, _, _ = client
    docs = [{"field": "text", "content": "x"}] * 3  # > max_files=2
    r = c.post("/jobs", json=body(doc=docs))
    assert r.status_code == 422
    assert "MAX_FILES" in r.text


# --- GET /jobs/{id} ---
def test_get_job_unknown_is_404(client):
    c, _, _ = client
    assert c.get("/jobs/nope").status_code == 404


def test_get_job_covers_all_four_statuses(client):
    c, _app, settings = client
    db = JobDB(settings.db_path)
    p = c.post("/jobs", json=body()).json()["job_id"]
    assert c.get(f"/jobs/{p}").json()["status"] == "pending"
    db.claim()
    assert c.get(f"/jobs/{p}").json()["status"] == "running"
    db.finish(p, status="done", finished_at="2026-09-05T00:00:00")
    assert c.get(f"/jobs/{p}").json()["status"] == "done"
    f = c.post("/jobs", json=body()).json()["job_id"]
    db.finish(f, status="failed", error="boom", finished_at="2026-09-05T00:00:00")
    js = c.get(f"/jobs/{f}").json()
    assert js["status"] == "failed" and js["error"] == "boom"


# --- GET /jobs/{id}/triples ---
def test_triples_unknown_404_pending_409(client):
    c, _, _ = client
    assert c.get("/jobs/nope/triples").status_code == 404
    _c2, _, settings = client
    db = JobDB(settings.db_path)
    jid = db.enqueue()  # pending
    assert c.get(f"/jobs/{jid}/triples").status_code == 409


def test_triples_threshold_slice_from_same_file(client):
    c, _, settings = client
    jid = seed_done(settings)
    hi = c.get(f"/jobs/{jid}/triples").json()  # default 0.9
    lo = c.get(f"/jobs/{jid}/triples", params={"threshold": 0.5}).json()
    zero = c.get(f"/jobs/{jid}/triples", params={"threshold": 0.0}).json()
    assert hi["threshold"] == 0.9 and len(hi["triples"]) == 1
    assert lo["threshold"] == 0.5 and len(lo["triples"]) == 2
    assert len(zero["triples"]) == 2
    # ทุก triple ที่คืน score >= threshold เสมอ
    assert all(t["score"] >= 0.9 for t in hi["triples"])
    # provenance ครบ
    t = hi["triples"][0]
    for k in (
        "source_file",
        "field",
        "sentence",
        "head",
        "head_type",
        "tail",
        "tail_type",
        "relation",
        "score",
    ):
        assert k in t


# --- DELETE /jobs/{id} ---
def test_delete_removes_job_and_result(client):
    c, _, settings = client
    jid = seed_done(settings)
    assert c.delete(f"/jobs/{jid}").status_code == 204
    assert c.get(f"/jobs/{jid}").status_code == 404
    assert c.get(f"/jobs/{jid}/triples").status_code == 404  # ไฟล์ผลถูกลบด้วย
    assert c.delete(f"/jobs/{jid}").status_code == 404
