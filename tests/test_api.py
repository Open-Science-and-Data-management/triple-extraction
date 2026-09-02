"""ทดสอบ API 3 endpoints ด้วย mock extractor + tmp SQLite"""

import time

from fastapi.testclient import TestClient

from triple_extraction.api import create_app
from triple_extraction.db import JobStore
from tests.test_worker import FakeExtractor


def make_client(tmp_path, extractor=None):
    store = JobStore(tmp_path / "jobs.db")
    ex = extractor if extractor is not None else FakeExtractor()
    app = create_app(store=store, extractor=ex, start_worker=True)
    return TestClient(app), store


def test_post_documents_returns_job_id_immediately(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        t0 = time.monotonic()
        res = client.post("/v1/documents", json={"text": "hello world"})
        elapsed = time.monotonic() - t0

    assert res.status_code == 200
    assert res.json()["job_id"]
    assert elapsed < 1  # ไม่บล็อกรอ inference


def test_post_documents_rejects_malformed_body(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        assert client.post("/v1/documents", json={"no_text": "x"}).status_code == 422
        assert client.post("/v1/documents", content=b"not json").status_code == 422


def test_post_documents_rejects_text_over_limit(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        res = client.post("/v1/documents", json={"text": "a" * 500_001})

    assert res.status_code == 413


def test_get_job_returns_triples_when_done(tmp_path):
    client, store = make_client(tmp_path)
    with client:
        job_id = client.post("/v1/documents", json={"text": "hello"}).json()["job_id"]

        deadline = time.monotonic() + 5
        while (store.get_job(job_id)["status"] != "done"
               and time.monotonic() < deadline):
            time.sleep(0.01)

        res = client.get(f"/v1/jobs/{job_id}")

    body = res.json()
    assert res.status_code == 200
    assert body["status"] == "done"
    assert body["triples"] == [{"head": "a", "relation": "r", "tail": "b",
                                "extractor": "rebel"}]
    assert "total_ms" in body["timing"]
    assert body["triples"][0]["extractor"] == "rebel"


def test_get_job_processing_has_no_triples(tmp_path):
    client, store = make_client(tmp_path)
    with client:
        job_id = client.post("/v1/documents", json={"text": "hello"}).json()["job_id"]
        # ยังไม่ให้ worker เริ่ม — บังคับ status ค้าง queued/processing
        store._conn.execute(
            "UPDATE jobs SET status = 'processing' WHERE id = ?", (job_id,))

        res = client.get(f"/v1/jobs/{job_id}")

    body = res.json()
    assert res.status_code == 200
    assert body["status"] == "processing"
    assert body["triples"] is None


def test_get_missing_job_returns_404(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        res = client.get("/v1/jobs/no-such-id")

    assert res.status_code == 404


def test_health_reports_device(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        res = client.get("/v1/health")

    body = res.json()
    assert res.status_code == 200
    assert body["status"] == "ok"
    assert body["device"] in ("cuda", "cpu")


def test_restart_recovers_done_jobs_and_fails_stale(tmp_path):
    # db คือแหล่งความจริง — app ใหม่บน db เดิมต้องอ่าน job เก่าได้
    store = JobStore(tmp_path / "jobs.db")
    app = create_app(store=store, extractor=FakeExtractor(), start_worker=False)
    with TestClient(app) as client:
        job_id = client.post("/v1/documents", json={"text": "hello"}).json()["job_id"]
    store.complete_job(job_id, triples=[{"head": "a", "relation": "r", "tail": "b",
                                         "extractor": "rebel"}],
                       timing={"total_ms": 1})
    stuck = store.create_job(text="stale", meta=None)["id"]
    store._conn.execute("UPDATE jobs SET status = 'processing' WHERE id = ?", (stuck,))

    app2 = create_app(store=store, extractor=FakeExtractor(), start_worker=False)
    with TestClient(app2) as client:
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "done"
        assert client.get(f"/v1/jobs/{stuck}").json()["status"] == "failed"
