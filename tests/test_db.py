"""ทดสอบ SQLite job store: schema, CRUD, restart recovery"""

from triple_extraction.db import JobStore


def make_store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def test_connect_creates_db_with_wal_and_busy_timeout(tmp_path):
    store = make_store(tmp_path)

    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0


def test_create_job_returns_queued_job(tmp_path):
    store = make_store(tmp_path)

    job = store.create_job(text="hello world", meta={"source": "test"})

    assert job["id"]
    assert job["status"] == "queued"
    assert job["text"] == "hello world"
    assert job["meta"] == {"source": "test"}


def test_get_job_roundtrip_and_missing(tmp_path):
    store = make_store(tmp_path)
    job = store.create_job(text="hello", meta=None)

    got = store.get_job(job["id"])
    assert got is not None
    assert got["id"] == job["id"]
    assert got["text"] == "hello"
    assert got["meta"] is None
    assert got["triples"] is None

    assert store.get_job("no-such-id") is None


def test_complete_job_stores_triples_and_timing(tmp_path):
    store = make_store(tmp_path)
    job = store.create_job(text="hello", meta=None)

    store.complete_job(job["id"], triples=[{"head": "a", "relation": "r", "tail": "b"}],
                       timing={"total_ms": 12})

    got = store.get_job(job["id"])
    assert got["status"] == "done"
    assert got["triples"] == [{"head": "a", "relation": "r", "tail": "b"}]
    assert got["timing"] == {"total_ms": 12}


def test_fail_job_stores_error(tmp_path):
    store = make_store(tmp_path)
    job = store.create_job(text="hello", meta=None)

    store.fail_job(job["id"], error="boom")

    got = store.get_job(job["id"])
    assert got["status"] == "failed"
    assert got["error"] == "boom"


def test_claim_next_queued_marks_processing_in_order(tmp_path):
    store = make_store(tmp_path)
    first = store.create_job(text="a", meta=None)["id"]
    second = store.create_job(text="b", meta=None)["id"]

    claimed = store.claim_next_queued()
    assert claimed["id"] == first
    assert claimed["status"] == "processing"

    store.fail_job(first, error="x")
    assert store.claim_next_queued()["id"] == second
    assert store.claim_next_queued() is None


def test_recover_stale_jobs_marks_processing_as_failed(tmp_path):
    store = make_store(tmp_path)
    stuck = store.create_job(text="a", meta=None)["id"]
    done = store.create_job(text="b", meta=None)["id"]
    store._conn.execute("UPDATE jobs SET status = 'processing' WHERE id = ?", (stuck,))
    store.complete_job(done, triples=[], timing={})

    recovered = store.recover_stale_jobs()

    assert recovered == [stuck]
    assert store.get_job(stuck)["status"] == "failed"
    assert store.get_job(done)["status"] == "done"
