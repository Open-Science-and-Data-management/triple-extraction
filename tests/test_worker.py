"""ทดสอบ worker: job lifecycle queued→processing→done/failed + restart recovery"""

import time

from triple_extraction.db import JobStore
from triple_extraction.worker import Worker, process_next


class FakeExtractor:
    """mock extractor — คืน triples ตายตัว ไม่แตะโมเดล"""

    def __init__(self, triples=None, error=None):
        self.device = "cpu"
        # เลียน contract ของ RebelExtractor: ทุก triple มี extractor ฝังมาแล้ว
        self.triples = triples if triples is not None else [
            {"head": "a", "relation": "r", "tail": "b", "extractor": "rebel"}
        ]
        self.error = error

    def extract(self, text):
        if self.error:
            raise RuntimeError(self.error)
        return self.triples


def test_process_next_completes_job_with_triples_and_timing(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create_job(text="hello", meta=None)

    assert process_next(store, FakeExtractor()) is True

    got = store.get_job(job["id"])
    assert got["status"] == "done"
    assert got["triples"] == [{"head": "a", "relation": "r", "tail": "b",
                               "extractor": "rebel"}]
    assert "total_ms" in got["timing"]


def test_process_next_marks_failed_when_extractor_raises(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create_job(text="hello", meta=None)

    process_next(store, FakeExtractor(error="boom"))

    got = store.get_job(job["id"])
    assert got["status"] == "failed"
    assert "boom" in got["error"]


def test_process_next_returns_false_when_queue_empty(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    assert process_next(store, FakeExtractor()) is False


def test_worker_background_thread_processes_jobs_and_stops(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job = store.create_job(text="hello", meta=None)
    worker = Worker(store, FakeExtractor(), poll_interval=0.01)

    worker.start()
    deadline = time.monotonic() + 5
    while store.get_job(job["id"])["status"] == "queued" and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert store.get_job(job["id"])["status"] == "done"


def test_worker_recovers_stale_processing_on_start(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    stale = store.create_job(text="old", meta=None)["id"]
    store._conn.execute("UPDATE jobs SET status = 'processing' WHERE id = ?", (stale,))

    worker = Worker(store, FakeExtractor())
    worker.recover()

    assert store.get_job(stale)["status"] == "failed"
