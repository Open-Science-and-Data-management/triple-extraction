"""Background worker: ดึง job จาก SQLite → extract → บันทึกผล

extractor inject ได้ (mock ใน test) — โมเดลจริง lazy-load ใน thread นี้ครั้งเดียว
"""

import threading
import time
from typing import Protocol


class _Extractor(Protocol):
    def extract(self, text: str) -> list[dict]: ...


def process_next(store, extractor: _Extractor) -> bool:
    """ทำงาน 1 job จาก queue; คืน False ถ้า queue ว่าง"""

    job = store.claim_next_queued()
    if job is None:
        return False

    t0 = time.perf_counter()
    try:
        triples = extractor.extract(job["text"])
    except Exception as exc:  # ตายกลาง extract → mark failed ไม่ปล่อยค้าง processing
        store.fail_job(job["id"], error=f"{type(exc).__name__}: {exc}")
        return True
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    store.complete_job(job["id"], triples=triples, timing={"total_ms": elapsed_ms})
    return True


class Worker:
    """รัน process_next เป็น loop ใน background thread"""

    def __init__(self, store, extractor: _Extractor, poll_interval: float = 0.5):
        self.store = store
        self.extractor = extractor
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def recover(self) -> list[str]:
        """ตอนบูต: mark job ที่ค้าง processing จากรอบก่อนเป็น failed"""
        return self.store.recover_stale_jobs()

    def start(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not process_next(self.store, self.extractor):
                self._stop.wait(self.poll_interval)
