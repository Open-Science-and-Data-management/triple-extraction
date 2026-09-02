"""FastAPI app: 3 endpoints — POST /v1/documents, GET /v1/jobs/{id}, GET /v1/health

worker เป็น background thread ใน process เดียวกับ uvicorn, extractor inject ได้เพื่อ test
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from triple_extraction.db import JobStore
from triple_extraction.extractor.rebel import RebelExtractor
from triple_extraction.settings import DB_PATH, MAX_TEXT_CHARS
from triple_extraction.worker import Worker


class DocumentIn(BaseModel):
    text: str
    meta: dict | None = None


class JobOut(BaseModel):
    id: str
    status: str
    triples: list[dict] | None
    timing: dict | None
    error: str | None = None


def create_app(store: JobStore | None = None, extractor=None,
               start_worker: bool = True) -> FastAPI:
    store = store or _default_store()
    extractor = extractor or RebelExtractor()
    worker = Worker(store, extractor)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"device: {extractor.device}", flush=True)
        worker.recover()  # ตอนบูต: mark job ค้าง processing เป็น failed เสมอ
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/documents")
    def create_document(doc: DocumentIn):
        if len(doc.text) > MAX_TEXT_CHARS:
            raise HTTPException(413, f"text exceeds {MAX_TEXT_CHARS} characters")
        job = store.create_job(text=doc.text, meta=doc.meta)
        return {"job_id": job["id"]}

    @app.get("/v1/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "device": extractor.device}

    return app


def _default_store() -> JobStore:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return JobStore(DB_PATH)


app = create_app()
