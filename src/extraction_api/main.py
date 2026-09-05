"""FastAPI app — lifespan สร้าง DB, extractor inject ผ่าน app.state (test แทน Fake ได้)"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from extraction_api.db import JobDB
from extraction_api.results import filter_triples, read_result, result_path
from extraction_api.schemas import (
    CreateJobRequest,
    JobStatusResponse,
    TriplesResponse,
    validate_job_request,
)
from extraction_api.settings import Settings
from extraction_api.worker import ExtractFn, start_worker


def get_db(request: Request) -> JobDB:
    return request.app.state.db


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


# module-level เพราะ __future__ annotations — FastAPI resolve ชื่อผ่าน globals เท่านั้น
DbDep = Annotated[JobDB, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def make_extract_fn() -> ExtractFn:
    """โหลด model จาก models/ ครั้งเดียวตอน startup — คืน closure ที่ worker เรียกได้"""
    from extraction_api.extractor import extract_raw, load_extractor

    model = load_extractor()

    def extract(documents: list[dict[str, str]], schema: dict[str, Any]) -> list[dict[str, Any]]:
        return extract_raw(model, documents, schema)

    return extract


def create_app(settings: Settings | None = None, spawn_worker: bool = True) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.db = JobDB(settings.db_path)
        app.state.settings = settings
        if spawn_worker:
            stop = threading.Event()
            app.state.worker = start_worker(app.state.db, make_extract_fn(), settings, stop)
            app.state.worker_stop = stop
        yield
        if spawn_worker:
            app.state.worker_stop.set()
        app.state.db.close()

    app = FastAPI(title="extraction-api", lifespan=lifespan)

    @app.post("/jobs", status_code=201)
    def create_job(req: CreateJobRequest, db: DbDep, settings: SettingsDep) -> dict[str, str]:
        try:
            validate_job_request(req, settings.max_files, settings.max_bytes)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        documents = [{"field": d.field, "content": d.content, "section": d.section} for d in req.documents]
        return {
            "job_id": db.enqueue(
                callback_url=req.callback_url,
                seed_relations=req.seed_relations,
                documents=documents,
            )
        }

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def job_status(job_id: str, db: DbDep) -> dict[str, Any]:
        job = db.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบ job")
        return {"job_id": job["id"], "status": job["status"], "error": job["error"]}

    @app.get("/jobs/{job_id}/triples", response_model=TriplesResponse)
    def job_triples(
        job_id: str,
        db: DbDep,
        settings: SettingsDep,
        threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    ) -> dict[str, Any]:
        job = db.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบ job")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail=f"job ยังไม่ done (status={job['status']})")
        payload = read_result(settings.results_dir, job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ผล")
        t = threshold if threshold is not None else settings.default_threshold
        return {**payload, "threshold": t, "triples": filter_triples(payload, t)}

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, db: DbDep, settings: SettingsDep) -> None:
        if db.get(job_id) is None:
            raise HTTPException(status_code=404, detail="ไม่พบ job")
        db.delete(job_id)
        result_path(settings.results_dir, job_id).unlink(missing_ok=True)

    return app


app = create_app()
