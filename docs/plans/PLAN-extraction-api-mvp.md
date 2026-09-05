# Implementation Plan: Extraction API MVP — GLiNER-Relex as a Service

อ้างอิง spec: `docs/specs/SPEC-extraction-api-mvp.md` (Decision Log ตกลงครบแล้ว — แผนนี้แค่สลายเป็น task)

## Overview

Async job API: POST papers → worker thread รัน GLiNER-Relex บน CUDA @ threshold 0.3 (rate once) → เก็บ raw triples ลงไฟล์ JSON ต่อ job → GET /triples slice ด้วย threshold ใดก็ได้ (slice many, ไม่ rerun) SQLite เก็บแค่ queue/track, retention ลบได้เฉพาะ `done`

## Architecture Decisions (สรุปจาก spec — ไม่เปิดประเด็นใหม่)

- uv project ใหม่ที่ root, `src/extraction_api/` — ไม่แตะ `bake-off/`
- pins ชุดเดียวกับ bake-off group `gliner` (transformers==4.52.4, huggingface-hub<1.0, loguru, sentencepiece, protobuf, tiktoken) + torch cu128 index, pysbd, FastAPI, pydantic-settings, httpx, pytest, ruff
- model อยู่ `models/gliner-relex-multi-v1.0/` — `from_pretrained` อ่าน path นี้เสมอ, `HF_HUB_OFFLINE=1` ตอน runtime
- SQLite = queue, ไฟล์ JSON = payload — ตามหัวข้อ "ทำไมไฟล์ ไม่ใช่ DB" ใน spec
- extractor เป็น callable ที่ inject เข้า app ผ่าน FastAPI dependency → test ทั้งหมด (ยกเว้น `-m gpu`) ใช้ FakeExtractor ไม่แตะ GPU

## Task List

### Phase 1: Foundation (scaffold + model, ยังไม่มี API)

- [ ] **Task 1: uv scaffold + settings** — `pyproject.toml` (root, deps ตามด้านบน, `[tool.pytest.ini_options] markers = ["gpu"]`, `[tool.ruff]`), `.gitignore` (`.env`, `data/`, `models/`), `.env.example` (ทุกตัวแปรใน spec พร้อม comment ไทย), `settings.py` (pydantic-settings, default ครบตามตาราง .env), `__init__.py`
  - **Acceptance:**
    - [ ] `uv sync` สำเร็จ, lock ได้ `uv.lock`
    - [ ] `Settings()` ไม่มี `.env` ก็ได้ default ครบทุก field
  - **Verification:** `uv run python -c "from extraction_api.settings import Settings; print(Settings())"` (ตั้ง PYTHONPATH=src ผ่าน `[tool.uv]` หรือ src-layout ของ uv) · `uv run ruff check src`
  - **Dependencies:** None · **Files:** `pyproject.toml`, `.gitignore`, `.env.example`, `src/extraction_api/{__init__,settings}.py` · **Scope: S**

- [ ] **Task 2: download_model** — `download_model.py`: `snapshot_download("knowledgator/gliner-relex-multi-v1.0", local_dir=Settings().MODEL_DIR)` — idempotent (มีแล้วข้าม), ตรวจไฟล์จำเป็นครบ (config.json, model.safetensors, pytorch_model.bin อย่างน้อยหนึ่ง)
  - **Acceptance:**
    - [ ] `uv run python -m extraction_api.download_model` ได้ `models/gliner-relex-multi-v1.0/` ครบ checkpoint
    - [ ] รันซ้ำไม่ re-download
  - **Dependencies:** Task 1 · **Files:** `src/extraction_api/download_model.py` · **Scope: S**

- [ ] **Checkpoint 1:** `uv sync` + `ruff check` เขียว, model ลง `models/` แล้ว (ไม่มี test ยัง — ยังไม่มี logic ที่ผิดได้)

### Phase 2: Core logic model-free (results, db, extractor shell)

- [ ] **Task 3: results.py — เขียน/อ่าน/filter/prune** — `write_result(job_id, payload)` แบบ atomic (`.tmp` → `os.replace`), `read_result(job_id)`, `filter_triples(payload, threshold)`, `prune(now, jobs)` — TTL ตาม `finished_at` จาก DB (ไม่ใช่ mtime) + ลบ `done` เก่าสุดจนใต้ `MAX_RESULTS_MB`; ทุก prune path ลบได้เฉพาะ `done`
  - **Acceptance:**
    - [ ] ไฟล์ผลที่เขียนแล้ว parse สำเร็จเสมอ (atomic)
    - [ ] filter ให้เฉพาะ `score >= threshold` — threshold ต่าง ๆ ได้ผลต่างจากไฟล์เดิม
    - [ ] TTL ลบเฉพาะ `done` เก่ากว่า `RETENTION_DAYS`; `MAX_RESULTS_MB` ลบ `done` เก่าสุดก่อน; `pending/running/failed` ไม่ถูกแตะทั้งสองกฎ
  - **Verification:** `uv run pytest tests/test_results.py` (tmp_path, model-free)
  - **Dependencies:** Task 1 · **Files:** `src/extraction_api/results.py`, `tests/test_results.py` · **Scope: S**

- [ ] **Task 4: db.py — jobs table** — สร้าง table ตาม schema ใน spec + `enqueue`, `claim` (pending→running, กัน claim ซ้ำด้วย `UPDATE ... WHERE status='pending'` แล้วเช็ค rowcount), `finish(id, status, error, finished_at)`, `get(id)`, `delete(id)`, `prunable_done(older_than)` — sqlite3 stdlib, `check_same_thread=False` + lock เดียว (ponytail: thread lock พอ, ไม่ต้อง WAL/จัดการ connection pool)
  - **Acceptance:**
    - [ ] claim จาก 2 thread พร้อมกันได้ job เดียว
    - [ ] restart (เปิด connection ใหม่) แล้ว pending ยังอยู่
  - **Verification:** `uv run pytest tests/test_db.py`
  - **Dependencies:** Task 1 · **Files:** `src/extraction_api/db.py`, `tests/test_db.py` · **Scope: S**

- [ ] **Task 5: extractor.py + schemas.py** — port `make_gliner_relex` จาก `bake-off/scripts/bakeoff.py:121`: `GLiNER.from_pretrained(Settings().MODEL_DIR)` (ไม่ยิง HF) → `.to(DEVICE).eval()`; `split_sentences` ด้วย pysbd; `extract_raw` inference @ 0.3/0.3/0.3 batch_size 8 → raw triples ครบ provenance (head/head_type/tail/tail_type/relation/score/sentence); `schemas.py`: Pydantic request/response (documents[{field, content}], callback_url, seed_relations, status, triples)
  - **Acceptance:**
    - [ ] `load_extractor()` โหลดจาก `models/` path ล้วน (ตั้ง `HF_HUB_OFFLINE=1` ใน module — ไม่มี network call)
    - [ ] default schema โหลดจาก `bake-off/schema/seed.json`; `seed_relations` ต่อ job ทับได้
    - [ ] POST body validation ตาม trust boundary: field รู้จัก, content string ไม่ว่าง, ≤ MAX_FILES, ≤ MAX_BYTES
  - **Verification:** unit: `uv run pytest tests/test_schemas.py` · จริง: หน้า Task 8 (gpu smoke)
  - **Dependencies:** Tasks 1–2 · **Files:** `src/extraction_api/{extractor,schemas}.py`, `tests/test_schemas.py` · **Scope: M**

- [ ] **Checkpoint 2:** `uv run pytest` (model-free tests) เขียว — results/db/validation logic จบ ยังไม่มี HTTP

### Phase 3: API + worker (vertical slice สุดท้าย: POST → done → GET triples)

- [ ] **Task 6: main.py + endpoints** — FastAPI app + lifespan (สร้าง DB → spawn worker); `POST /jobs` (validate → enqueue → คืน `job_id` ทันที, 422 ถ้าเกิน limit/ผิดรูป), `GET /jobs/{id}` (4 status), `GET /jobs/{id}/triples?threshold=` (default `DEFAULT_THRESHOLD`, อ่านไฟล์ → filter, 404/409 ถ้ายังไม่ done), `DELETE /jobs/{id}`; extractor inject ผ่าน dependency → ตอน test แทนด้วย FakeExtractor
  - **Acceptance:**
    - [ ] POST body ถูกรูป → `job_id` ทันที (ไม่บล็อก); ผิดรูป/เกิน limit → 422 พร้อมเหตุผลชัด
    - [ ] lifecycle ครบ: POST → (FakeExtractor) → GET เห็น pending→done → triples มี provenance ครบ 5 field ทุกอัน
    - [ ] triples ขอ job เดียว query ด้วย threshold ต่างกัน 2 ค่า ได้ผลต่างจากไฟล์เดิม ไม่ rerun
  - **Verification:** `uv run pytest tests/test_api.py` (TestClient + FakeExtractor)
  - **Dependencies:** Tasks 3–5 · **Files:** `src/extraction_api/main.py`, `tests/test_api.py` · **Scope: M**

- [ ] **Task 7: worker.py + webhook + prune hook** — loop: claim → running → extract → เขียนไฟล์ผล atomic → done/failed → webhook (`WEBHOOK_ENABLED`, httpx POST job_id+status เท่านั้น, timeout `CALLBACK_TIMEOUT`, พังไม่กระทบ status) → prune ตาม TTL/ขนาด (หลัง job เสร็จแต่ละ job — ไม่มี timer thread); startup resume: pending ค้างจากรอบก่อนถูก claim ต่อ
  - **Acceptance:**
    - [ ] claim→done lifecycle จบครบใน test, failed path เขียน `error` และไม่มีไฟล์ผลค้าง
    - [ ] `WEBHOOK_ENABLED=true` ยิง callback เมื่อ done/failed; `false` ไม่ยิง (test ทั้งสองทาง — httpx MockTransport)
    - [ ] job pending หลัง "restart" (สร้าง worker ใหม่บน DB เดิม) ถูกหยิบไปทำต่อ
    - [ ] prune วิ่งหลัง job เสร็จ (test: วางผล done เก่า → รัน 1 job → ผลเก่าหาย)
  - **Verification:** `uv run pytest tests/test_worker.py`
  - **Dependencies:** Task 6 · **Files:** `src/extraction_api/worker.py`, `tests/test_worker.py` · **Scope: M**

- [ ] **Checkpoint 3:** `uv run pytest` เขียวทั้งหมด (ยกเว้น gpu) — flow ครบ POST→poll→triples→delete บน FakeExtractor

### Phase 4: GPU smoke + ปิดงาน

- [ ] **Task 8: gpu smoke + success criteria** — `tests/test_gpu_smoke.py` (marker `gpu`, skipif ไม่มี CUDA): POST job จริง 10 ย่อหน้า → poll done → triples ไม่ว่าง + print wall-clock/job; ลอง GET ด้วย threshold 0.5/0.9 จากผลเดิม (ยืนยัน slice ไม่ rerun); กัน model reload ตอน test ด้วย session fixture โหลดครั้งเดียว
  - **Acceptance:**
    - [ ] `uv run pytest -m gpu` เขียวบนเครื่องมี CUDA, พิมพ์วินาที/job
    - [ ] ไม่มี network call ไป HF หลัง startup (ทดสอบโดยเล่น offline)
  - **Verification:** success criteria 1–8 ใน spec ไล่ทีละข้อ; `uv run ruff check src tests --fix && uv run ruff format src tests`
  - **Dependencies:** Checkpoint 3 + model จาก Task 2 · **Files:** `tests/test_gpu_smoke.py` · **Scope: S**

- [ ] **Checkpoint: Complete** — success criteria ครบ 8 ข้อ, `uv sync && uv run pytest` เขียว (ไม่มี GPU), ส่งงานให้ human review

## Parallelization

- หลัง Task 1: Tasks 2, 3, 4 ทำขนานกันได้ (คนละไฟล์ ไม่แชร์ state)
- Task 6 กับ 7 ผูกกันผ่าน lifecycle — ทำตามลำดับ, อย่าขนาน

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| pin ชุด gliner ชนกับ FastAPI/dep ใหม่ตอน resolve | Med | Task 1 resolve `uv.lock` ให้จบก่อนเขียนโค้ด — พังแก้ที่ scaffold ถูกกว่าแก้กลางงาน |
| gliner inference API ต่างจากตอน bake-off (version ต่าง) | Med | Task 5 port ตรงจาก `bakeoff.py` ด้วย pin เดียวกัน — ถ้าต่างจะเห็นตอน gpu smoke ทันที |
| SQLite claim ข้าม thread (uvicorn thread + worker thread) | Low | `check_same_thread=False` + single lock, test claim พร้อมกันใน Task 4 |
| `--reload` โหลด model ใหม่ทุกครั้ง | Low | ระบุใน .env.example/README แล้ว — dev command ใช้กับ non-model code เท่านั้น |

## Open Questions (ไม่บล็อก — จาก spec)

- Deploy บน GPU จริง / ตั้ง SLA — ใช้ตัวเลขจาก Task 8
- Auth, downstream pull ผล — หลังมี consumer จริง
