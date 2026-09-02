# Todo: REBEL Triple Extraction API

> Plan: `tasks/plan.md` · Spec: `docs/specs/SPEC-rebel-triple-api.md`

- [x] Task 1: uv project + deps พร้อม CUDA torch
  - Acceptance: `uv sync` สำเร็จ; `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`; torch เป็น cu128 build
  - Verify: คำสั่งด้านบน + `uv run python -c "import spacy; spacy.load('en_core_web_sm')"`
  - Files: `pyproject.toml`, `.gitignore`, `.python-version`
- [x] Task 2: settings + SQLite db (schema, CRUD, restart recovery)
  - Acceptance: สร้าง db อัตโนมัติ, `create_job/get_job/complete_job/fail_job/recover_stale_jobs` ทำงาน; WAL + busy_timeout ตั้งครบ
  - Verify: `uv run pytest tests/test_db.py`
  - Files: `triple_extraction/settings.py`, `triple_extraction/db.py`, `tests/test_db.py`
- [x] Task 3: REBEL output parser (TDD)
  - Acceptance: parse `<trip> h <sep> r <sep> t` ได้ triples; edge case (คอลัมน์ขาด/เกิน, ว่าง, ไม่มี tag) รอด + เก็บ raw ใน `unparsed`
  - Verify: `uv run pytest tests/test_rebel_parse.py`
  - Files: `triple_extraction/extractor/__init__.py`, `triple_extraction/extractor/rebel.py`, `tests/test_rebel_parse.py`
- [x] Task 4: RebelExtractor (sentence split + batch + GPU infer)
  - Acceptance: แบ่งประโยคด้วย spaCy, batch ยิงโมเดลบน device ที่ settings กำหนด (cuda default), คืน triples พร้อม `sentence_index, start, end`; โมเดล lazy-load
  - Verify: `uv run pytest tests/test_extractor.py` (โหลดโมเดลจริง, mark gpu) + `uv run python -m triple_extraction.smoke "..."` พิมพ์ device
  - Files: `triple_extraction/extractor/rebel.py` (ต่อ), `triple_extraction/smoke.py`, `tests/test_gpu_smoke.py`
- [x] Task 5: Worker thread + job lifecycle
  - Acceptance: ดึง job queued → processing → done/failed ด้วย extractor ที่ inject ได้ (mock ใน test); ตายระหว่าง extract → mark failed; ตอนบูต mark stale processing เป็น failed
  - Verify: `uv run pytest tests/test_worker.py`
  - Files: `triple_extraction/worker.py`, `tests/test_worker.py`
- [ ] Task 6: FastAPI app (3 endpoints) + wire worker
  - Acceptance: `POST /v1/documents` (422 ผิดรูป, 413 เกิน 500k chars) / `GET /v1/jobs/{id}` (404 ไม่พบ, คืน triples+timing เมื่อ done) / `GET /v1/health` (คืน device); response ฝัง `extractor: "rebel"`
  - Verify: `uv run pytest tests/test_api.py` (mock extractor)
  - Files: `triple_extraction/api.py`, `tests/test_api.py`
- [ ] Task 7: ปลายทางจริง end-to-end + save
  - Acceptance: startup พิมพ์ `cuda: NVIDIA GeForce RTX 5060 Ti`; ยิงเอกสารจริงผ่าน curl → triples จริง; kill+restart → job เดิมอ่านซ้ำได้; `uv run pytest` เขียว
  - Verify: รัน server จริง + curl ตาม README; commit spec/plan/tasks/โค้ด
  - Files: `README.md`, ทุกไฟล์ข้างบน (commit)
