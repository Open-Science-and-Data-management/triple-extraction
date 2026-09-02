# Spec: REBEL Triple Extraction API

## Objective

ห่อ `Babelscape/rebel-large` ด้วย HTTP API ที่รับข้อความเอกสารวิจัยทั้งฉบับ แล้วคืนไตรภาค
(head, relation, tail) เป็น JSON ซ่อน preprocessing ทั้งหมด (แบ่งประโยค, batching,
parse decoder output) ไว้ข้างใน — ผู้เรียกใช้ (app/ระบบอื่น) ไม่ต้องรู้จักโมเดลเบื้องหลัง

**เกณฑ์สำเร็จรอบนี้: ท่อเดินจริง (plumbing)** — ไม่รวม tuning คุณภาพ extraction

Acceptance criteria ระดับ API:
- `POST /v1/documents` รับ `{text, meta?}` → คืน `{job_id}` ทันที (ไม่บล็อกรอ inference)
- `GET /v1/jobs/{id}` → `{status, triples, timing}` — triples `[{head, relation, tail, sentence_index, start, end}]`
- `GET /v1/health` → `{status, device}` — device เป็น `cuda` เมื่อ GPU พร้อม
- text ที่ job ยังไม่เสร็จ → status `queued`/`processing` ไม่คืน triples
- text เกิน 500,000 ตัวอักษร → HTTP 413, body ผิดรูป → 422, job ไม่พบ → 404
- Restart server แล้ว job ที่เสร็จแล้วยังอ่านได้จาก SQLite; job ที่ค้าง `processing` ตอนบูตถูก mark `failed`

## Tech Stack

| ส่วน | เลือก | เหตุผล |
|---|---|---|
| ภาษา/env | Python ≥3.12 + **uv** (root-level pyproject.toml) | กำหนดโดยผู้ใช้; ต่อยอดแพทเทิร์น uv.lock เดิม |
| HTTP | FastAPI + uvicorn | async + pydantic validation ตาม idea doc |
| Persistence | SQLite (stdlib `sqlite3`) | กัน job หายตอน restart, ไม่ติดตั้ง Redis/Celery |
| Sentence split | spaCy `en_core_web_sm` | เร็ว, แค่ต้องการ sentencizer |
| โมเดล | `Babelscape/rebel-large` ผ่าน transformers | อันดับ 1 จาก report_model_selection_recommendation |
| Deep learning | torch **CUDA build (cu128)** — GPU default | RTX 5060 Ti 16GB (sm_120 ต้อง CUDA 12.8+) |
| Test | pytest + httpx (ASGI TestClient) | standard สำหรับ FastAPI |

GPU policy: ตรวจ `torch.cuda.is_available()` ตอน startup — **cuda เป็น default**, fallback
เป็น `cpu` พร้อม warning ถ้าไม่มี GPU และ **พิมพ์ device ที่ startup เสมอ** ทั้ง inference
serialize ภายใน worker เดียวกันอยู่แล้ว (กัน OOM ตาม idea doc)

## Commands

```bash
uv sync                                    # ติดตั้ง deps + spacy model (post-sync script)
uv run uvicorn triple_extraction.api:app --port 8000 --reload   # dev server
uv run pytest                              # รัน test ทั้งหมด (mock extractor, ไม่ต้องมี GPU)
uv run pytest -m gpu                       # smoke test บน GPU จริง (ต้องมี GPU + weight)
uv run python -m triple_extraction.smoke "Some text..."        # ยิง pipeline จริง 1 ข้อความ
```

## Project Structure

```
pyproject.toml                  → root project "triple-extraction" (uv)
uv.lock
triple_extraction/              → main package
  api.py                        → FastAPI app, 3 endpoints, request validation
  worker.py                     → background thread: poll SQLite → extract → save
  db.py                         → SQLite schema + job CRUD (busy_timeout, WAL)
  settings.py                   → MAX_TEXT_CHARS, DB_PATH, MODEL_NAME, device detection
  extractor/
    __init__.py                 → `Extractor` protocol (เตรียม backend อื่นในอนาคต)
    rebel.py                    → preprocess (sentence split + batch) + infer + parse <trip>
  smoke.py                      → CLI ทดสอบ pipeline ปลายทาง
tests/
  test_api.py                   → 3 endpoints: 200/404/413/422, mock extractor
  test_rebel_parse.py           → parse <trip> decoder output + fallback raw text
  test_worker.py                → job lifecycle queued→processing→done/failed, restart recovery
  test_gpu_smoke.py             → mark: gpu — โมเดลจริงบนเอกสารจริง 1 ฉบับ
data/                           → sqlite db file (gitignore)
docs/specs/                     → spec อยู่ที่นี่ (docs/specs/SPEC-rebel-triple-api.md)
tasks/                          → plan.md + todo.md ตาม convention ของ skill
```

## Code Style

```python
# ตัวอย่างสไตล์เป้าหมาย: type hint ครบ, ฟังก์ชันเล็ก, ไม่สร้าง abstraction เกินจำเป็น
def parse_rebel_output(raw: str) -> ParsedTriples:
    """Parse REBEL decoder output (`<trip> head <sep> relation <sep> tail`) ออกเป็น triples.

    แถวที่ parse ไม่ได้ถูกเก็บใน `unparsed` เป็น raw text เพื่อ debug ไม่ทิ้งเงียบ ๆ
    """
```

- ตั้งชื่อแบบ snake_case, class แบบ PascalCase, คงคำศัพท์เดิมจาก spec (`job_id`, `triples`)
- type hints ทุก public function; pydantic models กำหนด response shape ของ API
- log ด้วย print/logging แบบเรียบง่าย — ไม่เพิ่ม dependency logging framework
- ความคิดเห็นภาษาไทยได้ตาม convention เอกสารใน repo

## Testing Strategy

- **pytest** ทั้งหมดอยู่ `tests/`; unit tests **mock extractor** (fake คืน triples ตายตัว)
  เพื่อให้รันได้เร็วและไม่ต้องมี GPU
- ระดับ:
  - unit — parse `<trip>` output (รวม edge case: คอลัมน์เกิน/ขาด, ว่าง, escape)
  - integration — API + SQLite + worker loop ด้วย fake extractor (job lifecycle ครบ)
  - gpu smoke (`-m gpu`) — โมเดลจริง + เอกสารจริง 1 ฉบับ (ตรวจ assumption ว่า spaCy
    split text-from-PDF ไม่พัง)
- Coverage: ไม่ตั้งเพดาน % แต่ทุก endpoint ต้องมี test เคส error ครบทั้ง 404/413/422

## Boundaries

- **Always:** รัน `uv run pytest` ก่อนถือว่า task เสร็จ · พิมพ์ device ตอน startup ·
  ทุก triple มี `extractor: "rebel"` ฝังใน response/schema · เก็บ raw text ของ output
  ที่ parse ไม่ได้
- **Ask first:** เปลี่ยน SQLite schema · เพิ่ม dependency ใหม่ · implement backend อื่น
  (GLiREL/GLiDRE) · เพิ่ม endpoint นอก 3 ตัวที่กำหนด
- **Never:** ใส่ auth/API key ใน v1 (นอก scope) · ทำ PDF/section parsing ·
  dedup/entity normalization · commit `data/` หรือ `.env` · ใช้ Celery/Redis

## Success Criteria

1. `uv sync` แล้ว env พร้อมใช้บนเครื่องนี้ โดย torch เป็น CUDA build
2. สตาร์ท server แล้ว **terminal พิมพ์ device ที่ใช้** (คาดหวัง `cuda: NVIDIA GeForce RTX 5060 Ti`)
3. ยิง `POST /v1/documents` ด้วยข้อความจริง → ได้ `job_id` < 1 วินาที; poll
   `GET /v1/jobs/{id}` จน `done` แล้วได้ triples รูป `[{head, relation, tail,
   sentence_index, start, end}]`
4. `uv run pytest` เขียวทั้งหมด (ไม่รวม `-m gpu`); `uv run pytest -m gpu` ผ่านบนเครื่องจริง
5. Kill + restart server กลาง job → job เสร็จแล้วอ่านซ้ำได้, job ค้าง `processing` ถูก mark `failed`
6. Response ทุก triple ระบุ `extractor: "rebel"`

## Open Questions

(ปิดแล้วจากการรีวิว spec รอบแรก)
- Input limit = **500,000 chars** → 413 เมื่อเกิน ✅
- Response มี **character offset** (`start`, `end` เทียบตำแหน่งใน input text) ✅
- โค้ดอยู่ **root-level pyproject** ✅
