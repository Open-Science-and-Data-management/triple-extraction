# CLAUDE.md

API สกัด triples ด้วย GLiNER-Relex — FastAPI + SQLite queue + worker thread เดียว รันด้วย uv (Python 3.12+)

## คำสั่ง

```bash
uv run pytest            # ยูนิต+API test — ไม่แตะ GPU (fake extractor)
uv run pytest -m gpu     # test ที่แตะ model จริง — รันด้วย flag เสมอ
uv run ruff check .      # line-length 110
uvicorn extraction_api.main:app   # Swagger UI ที่ /docs
```

## โครงสร้าง

`src/extraction_api/` เดียว ไม่มีเลเยอร์:

- `main.py` — `create_app(settings, spawn_worker)` ผูก lifespan ทั้งหมด; endpoints 4 อัน; test inject `settings` กับ `spawn_worker=False` ได้
- `db.py` — `JobDB` SQLite queue; `check_same_thread=False` + `threading.Lock` เดียว, `claim()` atomic (ต้องยังใช้ lock เสมอ)
- `worker.py` — daemon thread: claim → run_job → webhook → prune หลังทุก job; `ExtractFn` type alias คือจุด inject fake
- `extractor.py` — โหลด GLiNER-Relex จาก `models/`, rate-once slice-many
- `results.py` — ไฟล์ JSON ต่อ job (write atomic, read, filter_triples, prune ตาม age+size)
- `schemas.py` / `settings.py` — pydantic; validation ที่ trust boundary ทำใน `validate_job_request` แล้วแปลงเป็น 422 ที่ endpoint

## กติกาของโค้ดนี้

- comment/commit เป็นภาษาไทย, ตัวระบุ (identifier) ภาษาอังกฤษ
- DI ผ่าน `app.state` + function params (`ExtractFn`, `transport` ของ httpx) — อย่า import model ตรงใน endpoint; model โหลดครั้งเดียวใน `make_extract_fn()` ตอน startup
- pipeline ของ worker: **claim → extract → write_result (atomic) → finish(done/failed) → webhook → prune** — webhook พังต้องไม่กระทบ status, prune ลบเฉพาะ job done
- job ที่ claim แล้ว process ค้าง (crash/restart) = worker resume claim ต่อเองตอน startup — ไม่ต้องทำ reaper
- pin ชุด `transformers==4.52.4`, `huggingface-hub<1.0`, torch จาก index `pytorch-cu128` — อย่าอัปเดตทั้งชุดพร้อมกัน (ดู memory glirel-setup)
- threshold เก็บไว้ที่ GET /triples เท่านั้น — ห้าม re-extract เพื่อเปลี่ยน threshold, filter จากไฟล์ผลเดิม
- เพิ่ม test เป็น pytest + fake (ไม่แตะ GPU); test ที่แตะ model จริงติด `@pytest.mark.gpu`
- default schema มาจาก `bake-off/schema/seed.json`; `seed_relations` ต่อ job ทับเฉพาะ `relation_hints`
