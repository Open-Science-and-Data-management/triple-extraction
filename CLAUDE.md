# CLAUDE.md

คู่มือสำหรับ AI coding assistant ที่ทำงานใน repo นี้

## โปรเจกต์

สกัด triples `(head, relation, tail)` จากข้อความ — โปรเจกต์ทดลอง มี 3 ส่วน:

| พาธ | คืออะไร |
|---|---|
| `triple_extraction/` | REBEL HTTP API — FastAPI + worker thread + SQLite + GPU inference (โฟกัสหลัก) |
| `glirel/` | GLiREL zero-shot relation extraction (มี uv env แยกของตัวเอง) |
| `glidre/` | GLiDRE document relation extraction (มี uv env แยกของตัวเอง) |

## คำสั่งที่ใช้บ่อย

```bash
uv sync                                        # ติดตั้ง deps (torch CUDA build + spaCy model)
uv run uvicorn triple_extraction.api:app --port 8000   # เริ่ม server
uv run pytest                                  # test ทั้งหมด (mock extractor, ไม่ต้องมี GPU)
uv run pytest -m gpu                           # smoke test โมเดลจริงบน GPU (ต้องมี GPU)
uv run python -m triple_extraction.smoke "text"  # ยิง pipeline จริง 1 ข้อความ
```

`glirel/` และ `glidre/` รันผ่าน `cd <dir> && uv run python <script>.py`

## สถาปัตยกรรม REBEL API

Flow: `POST /v1/documents` → `JobStore.create_job()` (SQLite) → worker thread
หยิบ job (`claim`) → `RebelExtractor.extract()` (spaCy sentence split → batched
GPU inference → parse `<triplet>` output) → `done`/`failed` พร้อม triples + timing.

- `api.py` — `create_app(store, extractor, start_worker)` inject ได้ทั้งหมดเพื่อ test; lifespan ทำ boot recovery (job ค้าง `processing` → `failed`) และเริ่ม/หยุด worker
- `worker.py` — background thread, loop claim→extract→mark
- `db.py` — `JobStore` (SQLite), `settings.py` — `DB_PATH` (env `TRIPLE_EXTRACTION_DB`), `MAX_TEXT_CHARS` = 500k
- `extractor/rebel.py` — โหลด `Babelscape/rebel-large`, มี `.device` (ใช้ตอน startup print)

## ข้อตกลง

- Test: `pytest` กับ TestClient + fake extractor (device จำลอง) — ไม่ mock ที่ db หรือ worker; GPU test แยกด้วย marker `-m gpu`
- Error contract: text เกิน 500k → `413`, body ผิดรูป → `422`, job ไม่พบ → `404`
- โค้ดและ comment ใน `triple_extraction/` เป็นภาษาไทยปนอังกฤษ (docstring อธิบายเป็นไทย)
- Dependency จัดการด้วย `uv` เท่านั้น — แก้ deps ที่ `pyproject.toml`, pin เวอร์ชันที่จำเป็น
  (ดู memory: transformers==4.52.4, huggingface_hub<1.0 สำหรับ GLiREL)
