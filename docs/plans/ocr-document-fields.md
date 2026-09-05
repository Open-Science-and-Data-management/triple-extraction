# Plan: OCR document fields + provenance (docs/specs/ocr-document-fields.md)

## Context

POST /jobs ตอนนี้รับได้แค่ `field="text"` (schemas.py:10 `KnownField = Literal["text"]`) ต้องรับ output ของ PaddleOCR ครบ 6 field (text, table, figure_caption, latex, image, section) โดย GLiNER extract เฉพาะ 3 field แรก (table strip HTML → รายแถว), ที่เหลือ pass-through, และ triple ทุกตัวแนบ provenance (source_file + field + section) — additive ทั้งหมด consumer เดิม (text) ต้องไม่พัง

**ข้อค้นพบจากการอ่านโค้ด (ประหยัดงาน):**
- `worker.py:68` **echo `documents` ใน result อยู่แล้ว** — spec ข้อ 4 ไม่ต้องแก้ worker เลย เพิ่ม section เข้า dict ตอน POST (Task 2) แล้ว echo ตามฟรี
- `extract_raw` (extractor.py:56-57) เดิน flat `(index, field, sentence)` อยู่แล้ว — เพิ่ม `section` ใน tuple ก็ stamp provenance ได้เลย
- `extract_raw` ต้องมี model → ยูนิต strip/filter ต้องแยก strip เป็นฟังก์ชัน model-free + stub model ที่มี `.inference()` ปลอม
- `tests/test_schemas.py:26` ใช้ `"table"` เป็นตัวทดสอบ reject — หลังแก้ต้องกลับทิศเป็น accept

## Architecture Decisions

- Strip ตารางด้วย stdlib `html.parser.HTMLParser` เก็บ `<tr>` → join `<td>`/`<th>` ด้วย " " — **ไม่ prefix ชื่อ column** ก่อน (ผูกกับ open question, ตัดสินจาก GPU test แล้วถามผู้ใช้ก่อนเปลี่ยน)
- แยก `_strip_table_rows(html: str) -> list[str]` ออกจาก `extract_raw` — model-free, ยูนิตได้
- provenance ของ table row: field=`table` + section ของ document แม่ (ตาม spec)
- GPU test วัด **ทั้ง 2 variant** (strip ล้วน vs prefix header) ในรอบเดียว พิมพ์ตัวเลขเทียบ — ไม่เปลี่ยน strip logic เอง

## Task List

### Phase 1: Schema + wiring (foundation)

- [ ] **Task 1: schemas.py — KnownField 6 แบบ + section** (S)
  - `KnownField = Literal["text", "table", "figure_caption", "latex", "image", "section"]`
  - `Document.section: str | None = None`
  - `TripleOut.section: str | None = None`
  - content ยังใช้ `min_length=1` เดิม (non-empty str อย่างเดียว — ไม่ validate URI/LaTeX, MAX_BYTES คุมอยู่)
  - แก้ `test_schemas.py`: `test_rejects_unknown_field` → ใช้ field นอก 6 แบบ (เช่น `"code"`) เป็น reject; เพิ่ม accept ครบ 6 field + section default None
  - **Verify:** `uv run pytest tests/test_schemas.py`
  - Files: `src/extraction_api/schemas.py`, `tests/test_schemas.py`

- [ ] **Task 2: main.py — ส่ง section ลง DB** (S)
  - `create_job` (main.py:73): `{"field": d.field, "content": d.content, "section": d.section}`
  - echo `documents` ใน result ตามอัตโนมัติ (worker.py:68 มีอยู่แล้ว — ไม่แก้ worker)
  - เพิ่ม `test_api.py`: POST ครบ 6 field → 201 ปกติ; body เดิม (text ไม่มี section) → 201 เหมือนเดิม (non-breaking)
  - **Verify:** `uv run pytest tests/test_api.py tests/test_worker.py` (test_worker ต้องผ่านตามเดิม — fake documents เก่าไม่มี section ยังทำงาน)
  - Files: `src/extraction_api/main.py`, `tests/test_api.py`

### Checkpoint 1
- [ ] `uv run pytest` ผ่านทั้งหมด, `uv run ruff check .` สะอาด

### Phase 2: Extraction (core)

- [ ] **Task 3: extractor.py — filter 3 field + strip table + provenance section** (M)
  - `_strip_table_rows(html: str) -> list[str]` — stdlib HTMLParser, 1 ประโยค/แถว, join cell ด้วยช่องว่าง (`# ponytail:` comment บอก ceiling: ไม่ prefix header — รอ GPU test)
  - `extract_raw`: สร้าง flat เฉพาะ field ใน `{"text", "table", "figure_caption"}`; table → `_strip_table_rows(doc["content"])` แทน `split_sentences`; tuple เป็น `(index, field, section, sentence)` — section จาก document แม่
  - dict ผลลัพธ์เพิ่ม `"section": section` (None ได้)
  - stub model (`.inference()` คืน canned) ใน `tests/test_extractor.py` (ไฟล์ใหม่): ยูนิต `_strip_table_rows` (ตาราง HTML เล็ก, แถวว่าง, ไม่มี table tag), ยูนิต filter (latex/image/section ไม่โดน extract), provenance ครบ (source_file/field/section ตรง document แม่)
  - **Verify:** `uv run pytest tests/test_extractor.py`
  - Files: `src/extraction_api/extractor.py`, `tests/test_extractor.py` (ใหม่)

- [ ] **Task 4: fake test end-to-end ครบ 6 field** (S)
  - ต่อยอด `test_worker.py` / `test_api.py`: job ที่มี document ครบ 6 field → result JSON echo `documents` ครบทั้ง 6 (รวมที่ไม่ extract) + triples เฉพาะจาก 3 field + ทุก triple มี field/section (section=null ได้)
  - **Verify:** `uv run pytest` (ทั้งชุด ไม่ติด GPU)
  - Files: `tests/test_worker.py` หรือ `tests/test_api.py` (ที่เดียวพอ)

### Checkpoint 2
- [ ] Success criteria ข้อ 1–3, 5 ของ spec ครบ: 6 field → 201, body เก่าไม่พัง, GET /triples มี source_file+field+section, echo ครบ, pytest ผ่าน

### Phase 3: GPU test (ตัดสิน open question)

- [ ] **Task 5: GPU test ตารางจริง** (S)
  - `tests/test_gpu_tables.py` (ใหม่) — `@pytest.mark.gpu` + skipif ไม่มี CUDA ตามแบบ `test_gpu_smoke.py`
  - ตาราง HTML จริง 2–3 อันจาก bake-off docs (`bake-off/report/bakeoff-r2-results.md` แปลงเป็น HTML แบบที่ PaddleOCR emit)
  - รัน extract_raw 2 variant: rows ล้วน vs rows prefix ชื่อ column — พิมพ์จำนวน relation ที่จับได้ต่อ variant (print สำหรับ `-s`)
  - assert ระดับต่ำ: ได้ triple ออกมา, provenance field="table" — **ไม่ assert ว่า variant ไหนชนะ**
  - สรุปตัวเลขให้ผู้ใช้ → ถามก่อนเปลี่ยน strip logic (boundary "Ask first" ของ spec)
  - **Verify:** `uv run pytest -m gpu -s tests/test_gpu_tables.py`
  - Files: `tests/test_gpu_tables.py` (ใหม่)

### Checkpoint: Complete
- [ ] `uv run pytest` (fake) + `uv run pytest -m gpu` ผ่าน
- [ ] `uv run ruff check .` สะอาด
- [ ] เสนอตัวเลข header prefix ให้ผู้ใช้ตัดสิน (แยกจาก task นี้)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| strip รายแถวทำ GLiNER จับ relation ไม่ได้ (context หลุด) | Med | GPU test วัดก่อน — prefix header เป็น upgrade path จุดเดียวใน `_strip_table_rows` |
| echo documents บวมไฟล์ผล (latex/image) | Low | MAX_BYTES คุมตั้งแต่ POST อยู่แล้ว — ไม่ทำอะไร |
| pysbd segment แถวตารางแปลก ๆ | Low | table ไม่ผ่าน split_sentences เลย — 1 cell-join/แถว |

## Open Questions (ตัดสินหลัง Task 5 ไม่บล็อกงาน)
- Prefix ชื่อ column ตอน strip ตารางไหม — รอตัวเลขจาก GPU test แล้วถามผู้ใช้
