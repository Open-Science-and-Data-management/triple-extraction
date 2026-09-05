# Spec: OCR document fields + provenance

จาก idea: [docs/ideas/ocr-document-fields.md](../ideas/ocr-document-fields.md) · คำสั่ง/โครงสร้าง/กติกาโค้ดทั่วไป ดู [CLAUDE.md](../../CLAUDE.md) — spec นี้ครอบเฉพาะส่วนที่ feature นี้เปลี่ยน

## Objective

POST /jobs รับ output จาก PaddleOCR ของเอกสารวิจัยวิศวกรรมได้ครบ 6 แบบ field (text, table, figure_caption, latex, image, section) โดย:

- GLiNER extract เฉพาะ `text` / `table` / `figure_caption` — table strip เป็น text รายแถวก่อน
- field ที่เหลือ pass-through ใน result JSON (echo documents เดิม) — consumer ภายนอก (LLM service) อ่านเอง
- triple ทุกตัวแนบ provenance: document index + field + section

Success = consumer ตัวเดิม (ส่งได้แค่ text) ไม่พัง, consumer ตัวใหม่ได้ครบทั้ง 6 field, ทุก triple รู้ที่มา

## Schema เปลี่ยน 4 จุด (ทั้งหมด additive)

1. `KnownField = Literal["text", "table", "figure_caption", "latex", "image", "section"]`
2. `Document.section: str | None = None`
3. `TripleOut.section: str | None` — สืบทอดจาก document ที่ triple มาจาก
4. Result payload เพิ่ม `documents`: echo list ของ `{field, content, section}` ตาม index เดิม (`source_file` ชี้เข้า array นี้)

`image`/`latex`/`section` ที่ content รับ non-empty str อย่างเดียว — ไม่ validate URI/LaTeX (MAX_BYTES คุมขนาดอยู่แล้ว)

## Implementation

- `extractor.py` — `extract_raw`: filter เอาเฉพาะ text/table/figure_caption; table strip HTML → 1 ประโยค/แถว ไม่ติดชื่อ column (จนกว่า GPU test จะชี้ว่าต้อง prefix header); provenance ของ table row ใช้ field=`table` + section ของ document แม่
- `worker.py` — `write_result` เพิ่ม `documents` echo
- `schemas.py` — 4 จุดข้างบน

## Testing Strategy

- ยูนิต + API test ที่มีอยู่ (fake extractor) ต้องผ่านตามเดิม — non-breaking
- เพิ่ม: fake test ครอบทุก 6 field (pass-through, provenance, echo)
- GPU test (`@pytest.mark.gpu`) 1 ตัว: ตารางจริง 2–3 อันจาก bake-off docs — วัดว่า strip รายแถวจับ relation ได้จริง ตัดสินว่าต้อง prefix ชื่อ column ไหม

## Boundaries

- Always: provenance stamp ครบทุก triple; comment/commit ภาษาไทย; fake test ก่อน GPU test
- Ask first: เปลี่ยน shape ของ `documents` echo; prefix header ถ้า GPU test บอก (เปลี่ยน strip logic ที่ extract เดียว rate-once)
- Never: extract จาก field ที่ไม่ใช่ 3 แบบ; re-extract เพื่อเปลี่ยน threshold; base64 รูปใน payload; LLM ใน service นี้ (ดู [docs/decisions/llm-separate-service.md](../decisions/llm-separate-service.md))

## Success Criteria

- [ ] POST /jobs รับ 6 field ได้ → 202 ปกติ; field เก่า (text ไม่มี section) ยังทำงานเหมือนเดิม
- [ ] GET /triples ทุก triple มี `source_file` + `field` + `section` (section เป็น null ได้)
- [ ] Result JSON มี `documents` echo ครบทุก field รวมที่ไม่ extract
- [ ] GPU test ตารางจริงรันผ่าน — มีตัวเลขตัดสิน header prefix หรือไม่
- [ ] `uv run pytest` (ไม่ติด GPU) ผ่านทั้งหมด

## Open Questions

- Header prefix ตอน strip ตาราง — รอ GPU test (ผูกกับ success criteria ข้อ 4)
