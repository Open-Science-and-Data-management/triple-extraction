# เอกสารวิจัยวิศวกรรม → field schema + provenance

## Problem Statement
**How might we** ให้ POST /jobs รับ output จาก PaddleOCR ของเอกสารวิจัยวิศวกรรมได้ครบ (text, ตาราง, caption, สูตร, รูป) โดยที่ triple ที่ได้รู้ที่มา และ GLiNER ยังเป็นตัว extract เดียวใน service นี้?

## Recommended Direction
ขยาย `KnownField` เป็น 6 แบบ (text, table, figure_caption, latex, image, section) — non-breaking เพราะเพิ่ม enum member. GLiNER extract เฉพาะ `text`/`table`/`figure_caption` (table strip เป็น text รายแถวก่อน), ที่เหลือเก็บผ่านใน result. Triple ทุกตัวแนบ provenance (document index + field + section). LLM multimodal เป็น consumer service นอก API นี้ — อ่าน webhook แล้ว merge triple กลับเอง (เหตุผล: ดู [docs/decisions/llm-separate-service.md](../decisions/llm-separate-service.md)).

## Key Assumptions to Validate
- [ ] Strip HTML → text รายแถว ให้ GLiNER จับ relation ในตารางได้จริง — *test: GPU test กับตารางจริง 2–3 อัน จาก bake-off docs*
- [ ] Provenance ต่อ field ช่วยคุณภาพ downstream จริง — *test: query ด้วย filter section != Related Work แล้วดู precision*
- [ ] Consumer ตัวที่สองอ่าน result เดิมแล้วพอ ไม่ต้องแก้ API — *ตรวจว่า result JSON มี field ที่ไม่ extract ครบ*

## MVP Scope
- `schemas.py`: ขยาย `KnownField`, เพิ่ม `section: str | None`, `image` รับ URI/path
- `extractor.py`: strip table HTML → text; extract เฉพาะ 3 field, อื่น pass-through
- `results.py` / schemas ผลลัพธ์: stamp provenance ต่อ triple
- test: fake extractor ครอบทุก field + 1 GPU test ตารางจริง

## Not Doing (and Why)
- **LLM ใน service นี้** — ดู [docs/decisions/llm-separate-service.md](../decisions/llm-separate-service.md)
- **base64 รูปใน payload** — queue/SQLite บวม, URI พอ
- **section required** — OCR บางเอกสารแยก section ไม่ได้
- **page number, reference list** — เพิ่มตอน provenance เป็นปัญหาจริง / triple ของเอกสารอื่นไม่ควรเก็บ

## Open Questions
- Table strip: ต้องการรู้ว่าแถวไหนคือ header ไหม (ส่งชื่อ column ติดไปทุกแถวไหม) — ตัดสินตอนลงมือ ทำ GPU test ก่อน
