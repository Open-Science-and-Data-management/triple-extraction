# REBEL Triple Extraction API

## Problem Statement
How might we ห่อ REBEL (Babelscape/rebel-large) ด้วย HTTP API ที่รับข้อความ
เอกสารวิจัยทั้งฉบับ แล้วคืนไตรภาค (head, relation, tail) เป็น JSON — โดยซ่อน
preprocessing (แบ่งประโยค, batching, parse decoder output) ไว้ข้างในทั้งหมด
เพื่อให้ app/ระบบอื่นเรียกใช้ต่อได้ทันทีโดยไม่ต้องรู้จักโมเดลเบื้องหลัง

## Recommended Direction
Async mini-queue บน FastAPI + SQLite:

- `POST /v1/documents` รับ `{text, meta?}` → คืน `{job_id}` ทันที (ไม่บล็อก)
- Worker เดียว (background thread) ดึง job จาก SQLite ทีละงาน →
  spaCy แบ่งประโยค → batch ยิงเข้า REBEL (GPU, serialize inference กัน OOM) →
  parse output → บันทึก triples กลับลง SQLite
- `GET /v1/jobs/{id}` → status + triples เมื่อเสร็จ
- SQLite กัน job หายตอน restart; job ที่ค้าง `processing` ตอนบูตถูก mark failed/re-queue
- Schema ฝัง `extractor: "rebel"` + แยกโมดูล extraction ออกจาก HTTP layer
  เผื่อ backend อื่น (GLiREL/GLiDRE) ในอนาคต — โดยไม่สร้าง abstraction หลายชั้น

## Key Assumptions to Validate
- [ ] spaCy แบ่งประโยค text-from-PDF ได้ไม่พัง — ทดสอบกับเอกสารจริง 1 ฉบับ
- [ ] Parse `<trip>...</trip>` รอดทุก edge case — มี fallback เก็บ raw text เมื่อ parse ไม่ได้
- [ ] Worker เดียวพอสำหรับ load ปัจจุบัน — ดู queue depth หลังใช้จริง

## MVP Scope
- 3 endpoints: `POST /v1/documents`, `GET /v1/jobs/{id}`, `GET /v1/health`
- โมดูล: api (FastAPI) / worker / extractor/rebel (preprocess + infer + parse) / db (SQLite)
- Response: triples `[{head, relation, tail, sentence_index}]`, status, timing
- Error handling: text เกินขนาด → 413, body ผิด → 422, job ไม่พบ → 404

## Not Doing (and Why)
- PDF/section parsing — โจทย์กำหนดรับ "ข้อความเท่านั้น"
- Dedup / entity normalization — เป็น post-processing, ไว้หลังเห็นผลจริง
- GLiREL/GLiDRE backend — เตรียม contract ไว้เท่านั้น ยังไม่ implement
- Celery/Redis — SQLite เพียงพอสำหรับ worker เดียว
- Auth/API keys — รัน local
- Batch endpoint หลายเอกสาร — v1.1 เมื่อ client เริ่มต้องการจริง

## Open Questions
- ขีดจำกัดขนาด input (ตัวอย่าง 500KB?) และ behavior เมื่อเกิน
- ต้องการ character offset (ไม่ใช่แค่ `sentence_index`) ใน response ไหม —
  รายงานชี้ว่ามีค่าต่อ provenance และใส่ตอนนี้แทบฟรี

## บริบทที่เกี่ยวข้อง
- ต่อยอดจาก `docs/report_model_selection_recommendation.md` (REBEL = อันดับ 1 สำหรับ v1 API)
- ยึดหน่วยประมวลผลระดับประโยค (sentence-level) ตามข้อสรุปในรายงานเดียวกัน
- ผู้เรียกใช้จริง: app/ระบบอื่น · เกณฑ์สำเร็จรอบนี้: ท่อเดินจริง (plumbing) ไม่รวม tuning คุณภาพ
- Hardware: local + NVIDIA GPU · ยอม in-flight job ตายตอน restart แลกกับไม่ติดตั้ง Redis
