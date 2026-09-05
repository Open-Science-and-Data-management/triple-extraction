# Extraction API MVP (GLiNER-Relex as a Service)

## Problem Statement
How might we เปิดบริการ triple extraction เป็น async API ที่รับ academic papers
(batch 5–10 ไฟล์, JSON field text) รัน GLiNER-Relex เบื้องหลัง คืน job ID
และให้ผลลัพธ์ที่ตัด threshold ได้ภายหลังโดยไม่ต้อง rerun?

## Recommended Direction
**Lean Job API** — FastAPI กระบวนการเดียว + SQLite ทำหน้าที่ทั้ง queue และ result store,
in-process worker โหลด model บน GPU ครั้งเดียวตอน startup แล้วประมวลผล job ทีละอัน

Endpoints:
- `POST /jobs` — รับ list ของ `{field: "text", content: "..."}`
  (optional: `callback_url`, `seed_relations`), คืน `job_id`
- `GET /jobs/{id}` — status (`pending` / `running` / `done` / `failed`)
- `GET /jobs/{id}/triples?threshold=0.9` — ผลลัพธ์ **ตัดตอน read จาก raw scores**
  (rate once, slice many — เปลี่ยน threshold = slice ข้อมูลเดิม ไม่ rerun)
- optional webhook: ถ้าแนบ `callback_url` มา POST ผลไปแจ้งเมื่อเสร็จ

ทำไมรูปแบบนี้: workload จริงเบามาก (~7.75 นาที GPU/วัน ที่ 100 papers) จึงไม่ต้องมี
Celery/Redis — table `jobs` ธรรมดาใน SQLite ทำหน้าที่ queue (status='pending')
restart แล้วงานไม่หาย พร้อมขยายภายหลังด้วย worker process เพิ่มโดยไม่เปลี่ยน schema

ทุก triple เก็บ provenance: `source_file`, `sentence`, `score`, `model_version`,
`seed_schema_hash` — เปิดทาง human review และ audit ย้อนหลังได้ทันทีที่ต้องการ

## Key Assumptions to Validate
- [ ] "Server จะหนักถ้า realtime" — ตัวเลขบอกตรงข้าม (7.75 นาที/วัน)
      ทดสอบ: รัน 10 papers จริง วัด wall-clock ต่อ job ก่อนตัดสินใจ infra เพิ่ม
- [ ] "15.5 ms/ประโยค โอนมา production ได้" — bake-off วัดบนประโยคสั้น 28 อัน
      production อาจช้ากว่า 2–5× — วัดจริงแล้วค่อยตั้ง SLA (ยังไม่ promise อะไร)
- [ ] "Precision ~0.88 อยู่ตัวเดิมบน paper เต็ม" — distribution จริงต่างจากชุดทดสอบ
      — ทำ spot-check: sample triple จาก production ตรวจตาเป็นระยะ (report §6.3)
- [ ] "Input เป็น JSON field text เสมอ" — consumer จะส่งของผิดตั้งแต่วันแรก
      — validation ที่ trust boundary: field ที่รู้จัก, content เป็น string
      ไม่ว่าง, จำกัดขนาด/จำนวนไฟล์ต่อ job

## MVP Scope
**ใน:** 3 endpoints ข้างบน, SQLite, in-process worker (โหลด model ครั้งเดียว),
raw scores ทุก triple, threshold เป็น query param (default 0.9),
schema seed ต่อ job (default = bake-off/schema/seed.json), webhook callback,
input validation, provenance fields, job retention (เก็บผลตามอายุ/จำนวน)

**ออก:** ทุกอย่างใน Not Doing ด้านล่าง

## Not Doing (and Why)
- **PDF / table / image fields** — โฟกัส field text ตามที่ตกลง; interface
  (`field` enum) รองรับการเพิ่มภายหลังโดยไม่ทำ breaking change
- **Celery / Redis / message broker** — 7.75 นาที/วัน ไม่สมเหตุผล; SQLite
  queue ขยายได้ถ้าจำเป็นจริง
- **Entity coreference / รวม triple ข้ามประโยค** — report §7 ระบุเป็นขั้นถัดไป
  ของ pipeline อยู่แล้ว ทำใน MVP คือเพิ่มงานก่อนงานหลักเสร็จ
- **Human review tier (accepted/pending_review)** — เลือกไว้ตอนนี้ว่าไม่ทำ แต่
  raw scores + provenance ทำให้เปิดได้ทีหลังโดยไม่ rerun (ค่าตัดสินใจ = 0)
- **Auth เต็มรูปแบบ** — MVP อยู่หลัง network ภายใน; เพิ่ม API key ตอน expose จริง
- **Horizontal scaling / multiple workers** — ทำเมื่อ load จริงบังคับ

## Open Questions
- Job retention: เก็บผลนานแค่ไหน / ลบเมื่อไหร่ (แตะไม่ได้จนกว่า consumer
  downstream ยืนยันว่าดึงไปแล้ว)
- ขนาด job สูงสุด: hard cap ที่เท่าไหร่ (ปัจจุบันบอกว่า 5–10 ไฟล์ — ตั้ง limit
  เช่น 20 ไฟล์ / 5 MB ต่อ job กันใครส่ง 500 มา)
- เมื่อไหร่ deploy บน GPU จริง และ GPU ตัวไหน (model 1.29 GiB — การ์ดไหนก็ไหว
  แต่ต้องยืนยันว่า co-locate กับ service อื่นได้จริง)
