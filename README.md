# Triple Extraction — โปรเจกต์ทดลองสกัดความสัมพันธ์ (Relation Extraction)

พื้นที่เก็บโค้ดสำหรับศึกษาและทดลองการสกัดความสัมพันธ์/ไตรภาค `(หัว, ความสัมพันธ์, เป้าหมาย)` จากข้อความ
ด้วยโมเดลหลากหลายแนวทาง ทั้งแบบ zero-shot (GLiREL, GLiDRE) และแนวทางตามรายงานวิจัยอื่น ๆ

## REBEL HTTP API (`triple_extraction/`)

ห่อ `Babelscape/rebel-large` ด้วย FastAPI — ส่งข้อความทั้งฉบับ ได้ triples
`{head, relation, tail, sentence_index, start, end, extractor}` โดยไม่ต้องจัดการ
sentence split / batching / parsing เอง (spec: `docs/specs/SPEC-rebel-triple-api.md`)

```bash
uv sync                                    # ติดตั้ง deps + spaCy model (torch CUDA build)
uv run uvicorn triple_extraction.api:app --port 8000            # เริ่ม server (พิมพ์ device ที่ startup)
uv run pytest                              # รัน test ทั้งหมด (mock extractor, ไม่ต้องมี GPU)
uv run pytest -m gpu                       # smoke test โมเดลจริงบน GPU
uv run python -m triple_extraction.smoke "Some text..."         # ยิง pipeline จริง 1 ข้อความ
```

### เรียกใช้ API

Swagger UI (พร้อม input examples กด Try it out ได้เลย): <http://localhost:8000/docs>

```bash
# ส่งเอกสาร → ได้ job_id ทันที (ไม่บล็อกรอ inference)
curl -s -X POST localhost:8000/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"text": "Barack Obama was born in Honolulu.", "meta": {"source": "demo"}}'
# → {"job_id": "..."}

# คำถามสถานะ job (queued/processing → done/failed พร้อม triples + timing)
curl -s localhost:8000/v1/jobs/<job_id>

curl -s localhost:8000/v1/health    # → {"status": "ok", "device": "cuda"}
```

- text เกิน 500,000 ตัวอักษร → `413`, body ผิดรูป → `422`, job ไม่พบ → `404`
- Job เก็บใน SQLite (`data/jobs.db`, env `TRIPLE_EXTRACTION_DB` เปลี่ยนพาธได้) —
  restart server แล้ว job เสร็จแล้วอ่านซ้ำได้, job ค้าง `processing` ตอนบูตถูก mark `failed`

## โครงสร้าง

| พาธ | คำอธิบาย |
|---|---|
| `triple_extraction/` | **REBEL API** — FastAPI + worker thread + SQLite + GPU inference |
| `glirel/` | **GLiREL** — zero-shot relation extraction (spaCy NER → GLiREL) |
| `glidre/` | **GLiDRE** — document relation extraction (GLiNER-based) พร้อม wrapper `process_text()` |
| `docs/` | รายงานการทดลอง + เอกสารอ้างอิง |

## วิธีใช้งาน

โปรเจกต์ย่อยทั้งสองจัดการ dependency ด้วย [`uv`](https://docs.astral.sh/uv/):

```bash
# GLiREL — ตัวอย่างพื้นฐาน (Marco Polo / Great Khan)
cd glirel
uv run python run.py

# GLiREL — extract relations จาก 3 paragraph ตัวอย่าง (spaCy NER + NER override)
uv run python test_paragraphs.py

# GLiDRE — ทดสอบ wrapper process_text() กับ 3 paragraph
cd glidre
uv run python test_3para.py

# GLiDRE — ทดสอบแบบง่าย (ตัวอย่าง Rihanna)
uv run python test_glidre.py
```

โมเดลจะถูกโหลดครั้งแรกจาก Hugging Face Hub (`jackboyla/glirel-large-v0`, `cea-list-ia/glidre_large`, `Babelscape/rebel-large`)
จำเป็นต้องดาวน์โหลดครั้งแรกก่อนใช้งาน

## เอกสาร / รายงาน

| ไฟล์ | เนื้อหา |
|---|---|
| [`docs/report_glirel.md`](docs/report_glirel.md) | ผลทดสอบ GLiREL กับ paragraph ตัวอย่าง 3 บท + ปัญหา NER ของ spaCy |
| [`docs/report_glidre.md`](docs/report_glidre.md) | วิเคราะห์ `process_text()` (GLiDRE) เชิงลึก |
| [`docs/report_slm_triple_extraction.md`](docs/report_slm_triple_extraction.md) | แนวทาง instruction-tuned SLMs สำหรับ triple extraction |
| [`docs/report_triple_extraction_models.md`](docs/report_triple_extraction_models.md) | โมเดลเฉพาะงาน (UniRel, REBEL, PFN, ESGM, SPN) พร้อมคำแนะนำ |
| [`docs/ref/`](docs/ref/) | บันทึกการ setup/รัน (glirel, glidre), SLM guide, รายงานต้นฉบับ |
