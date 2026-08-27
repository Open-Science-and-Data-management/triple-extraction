# Triple Extraction — โปรเจกต์ทดลองสกัดความสัมพันธ์ (Relation Extraction)

พื้นที่เก็บโค้ดสำหรับศึกษาและทดลองการสกัดความสัมพันธ์/ไตรภาค `(หัว, ความสัมพันธ์, เป้าหมาย)` จากข้อความ
ด้วยโมเดลหลากหลายแนวทาง ทั้งแบบ zero-shot (GLiREL, GLiDRE) และแนวทางตามรายงานวิจัยอื่น ๆ

## โครงสร้าง

| พาธ | คำอธิบาย |
|---|---|
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

โมเดลจะถูกโหลดครั้งแรกจาก Hugging Face Hub (`jackboyla/glirel-large-v0`, `cea-list-ia/glidre_large`)
จำเป็นต้องดาวน์โหลดครั้งแรกก่อนใช้งาน

## เอกสาร / รายงาน

| ไฟล์ | เนื้อหา |
|---|---|
| [`docs/report_glirel.md`](docs/report_glirel.md) | ผลทดสอบ GLiREL กับ paragraph ตัวอย่าง 3 บท + ปัญหา NER ของ spaCy |
| [`docs/report_glidre.md`](docs/report_glidre.md) | วิเคราะห์ `process_text()` (GLiDRE) เชิงลึก |
| [`docs/report_slm_triple_extraction.md`](docs/report_slm_triple_extraction.md) | แนวทาง instruction-tuned SLMs สำหรับ triple extraction |
| [`docs/report_triple_extraction_models.md`](docs/report_triple_extraction_models.md) | โมเดลเฉพาะงาน (UniRel, REBEL, PFN, ESGM, SPN) พร้อมคำแนะนำ |
| [`docs/ref/`](docs/ref/) | บันทึกการ setup/รัน (glirel, glidre), SLM guide, รายงานต้นฉบับ |
