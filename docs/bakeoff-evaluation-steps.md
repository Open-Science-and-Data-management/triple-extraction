# Step การประเมิน extractor (Bake-off M1)

> เรียบเรียงจาก `docs/bakeoff-accuracy-report.md` — วิธีการเท่านั้น ไม่ผูกกับโมเดลใด
> ผลดิบ: `docs/bakeoff-results.md` · สคริปต์: `scripts/bakeoff.py`

## Step 0 — เตรียมชุดทดสอบ

- สร้าง **ชุดประโยคเดี่ยว** (~20 ประโยค) จาก paper จริง — hardcode ใน `scripts/bakeoff.py` อ่านตรวจได้ทุกประโยค
- เลือกให้คลุมรูปแบบที่ GraphRAG ต้องการ: การเปรียบเทียบ (`outperforms`), การเทรน (`pretrained on` / `fine-tuned on`), benchmark (`evaluated on` / `achieves metric`), ผลกระทบ (`reduces` / `improves`)
- ระบุ **ประโยคเป้าเกณฑ์ success** ไว้ 1 ประโยค (triple ที่ spec กำหนดว่าต้องจับได้)
- **สิ่งที่ต้องการจาก step นี้:** ชุดประโยคที่อ่านตรวจได้ทุกประโยค + เกณฑ์ success ที่ชัด

## Step 1 — ตั้งเงื่อนไขให้ทุก config เท่ากัน

- ใช้ **seed schema ชุดเดียวกัน** (`schema/seed.json`): relation hint ~10 อัน (วลีธรรมชาติ) + entity labels — ส่งเป็น *hint* เท่านั้น
- ทุก triple ที่โมเดลคืนถูกเก็บหมด **ไม่มี mapper กรอง** เพื่อวัดพฤติกรรมดิบของโมเดล

## Step 2 — รัน pipeline เหมือนของจริง

- แบ่งประโยคด้วย spaCy `en_core_web_sm` (ตัวเดียวกับ pipeline จริง) → ยิง batch ต่อประโยค
- **Warm-up ก่อนจับเวลา** — call แรกของ GPU ช้ามาก (kernel compile) จึงยิงประโยคแรกทิ้ง ตัวเลข ms ที่วัดจึงเป็นความเร็วสถานะคงที่
- **Dedupe triple ซ้ำเป๊ะภายในประโยค** — ลบ artifact ของโมเดล ไม่ใช่การกรองความหมาย — triple ที่ head/relation/tail ต่างกันเก็บครบ
- ทำซ้ำต่อ 1 config (โมเดล × ค่า threshold)

## Step 3 — วัดผลลัพธ์ (ตาเปล่า ไม่มี metric auto)

| สิ่งที่วัด | วิธี | รูปแบบคำตอบที่ได้ |
|---|---|---|
| **จำนวน triples** | นับจาก output จริงของโมเดล | จำนวนเต็ม ต่อ 1 config (ชี้ recall เชิงปริมาณหยาบ ๆ) |
| **Precision ~%** | อ่าน triple ทุกอันเทียบความหมายจริงของประโยคต้นทาง นับอันที่ "ผิดความหมาย" (head/tail ไม่เกี่ยวกันจริง, relation อ่านผิดกริยา, garbage แตกจากประโยค) | % โดยประมาณ ±5–10 จุด (ตัดสินโดยคนเดียว) |
| **Recall** | ❌ ไม่ได้วัดเป็นตัวเลข (ต้องมี gold labels ซึ่ง M1 ยังไม่ทำ) | qualitative: กี่ประโยคว่าง, ความสัมพันธ์ที่ควรจับแต่หลุด (เช่น eval list ยาวถูกจับแค่บางตัว) |
| **ความเร็ว** | ms/ประโยค หลัง warm-up | จำนวน ms ต่อประโยค |

## Step 4 — สรุปเป็นตารางเทียบ (1 แถวต่อ config)

รูปแบบคำตอบของทั้ง evaluation:

| config | triples รวม | ms/ประโยค | precision ~ | noise ลักษณะ |
|---|---|---|---|---|
| config A | … | … | ~…% | … |
| config B | … | … | ~…% | … |

## Step 5 — เช็คเกณฑ์ success ของ spec

ตรวจว่าแต่ละ config จับประโยคเป้า (triple ที่ spec กำหนด) ได้ไหม และรูปแบบสำคัญอื่น (เช่น `outperforms`) จับครบทุกประโยคไหม:

```
config A:  (head, relation, tail) ✓
config B:  ✓ แต่มี noise แทรก
```

## Step 6 — ตัดสินใจจากผล

- เทียบ trade-off ระหว่าง threshold: threshold สูงขึ้น จำนวน triple ลด โดย noise หายเร็วกว่า triple ที่ถูกต้องหลุด → เลือกจุดที่ **precision สูงสุดก่อน recall จะทลาย** (นับจำนวนประโยคว่างกำกับไว้)
- เลือก config ที่ชนะเป็น production default ตัด backend ที่แพ้ออก

---

## ข้อจำกัดของตัวเลขที่ได้จากวิธีนี้

- n เล็ก, นับด้วยคนเดียว, ไม่มี gold labels → **พอสำหรับ pilot ไม่พอ final quality bar**
- รอบถัดไป: ยิง paper เต็มจาก PaddleOCR (~200–400 triples/รอบ) หรือ label เอง 100–200 ประโยคเพื่อทำ F1 จริง — รายละเอียดเพิ่มเติมดูใน `docs/bakeoff-accuracy-report.md`
