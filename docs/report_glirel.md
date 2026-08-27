# รายงานผลทดสอบ GLiREL (Relation Extraction)

**วันที่ทดสอบ:** 2026-08-28
**โมเดล:** `jackboyla/glirel-large-v0`
**Pipeline:** spaCy `en_core_web_sm` (NER) → GLiREL (relation extraction)
**Threshold:** 0.3 | **top_k:** 1

## ภาพรวม

ทดสอบกับ paragraph จำนวน 3 ตัวอย่าง ครอบคลุม relation หลายประเภท (founder, spouse, subsidiary of)
ผลสรุป: โมเดลจับความสัมพันธ์หลักได้ถูกต้องในทุกตัวอย่าง แต่ยังมี noise ทั้งจาก NER ของ spaCy
และการเดาทิศทาง/ชนิด relation ของโมเดลเอง

## ผลทดสอบ

### ตัวอย่างที่ 1 — SpaceX 🚀

> SpaceX was founded by Elon Musk in 2002. The company is headquartered in Hawthorne, California.

| Head | Relation | Tail | Score |
|---|---|---|---|
| Elon Musk | founder | SpaceX | 0.882 ✅ |
| SpaceX | headquartered in | California | 0.735 ✅ |
| SpaceX | headquartered in | Hawthorne | 0.727 ✅ |

**หมายเหตุ:** spaCy แท็ก "SpaceX" ผิดเป็น `PERSON` (จริงเป็น `ORG`) ทำให้โดนกรองด้วย constraint
ของ label `founder` ที่กำหนด `allowed_tail: ["ORG"]` → **แก้โดย override NER** ผ่าน `NER_OVERRIDES["SpaceX"] = "ORG"`
ผลหลังแก้: เจอทั้ง `founder` และ `headquartered in` ครบ

### ตัวอย่างที่ 2 — ครอบครัว Obama 👨‍👩‍👧

> Barack Obama is married to Michelle Obama, and their daughter Malia Obama was born in 1998.

| Head | Relation | Tail | Score |
|---|---|---|---|
| Barack Obama | spouse | Michelle Obama | 0.931 ✅ |
| Michelle Obama | spouse | Barack Obama | 0.910 ✅ |
| Malia Obama | spouse | Michelle Obama | 0.853 ⚠️ |
| Michelle Obama | spouse | Malia Obama | 0.740 ⚠️ |
| Malia Obama | spouse | Barack Obama | 0.557 ⚠️ |
| Barack Obama | spouse | Malia Obama | 0.423 ⚠️ |

**หมายเหตุ:** จับคู่ Barack ↔ Michelle ได้แม่น (0.93) แต่มี false positive:
Malia ถูกเดาเป็น `spouse` ของทั้งพ่อและแม่ ทั้งที่จริงควรเป็น `child` — โมเดล biased ไปทาง `spouse`
(label `child` ที่ตั้งไว้ไม่ถูกเลือกใช้)

### ตัวอย่างที่ 3 — Instagram / Meta 📱

> Instagram was acquired by Facebook in 2012. Today Instagram is a subsidiary of Meta Platforms.

| Head | Relation | Tail | Score |
|---|---|---|---|
| Meta Platforms | subsidiary of | Instagram | 0.574 ⚠️ |

**หมายเหตุ:** เดาถูกว่าคู่ Instagram–Meta มีความสัมพันธ์แบบ `subsidiary of` แต่**ทิศทางสลับ**
(จริงต้องเป็น Instagram → Meta) ส่วน label `acquired by` ไม่ถูกเลือก

## สรุปจุดแข็ง / จุดอ่อน

| ด้าน | สรุป |
|---|---|
| ✅ จุดแข็ง | จับ relation หลักได้ดี (founder, spouse, headquartered in) — โมเดล zero-shot ไม่ต้องเทรน |
| ⚠️ NER upstream | spaCy แท็ก entity ผิดบางตัว (SpaceX → PERSON) — แก้ได้ด้วย `NER_OVERRIDES` |
| ⚠️ ทิศทาง relation | บางครั้งสลับ head/tail (Instagram–Meta) |
| ⚠️ Label bias | โมเดลชอบ label บางตัวเกินไป (spouse แทน child) |
| ⚠️ ข้อจำกัด constraint | `allowed_head`/`allowed_tail` ช่วยกรอง แต่พึ่งพา NER ที่ถูกต้องก่อน |

## ไฟล์ที่เกี่ยวข้อง

- `relation_extractor.py` — ฟังก์ชันหลัก `extract_relations(text, labels, threshold)` + `NER_OVERRIDES`
- `relation_results.json` — ผลดิบทั้ง 3 ตัวอย่าง (JSON)
- `test_paragraphs.py` — สคริปต์ทดสอบที่สร้างรายงานนี้
- `run.py` — ตัวอย่างการเรียกโมเดลแบบ raw (ไม่ผ่าน spaCy)
