# ตัดสินใจ: ไม่รวม LLM multimodal ไว้ใน service นี้ — แยกเป็น consumer service นอก API

วันที่: 2026-09-05
บริบท: เอกสารที่ GLiNER ทำไม่ได้ (รูปภาพ figure, สูตร LaTeX) จะให้ LLM multimodal สกัด triples แทน คำถามคือ LLM ควรอยู่ใน extraction_api หรือแยก service

## คำตอบ: แยก

เหตุผล (เรียงตามน้ำหนัก):

1. **worker เป็น thread เดียว** (`worker.py`: claim → extract → done) — ถ้ายัด LLM call เข้า pipeline เดียวกัน ทุก job ต่อคิวหลัง request ที่ช้า 5–30s/รูป + retry + rate limit GLiNER วัดแล้ว 1.5s/job — LLM จะทำให้ queue ทั้งอันสำลักโดยไม่จำเป็น
2. **failure domain ต่างกัน** — กติกาของโค้ดนี้: webhook พังต้องไม่กระทบ status ของ job. LLM (network, quota, API key, model ถูก sunset) เป็น dependency ภายนอกที่พังบ่อยกว่านั้นอีก — แยกแล้ว LLM ล่ม = triples จาก GLiNER ยังออกปกติ
3. **scale คนละแกน** — GLiNER scale ด้วย GPU ในเครื่อง, LLM scale ด้วย quota/concurrency ปลายทาง รวมกันแล้วขยายไม่ได้คนละทาง
4. **secret แยก** — API key ของ LLM ไม่ควรอยู่ใน process ที่หน้าที่เดียวคือ local GPU inference

## วิธีที่เลือกแทน

- service นี้รับทุก field (`latex`, `image`) เก็บผ่านใน result JSON ไม่ extract
- consumer service ตัวที่สอง (เขียนภายหลัง) อ่าน webhook → หยิบ field ที่ไม่ extract ไปให้ LLM → merge triple กลับเอง
- เงื่อนไขที่ควรทบทวนการตัดสินใจ: ถ้า consumer ต้อง merge triple กลับเข้า job เดิมจนต้องเขียน API เพิ่มเยอะ หรือ queue ต้องรู้จักสอง extractor ขึ้นไป — ตอนนั้นค่อยพิจารณา multi-extractor ใน worker

## ทางเลือกที่ปัดทิ้ง

- **LLM ใน worker นี้** — เหตุผลข้อ 1–4 ข้างบน
- **LLM ใน endpoint (synchronous)** — แย่กว่า: request ถูกบล็อกนานเป็นสิบวินาที, timeout ของ reverse proxy จะโดนก่อน
