# Bake-off รอบ 2: แผนที่ภูมิทัศน์ Encoder Zero-shot Dedicated IE

> สร้างจาก idea-refine session 2026-09-04 · scope: โมเดล Dedicated Information Extraction เฉพาะ 5 ตัว encoder zero-shot เท่านั้น

## Problem Statement

How might we เห็นภาพรวม encoder zero-shot Dedicated IE 5 ตัว (GLiNER-Relex, GLiNER→GLiREL, GLiNER 2.x, ReLiK, NuExtract 2.0) บนแกน precision × speed × ความกว้าง schema โดยไม่ตัดสินตัวเดียว — เป้าหมายรอบนี้คือ "แผนที่" ไม่ใช่การเลือก production default

## Recommended Direction

Extend `scripts/bakeoff.py` เดิม: adapter ต่อ model 5 ตัว, ชุด 20 ประโยคเดิม, seed schema เดิม (`schema/seed.json`), GPU-only, warm-up + dedupe เดิมตาม `docs/bakeoff-evaluation-steps.md`

จุดใหม่ของรอบนี้:

- **Rate triple ทุกอัน (unique) ครั้งเดียว พร้อม score ของมัน แล้ว precision@threshold = สัดส่วน triple ที่ "ถูกต้อง" ที่มี score ≥ threshold** — อ่านครั้งเดียว สไลซ์ได้ทุกจุด threshold ทำให้ sweep ต่อ model ไม่แพงเพิ่ม (encoder scores deterministic พอ)
- **Threshold sweep ต่อ model** แทนตัวเลขเดียวกันทุกตัว — score scale ของ encoder แต่ละตัวไม่เท่ากัน รายงานจุด best ของแต่ละ model ด้วยมาตราของมันเอง
- **แผนที่ 2 มิติ**: precision (best-of-sweep) × ms/ประโยค — จุดต่อ model บน grid คุณภาพ-ความเร็ว
- **Distinct relation count**: นับจำนวน relation strings ต่างกันที่ออกมาทั้ง 20 ประโยค ต่อ model — proxy หยาบของ recall/ความกว้าง open schema โดยไม่ต้องมี gold labels

## Key Assumptions to Validate

- [ ] ReLiK ยัด seed schema เป็น custom relation ได้ — smoke test 1 ประโยคก่อนเขียน adapter (ถ้าไม่ได้ บันทึกไว้ในแผนที่ว่า "ปิด schema" — นั่นคือข้อมูล)
- [ ] NuExtract 2.0 รันได้บน GPU เครื่องนี้ (fp16 ~8-10GB VRAM) — smoke test ก่อน นี่คือตัวที่ทำให้รอบล้มได้ทั้งรอบ
- [ ] GLiREL ใช้ GLiNER checkpoint สายเดียวกับ Relex config ปัจจุบัน — ผลขึ้นกับ NER ต้นทางมากกว่าตัวมันเอง ล็อกตั้งแต่ config
- [ ] encoder scores deterministic พอสำหรับ rate-once-slice-many (GPU เพี้ยนเล็กน้อยยอมรับได้)

## MVP Scope

- adapter 5 ตัวบน `scripts/bakeoff.py`
- threshold sweep ต่อ model (rate-once-slice-many)
- ตาราง Step 4 เดิม (config | triples | ms/ประโยค | precision ~ | noise) + จุด best ต่อ model
- แผนที่ 2 มิติ precision × ms/ประโยค
- distinct relation count ต่อ model

## Not Doing (and Why)

- **Blind rating** — เพิ่มงานจัดเตรียม ไม่ใช่แกนของแผนที่
- **CPU run** — รู้อยู่แล้วว่าเดินสาย GPU
- **Table/LaTeX preprocess** — คุมตัวแปรกับรอบที่แล้วก่อน (ประโยคเดิม 20 ประโยค)
- **Gold labels / F1** — M1 ยังไม่ทำ ตามข้อจำกัดของ `docs/bakeoff-evaluation-steps.md`
- **ตัดสิน production default** — เป้าหมายรอบนี้คือแผนที่ จุดที่อยู่ใกล้กันบน grid ถือว่าเสมอกัน (±5-10 จุดของ single-rater)
- **SciER, code-LLM IE (GoLLIE/KnowCoder/InstructUIE/CodeKGC)** — นอก scope: fine-tune schema ตายตัว / เก่า 2023-2024 และถูก general LLM กลืน

## Open Questions

- GLiNER 2.x checkpoint ตัวไหน (multi-task large?) และ input format ต่างจาก Relex แค่ไหน
- ถ้า ReLiK ปิด schema จริง — เก็บไว้ในแผนที่เป็นข้อมูล หรือถอดออกจากรอบ
