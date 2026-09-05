# Bake-off รอบ 3: ขยายแผนที่สู่ code/instruction-LLM Dedicated IE

> ต่อจาก `bakeoff-r2-dedicated-ie-landscape.md` · scope: 4 ตัว decoder-based (GoLLIE-7B, KnowCoder, InstructUIE, CodeKGC) ที่รอบก่อนตัดออก — ตอนนี้เอากลับมาเพื่อให้แผนที่ครบทุกสาย ไม่ใช่เพื่อตัดสิน paradigm

## Problem Statement

How might we ทำให้แผนที่ precision × ms ของ dedicated IE ครอบทั้งสาย encoder zero-shot (5 ตัวเดิม) และ code/instruction-LLM ยุค 2023-2024 (4 ตัวใหม่) โดยตัวที่รันไม่ได้จริง (ไม่มี checkpoint / ไม่พอดี VRAM) ก็เป็นข้อมูลบนแผนที่เหมือนกัน

## Recommended Direction

**D2 → D1: smoke-gate ก่อน แล้วค่อยเขียน adapter** — ทุกตัวต้องผ่าน smoke 1 ประโยค (load บน 6GB, parse เป็น Triple, มี score) ก่อนลงทุนเขียน adapter เต็ม ตาม pattern ReLiK/NuExtract ของรอบ 2 ตัวที่รอดตกบน grid precision×ms เดิมเป็นจุดใหม่ แยกสี encoder/decoder · score จาก token-prob (pattern เดียวกับ adapter NuExtract) → rate-once-slice-many ใช้ได้ทันที

เงื่อนไขที่ตกลงไว้ล่วงหน้า: **GoLLIE-13B ตัดที่ VRAM 6GB**, **CodeKGC ตัดถ้าไม่มี public checkpoint** — ทั้งคู่เขียนลง report พร้อมเหตุผล ไม่ใช่หายเงียบ ๆ

## Key Assumptions to Validate

- [ ] checkpoint GoLLIE-7B / KnowCoder / InstructUIE โหลดได้บน `transformers==4.52.4` ที่ pin ไว้ — smoke ตัวแรกตอบทันที (ถ้าต้องการเวอร์ชันต่าง → แยก venv หรือตัด)
- [ ] 6GB พอสำหรับ 7B Q4 + KV cache prompt ยาว (GoLLIE guideline เป็น Python class ยาวมาก) — smoke วัด VRAM จริง ตัดทันทีถ้า OOM
- [ ] token-prob ของ code generation deterministic พอสำหรับ rate-once-slice-many
- [ ] output Python-object ของ GoLLIE/KnowCoder แปลงเป็น (head, relation, tail) ตรง seed schema ได้โดยไม่แก้โค้ดต้นฉบับ

## MVP Scope

- smoke-gate 4 ตัว (load / VRAM / parse / score) — บันทึกผลตายเป็นข้อมูล
- adapter เฉพาะตัวรอด บน `bakeoff.py` เดิม · 20 ประโยค + `schema/seed.json` เดิม · GPU 6GB
- threshold sweep rate-once-slice-many (token-prob)
- จุดใหม่บนแผนที่ precision×ms เดิม แยกสี paradigm
- หัวข้อใหม่ใน report: **"ตัดแล้วและเหตุผล"** (GoLLIE-13B, CodeKGC ถ้าตัด, ตัว smoke ตาย)

## Not Doing (and Why)

- **Fine-tune CodeLlama สำหรับ CodeKGC** — ตัดสินไปแล้ว: เฉพาะ public checkpoint
- **GoLLIE-13B** — VRAM 6GB
- **Quantize แบบพิเศษ (AWQ/GPTQ/GGUF)** นอกจาก Q4 ที่จำเป็น — อย่าเพิ่มตัวแปร
- **ประโยคใหม่ / preprocess table-LaTeX** — คุมตัวแปรกับรอบ 2
- **Blind rating / gold labels / F1** — ข้อจำกัดเดิมของ `docs/bakeoff-evaluation-steps.md`
- **ตัดสิน paradigm ผู้ชนะ** — เป้าคือแผนที่ครบ จุดใกล้กันถือว่าเสมอกันตามมาตราเดิม (±5-10 จุด)

## Open Questions

- GoLLIE checkpoint ตัวไหน public จริง (HiTZ release มีเงื่อนไข) และ Q4 ผ่าน bitsandbytes ได้ไหม
- KnowCoder checkpoint ตัวไหนบน HF + prompt format เป็น code schema แบบไหน
- InstructUIE ขนาดไหน (base/large) ที่มี checkpoint และพอดี 6GB
- มี community port ของ CodeKGC จริงไหม — ตรวจก่อนเข้า smoke ถ้าไม่มีตัดทันที
