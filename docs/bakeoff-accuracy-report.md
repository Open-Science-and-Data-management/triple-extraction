# Report: ความแม่นยำของ extractor และวิธีการทดสอบ (Bake-off M1)

> วันที่: 2026-09-03 · โมเดล: `knowledgator/gliner-relex-multi-v1.0` (0.3B) vs
> `numind/NuExtract-1.5-tiny` (0.5B) · ฮาร์ดแวร์: RTX 5060 Ti (16GB), fp32/bf16, local
> ผลดิบทั้งหมด: `docs/bakeoff-results.md` · สคริปต์: `scripts/bakeoff.py`
> **ผลสรุป: ตัดเหลือ gliner-relex ที่ threshold 0.5/0.65/0.9 เป็น production default**

## วิธีการทดสอบ

### ชุดทดสอบ

- **27 ประโยคเดี่ยว** จาก paper AI/LLM ที่รู้จักกันดี (LoRA, GPT-4, BERT, RLHF,
  Chain-of-thought, FlashAttention, DeepSeek-R1, quantization, DPO ฯลฯ) — hardcode ใน
  `scripts/bakeoff.py` อ่านตรวจได้ทุกประโยค
- ประโยคถูกเลือกให้คลุมรูปแบบที่ GraphRAG ต้องการ: การเปรียบเทียบโมเดล (`outperforms`),
  การเทรน (`pretrained on` / `fine-tuned on`), ผล benchmark (`evaluated on` /
  `achieves metric`), ผลกระทบเชิงคุณภาพ (`reduces` / `improves`) — รวมถึงประโยคเป้า
  `(LoRA, reduces, Hallucination)` ซึ่งเป็นเกณฑ์ success ใน spec

### ขั้นตอนการวัด

1. ทุก backend ใช้ **seed schema ชุดเดียวกัน** (`schema/seed.json`): relation hint 10 อัน
   (วลีธรรมชาติ) + entity labels — ส่งให้โมเดลเป็น hint เท่านั้น ทุก triple ที่โมเดลคืนถูกเก็บ
   หมด (ไม่มี mapper กรอง)
2. ข้อความถูกแบ่งประโยคด้วย spaCy `en_core_web_sm` เหมือน pipeline จริง แล้วยิง batch
   ต่อประโยค
3. **Warm-up ก่อนจับเวลา** — call แรกของ GPU บนการ์ดนี้ช้า ~89s (kernel compile ของ
   torch) จึงยิงประโยคแรกทิ้งก่อน ตัวเลข ms ในตารางจึงเป็นความเร็วสถานะคงที่
4. **Dedupe triple ซ้ำเป๊ะภายในประโยคเดียว** — gliner-relex พ่น triple เดียวกันซ้ำ 2–4
   ครั้งจาก artifact ของ adjacency matrix (รอบแรก 119 → หลัง dedupe 82) นี่คือการลบ
   artifact ไม่ใช่การกรองความหมาย — triple ที่ head/relation/tail ต่างกันเก็บครบ

### วิธีที่ตัวเลขความแม่นยำได้มา — ตาเปล่า (manual), ไม่มี metric auto

ตาม spec M1 กำหนดให้เป็น "strict F1 นับด้วยตา":

- **จำนวน triples** — นับจาก output จริงของโมเดล (คอลัมน์ "triples รวม" ใน
  `bakeoff_results.md`) เป็นตัวชี้ *recall เชิงปริมาณ* หยาบ ๆ
- **Precision (~%)** — ผู้เขียนอ่าน triple ทุกอันเทียบกับความหมายจริงของประโยคต้นทาง
  แล้วนับว่าอันไหน "ผิดความหมาย" (head/tail ไม่เกี่ยวกันจริง, relation อ่านผิดกริยา,
  garbage แตกจากประโยค) — ตัวเลข % จึงมีความคลาดเคลื่อน ~±5-10 จุดจากการตัดสินของ
  คนเดียว
- **Recall ไม่ได้วัดเป็นตัวเลข** — ต้องมี gold labels ซึ่ง M1 ยังไม่ทำ ที่รายงานได้คือ
  qualitative: กี่ประโยคที่ว่าง, กี่ความสัมพันธ์ที่ควรจับแต่หลุด (เช่น eval list ยาว
  "GSM8K, HumanEval, and MMLU" ถูกจับแค่ GSM8K ที่ threshold สูง)

## ผล (27 ประโยค × 4 configs)

| config | triples รวม | ms/ประโยค | precision ~ | noise ลักษณะ |
|---|---|---|---|---|
| gliner lo 0.3/0.5/0.5 | 130 | 19 | ~40–50% | pairing ข้าม entity ที่ไม่เกี่ยวจำนวนมาก |
| gliner mid 0.5/0.6/0.7 | 73 | 17 | ~60–70% | `pretrained on` โดนใช้แทน `fine-tuned on` เป็นระบบ |
| **gliner high 0.5/0.65/0.9** | **29** | **16** | **~85–90%** (3 ผิด จาก 29) | ผิดเฉพาะ "อ่านกริยาคลาดเคลื่อน" เช่น `(attention mechanism, reduces, recurrence)` |
| nuextract | 30 | 450 | ~65–75% | head/tail ลากยาว + garbage แตกประโยค เช่น `(on, the, Alpaca dataset)` |

**Trade-off ระหว่าง threshold ของ gliner-relex ชัดเจน:** สูงขึ้นทีละช่วง จำนวน triple ลด
130 → 73 → 29 โดย noise หายเร็วกว่า triple ที่ถูกต้องหลุดไป — จุด 0.5/0.65/0.9 คือจุดที่
precision สูงสุดก่อน recall จะทลาย (6/27 ประโยคว่าง)

**จับเกณฑ์ success ของ spec ได้ที่ (LoRA, reduces, Hallucination):**
- gliner @ high: `(Low-Rank Adaptation, reduces, hallucination)` ✓
- gliner @ mid: ✓ (แต่มี noise แทรก)
- nuextract: ✓ (span ยาวกว่าจริงเล็กน้อย)
- รูปแบบ `outperforms`: gliner จับครบทุกประโยคเปรียบเทียบ ✓, nuextract จับบางส่วน

## ตัวเลขนี้แม่นพอ production จริงไหม?

**พอสำหรับ pilot, ไม่พอสำหรับ final quality bar** — เหตุผล:

1. **ข้อผิดพลาดของ gliner @ high มีแบบแผน ไม่สุ่ม** — head/tail ถูก ~90% แล้ว เหลือ
   ความเพี้ยนที่ relation อ่านผิดกริยา และประโยคโครงสร้างซับซ้อนถูกข้าม (recall gap ทำนายได้)
   โปรไฟล์นี้เหมาะกับ GraphRAG ที่ precision > recall (edge ผิดพิษภัยกว่า edge ขาด)
2. **ขนาดตัวอย่างเล็กและนับด้วยคนเดียว** — 27 ประโยคนับตาไม่ใช่ gold-standard F1;
   ก่อนใช้จริงจังควรยิง paper เต็ม (text จาก PaddleOCR) ดูผลรวม ~200-400 triples ต่อรอบ
3. **เพดาน zero-shot ใกล้แล้ว** — การปรับ threshold ต่อไปไม่ช่วยแล้ว

## ทางต่อไป (ถ้าต้องการคุณภาพเหนือนี้)

1. **Fine-tune GLiNER-Relex** — lib `gliner` รองรับ train; label เอง ~100-200 ประโยค
   จาก paper จริง ได้ precision+recall สูงขึ้นทั้งคู่ที่ความเร็วเดิม — ทางที่ถูกต้องที่สุด
2. **โมเดล local ใหญ่ขึ้นเป็นตัวสำรวจ** — Qwen2.5-7B-Instruct 4-bit (~5.5GB VRAM)
   น่าจะแม่นกว่า NuExtract-tiny แต่ ~2-4 s/ประโยค (paper 800 ประโยค ≈ 30-50 นาที)
3. **โมเดล zero-shot ตัวอื่นบน HF** — หมดแล้วในเลข ≤3B (OneKE = 13B เกินการ์ดนี้,
   ไม่มี RE zero-shot ตัวอื่นที่ดีกว่า — ตรวจแล้ว 2026-09)

## การตัดสินใจที่เกิดจาก bake-off นี้

- ตัด `nuextract` backend ออก (โค้ดลบ, คง report ไว้) — เหลือ gliner-relex ตัวเดียว
- Threshold default ของ backend = **0.5 / 0.65 / 0.9** (ชุด high)
- คง field `"model"` ไว้ใน API (ตอนนี้รับค่าเดียว) — เผื่อเพิ่ม backend ใหม่
  (fine-tune/Qwen-7B) ภายหลังโดยไม่แก้ contract
