# Spec: Bake-off รอบ 2 — แผนที่ Encoder Zero-shot Dedicated IE

> จาก `docs/ideas/bakeoff-r2-dedicated-ie-landscape.md` · เครื่องใหม่: RTX 4050 Laptop 6GB VRAM, driver CUDA 13.3, sm_89

## Objective

ไม่ใช่การเลือก production default — เป้าหมายคือ **แผนที่ 2 มิติ** (precision best-of-sweep × ms/ประโยค) ของ dedicated IE encoder 5 ตัว บนชุดประโยคและ seed schema ชุดเดิมของ r1:

| # | โมเดล | checkpoint / lib |
|---|---|---|
| 1 | GLiNER-Relex | `knowledgator/gliner-relex-multi-v1.0` (lib `gliner`) |
| 2 | GLiNER→GLiREL | GLiNER NER + `glirel` (สายเดียวกับ Relex config) |
| 3 | GLiNER 2.x | `knowledgator/gliner-pyrheads-large-v0.5` |
| 4 | ReLiK | lib `relik` — smoke test ว่ายัด seed schema เป็น custom relation ได้ไหม; ถ้าไม่ได้ = "ปิด schema" ซึ่งเป็นข้อมูลของแผนที่ |
| 5 | NuExtract 2.0 | `numind/NuExtract-2.0-1.5B` fp16 (~3GB) — ตัวใหญ่สุดที่ลงในงบ 6GB แบบมี headroom |

**Success ของรอบนี้:** ตาราง Step 4 (ต่อ model: triples | ms/ประโยค | precision ~ | noise) + threshold sweep ต่อ model (rate-once-slice-many) + จุด best ต่อ model + แผนที่ 2 มิติ — เขียนลง `bake-off/report/bakeoff-r2-results.md` และ distinct relation count ต่อ model — เขียนลง `bake-off/report/distinct-relations.md`

**ทุกโมเดลต้องผ่านงบ VRAM 6GB** — RTX 4050 Laptop ใช้เป็นจอด้วย ต้องเหลือ headroom ให้ Xorg/KV cache

## Tech Stack

- **uv** เป็น env manager (โปรเจกต์ใหม่ ไม่กู้ repo เก่า)
- Python 3.12, torch (cu128 wheel — รองรับ sm_89 + driver 13.3)
- transformers `==4.52.4`, `huggingface_hub<1.0` — **pin ตาม memory เพราะ glirel ต้องใช้** (pin นี้ผูกทั้ง env)
- `gliner`, `glirel`, `relik`, `spacy` + `en_core_web_sm`
- ทุก run บน GPU เท่านั้น — script ตรวจ `torch.cuda.is_available()` แล้ว fail ถ้าไม่มี

## Commands

```bash
# env (ทำงานจาก bake-off/)
cd bake-off
uv sync
uv run python -m spacy download en_core_web_sm

# smoke test ก่อนเขียน adapter เต็ม (1 ประโยค/model — ตาม assumptions ใน idea doc)
uv run bakeoff --smoke relik
uv run bakeoff --smoke nuextract

# รันเต็ม
uv run bakeoff                      # ทุก model, sweep threshold, เขียน report/
uv run bakeoff --only gliner-relex  # model เดียว
```

## Project Structure

```
bake-off/
  pyproject.toml            → deps + uv (+ [project.scripts] bakeoff)
  scripts/bakeoff.py        → ประโยค + schema + adapters 5 + sweep + เขียน report/ (ไฟล์เดียว)
  schema/seed.json          → relation hints 10 อัน + entity labels (r1 ตรงจาก bakeoff-results.md)
  report/
    bakeoff-r2-results.md   → ตาราง Step 4 + sweep + แผนที่ 2 มิติ (สร้างตอนรัน)
    distinct-relations.md   → distinct relation count ต่อ model (สร้างตอนรัน)
docs/specs/SPEC-bakeoff-r2.md
```

ทุกอย่างอยู่ใต้ `bake-off/` — ไม่มีไฟล์ใหม่หลุดไป root

## ชุดประโยค — จัดหมวดตามสิ่งที่วัด

ไม่ยึดติด 27 ประโยค — ปรับเพิ่ม/ลดได้ แต่ **ทุกประโยคต้องมี category กำกับว่าวัดอะไร** และ report ต้องแตกผลตาม category ไม่ใช่รวมเฉย ๆ:

| category | วัดว่าโมเดลจับอะไรได้ | ตัวอย่าง |
|---|---|---|
| `alias` | ผูก alias กับชื่อเต็ม (LoRA = Low-Rank Adaptation) | "Low-Rank Adaptation, or LoRA, reduces…" |
| `comparison` | กริยาเปรียบเทียบ `outperforms`/`surpasses` บน entity คู่ชัด | "GPT-4 outperforms GPT-3.5…" |
| `training` | provenance `pretrained on` / `fine-tuned on` แยกกริยาให้ถูก | "BERT was pretrained on…" |
| `benchmark` | metric + dataset คู่กัน (`evaluated on`, `achieves metric`) | "achieves 86.4% on MMLU" |
| `effect` | ผลกระทบเชิงคุณภาพ `reduces`/`improves` | "Quantization… reduces GPU memory" |
| `multi-rel` | ประโยคซ้อนหลาย relation — นับว่าจับได้กี่ relation จากที่มีจริง | "fine-tune LLaMA-7B on Alpaca using LoRA and evaluate on MMLU" |
| `hard` | โครงสร้างลวง — subject กลางประโยค, ตัวเลขลอย, กริยา passive ที่ชวนอ่านผิด | "(model) —[achieves metric]→ (size) คือ noise รูปแบบนี้" |

ชุดเริ่มต้น = ประโยค r1 ที่แมปเข้า category ได้หมด + เติม/ตัดให้แต่ละ category มี ≥2 ประโยค — ประโยคเป้า success `(LoRA, reduces, hallucination)` เป็น `alias`+`effect`

## Code Style

- สไตล์เดิมจาก r1: type hints, f-string, ภาษาไทยใน comment สรุป / อังกฤษใน identifier
- adapter ต่อ model เป็น dataclass เดียวที่ implement `extract(sentences) -> list[list[Triple]]` โดย Triple = `(head, relation, tail, score)` — score ต้องคืนมาด้วยเสมอ (แกนของ rate-once-slice-many)
- `# ponytail:` comment กำกับทุก shortcut

## Testing Strategy

- ไม่มี test suite — นี่คือสคริปต์วัดผล one-off
- **Gate = smoke test ต่อ model**: 1 ประโยค (ประโยคเป้า LoRA/hallucination) ต้องได้ triple ออกมา + ตรวจ score scale ของแต่ละ model
- ตัวเลขผ่านการตรวจด้วยตา (ตาม `docs/bakeoff-evaluation-steps.md` Step 3)

## Boundaries

- **Always:** warm-up ก่อนจับเวลา · dedupe triple ซ้ำเป๊ะในประโยค · เก็บทุก triple ไม่กรอง · เก็บ score ทุก triple · รัน GPU-only
- **Ask first:** เปลี่ยน checkpoint / เพิ่ม dependency นอก list · ตัด model ออกจากรอบ (ยกเว้น ReLiK ปิด schema ซึ่งบันทึกเป็นข้อมูล)
- **Never:** ตัดสิน production default จากรอบนี้ · ใส่ประโยคใหม่โดยไม่ติด category · CPU run

## Success Criteria

1. 5 ทุกตัวผ่าน smoke และให้ผลตารางครบ (ReLiK ปิด schema = บันทึกแทนค่า precision)
2. ต่อ model มี threshold sweep (≥3 จุด) จาก raw scores เดียว — precision@threshold คำนวณจากการ rate ทุก unique triple ครั้งเดียว
3. แผนที่ 2 มิติอยู่ใน `report/bakeoff-r2-results.md`, distinct relation count อยู่ใน `report/distinct-relations.md`
4. ทุกประโยคมี category และ report แตกผลตาม category ได้ (จับหลุดต่อ category ชัด)
5. ms/ประโยค วัดหลัง warm-up, ทุก model บน GPU เดียวกันภายในงบ 6GB

## Open Questions / ความเสี่ยง

- `gliner-pyrheads` อาจต้อง lib/install ต่างจาก `gliner` — smoke เป็นตัวตัดสิน
- ReLiK custom relation support — ยังไม่ยืนยัน ถ้าไม่ได้ = ข้อมูล ไม่ใช่ความล้มเหลว
- torch wheel บน Python 3.12 + cu128 กับ pin transformers==4.52.4 อาจชนกัน (pyrheads/relik อาจต้อง transformers ใหม่) — ถ้าชน แยก uv env ต่อ model group (`uv sync --group <model>`) แทนการรวม env เดียว
