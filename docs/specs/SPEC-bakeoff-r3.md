# Spec: Bake-off รอบ 3 — ขยายแผนที่สู่ code/instruction-LLM Dedicated IE

> จาก `docs/ideas/bakeoff-r3-code-llm-ie-extension.md` · เครื่องเดิม: RTX 4050 Laptop 6GB VRAM · คุมตัวแปรกับ r2 ทุกอย่างยกเว้นตัวโมเดล

## Objective

เติมจุด decoder-based บนแผนที่ precision × ms เดิมของ r2 — ไม่ตัดสิน paradigm ผู้ชนะ ตรวจ checkpoint แล้ว (2026-09) scope รอบนี้:

| # | โมเดล | checkpoint | สถานะ |
|---|---|---|---|
| 1 | KnowCoder | `golaxy/KnowCoder-7B-IE` (Llama-2-7B, Apache-2.0, โหลดมาตรฐาน) | เข้า smoke |
| 2 | GoLLIE | `HiTZ/GoLLIE-7B` (Code Llama-7B, ไม่ gated แล้ว, fp16 .bin) | เข้า smoke — ใช้ generation loop + preprocessor ของ HiTZ (`trust_remote_code=True`) |
| 3 | InstructUIE | `ZWK/InstructUIE` = T5-11B fp32 ~45GB — Q4 ≈ 6GB+ ไม่มี headroom (การ์ดใช้เป็นจอ) | **ตัด** — เหตุผลเดียวกับ GoLLIE-13B |
| 4 | CodeKGC | ไม่มี public checkpoint (โค้ดใน `zjunlp/DeepKE` ต้อง fine-tune เอง) | **ตัด** — ตามเงื่อนไขที่ตกลงไว้ล่วงหน้า |
| — | GoLLIE-13B | — | **ตัด** — VRAM 6GB (ยืนยันจาก idea doc) |

**Success ของรอบนี้:** smoke 2 ตัว · adapter ของตัวที่ผ่าน smoke บน `bakeoff.py` เดิม · sweep rate-once-slice-many ด้วย token-prob · จุดใหม่บนแผนที่ precision×ms รวมกับ 5 จุด r2 (แยกสี encoder/decoder) · ทั้งหมดใน `bake-off/report/bakeoff-r3-results.md` + หัวข้อ **"ตัดแล้วและเหตุผล"** (InstructUIE, CodeKGC, GoLLIE-13B, และ smoke ที่ตายถ้ามี)

## Tech Stack

- ต่อยอด env `bake-off/` เดิม (transformers `==4.52.4`, `huggingface_hub<1.0` pin คงเดิม)
- เพิ่ม `bitsandbytes` สำหรับ Q4 nf4 (`load_in_4bit=True`) — quantization รูปแบบเดียวที่อนุญาต
- **ถ้า smoke พบว่าต้องการ transformers ใหม่กว่า pin** → แยก uv dependency-group ต่อ model (`uv sync --group <model>`) — ตัดสินใจแล้วที่ Phase 1 · ms ยังเทียบกันได้เพราะ GPU/hardware เดิม
- GoLLIE: ถ้าต้องการ flash-attention / โค้ด HiTZ ที่ชนกับ pin → กลุ่ม env แยกเช่นกัน

## Commands

```bash
cd bake-off
uv sync

# smoke ก่อนเขียน adapter เต็ม (1 ประโยคเป้า — load/VRAM/parse/score)
uv run bakeoff --smoke knowcoder
uv run bakeoff --smoke gollie

# รันเต็มเฉพาะตัวใหม่ (5 ตัว r2 ไม่ต้องรันซ้ำ)
uv run bakeoff --only knowcoder
uv run bakeoff --only gollie
```

## Project Structure

```
bake-off/
  scripts/bakeoff.py        → เพิ่ม make_knowcoder()/make_gollie() ใน ADAPTER_FACTORIES + field paradigm ต่อ adapter (ไฟล์เดียวเหมือนเดิม)
  schema/seed.json          → ไม่แตะ
  report/bakeoff-r3-results.md → สร้างตอนรัน: ตาราง 2 ตัวใหม่ + แผนที่รวม 7 จุด + "ตัดแล้วและเหตุผล"
docs/specs/SPEC-bakeoff-r3.md
```

## Code Style

- pattern เดิม: dataclass `Adapter` + factory `make_<name>()`, Triple = `(head, relation, tail, score)`, score มาจาก token-prob แบบ `make_nuextract()` (`output_scores=True` → softmax ต่อ token ของ object span) — sweep/slice/best ในไฟล์เดิมใช้ได้ทันทีไม่แก้
- เพิ่ม `paradigm: str` ("encoder" | "decoder") ที่ `Adapter` — ใช้แยกสีบนแผนที่
- ภาษาไทยใน comment สรุป / อังกฤษใน identifier · `# ponytail:` กำกับ shortcut

## Testing Strategy

- ไม่มี test suite — one-off measurement script (เหมือน r2)
- **Gate = smoke ต่อ model**: load บน 6GB ได้ (วัด VRAM peak) · parse ออกมาเป็น Triple ได้ · มี score
- ตัวเลขตรวจด้วยตา ตาม `docs/bakeoff-evaluation-steps.md`

## Boundaries

- **Always:** smoke ผ่านก่อนเขียน adapter เต็ม · warm-up ก่อนจับเวลา · dedupe ซ้ำเป๊ะ · เก็บทุก triple + score · GPU-only · Q4 nf4 เท่านั้น
- **Ask first:** ต้องการ transformers คนละเวอร์ชัน (มีข้อตกลง: แยก group — แต่ถ้า group ที่ต้องเปิดเกิน 2 กลุ่มให้กลับมาถาม) · ตัดโมเดลที่ checkpoint มีจริงแต่ smoke ตายด้วยเหตุผลอื่นนอกจาก OOM/lib
- **Never:** fine-tune CodeLlama เพื่อ CodeKGC · AWQ/GPTQ/GGUF · ประโยคใหม่ / แก้ seed.json · blind rating / gold labels / F1 · ตัดสิน paradigm ผู้ชนะ

## Success Criteria

1. smoke KnowCoder + GoLLIE จบด้วยผลสดหรือผลตายที่บันทึกเหตุผล — ไม่มีตัวหายเงียบ
2. ตัวที่ผ่าน smoke มี adapter เต็ม + sweep ≥3 thresholds จาก raw scores เดียว (rate-once-slice-many)
3. `bake-off/report/bakeoff-r3-results.md`: ตารางผล + แผนที่ 7 จุด (5 r2 + ใหม่) แยกสี paradigm + หัวข้อ "ตัดแล้วและเหตุผล" ครบ 3 รายการตัด + ผลตายถ้ามี
4. ms/ประโยค วัดหลัง warm-up บน GPU เดียวกัน r2, ภายใน 6GB รวม headroom จอ
5. ประโยค + seed schema ไม่เปลี่ยนจาก r2

## Open Questions / ความเสี่ยง

- **GoLLIE integration** คือความเสี่ยงใหญ่สุด: ต้อง generation loop ของ HiTZ (preprocessor + golden-example guideline เป็น Python class ยาว → KV cache กิน VRAM ผิดคาด) — smoke วัด VRAM จริง OOM = ตัดเป็นข้อมูล
- GoLLIE output เป็น Python object literal — ต้อง parse เป็น (head, relation, tail) ตรง seed schema โดยไม่แก้โค้ดต้นฉบับ (ast.literal_eval หลัง normalize หากจำเป็น)
- KnowCoder prompt format เป็น code-style schema — ต้องหา format จาก model card / eval scripts ตอน smoke
- token-prob ของ code generation deterministic พอหรือไม่ — sweep เป็นตัวตอบ
