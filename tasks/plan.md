# Implementation Plan: Bake-off รอบ 2 — Encoder Zero-shot Dedicated IE

> จาก `docs/specs/SPEC-bakeoff-r2.md` · ทำงานทั้งหมดใต้ `bake-off/` · GPU-only (RTX 4050 Laptop 6GB)

## Overview

สคริปต์วัดผล one-off (`bake-off/scripts/bakeoff.py`, ไฟล์เดียว) ที่รัน dedicated IE encoder 5 ตัวบนชุดประโยค category-annotated + seed schema ของ r1 — ได้แผนที่ 2 มิติ (precision best-of-sweep × ms/ประโยค) + threshold sweep ต่อ model (rate-once-slice-many) + distinct relation count เขียนลง `bake-off/report/` ไม่ตัดสิน production default

## Architecture Decisions

- **ไฟล์เดียว** `scripts/bakeoff.py` — สคริปต์วัดผล one-off ไม่มี test suite ไม่แตก package
- **adapter = dataclass เดียว** ต่อ model มี `extract(sentences) -> list[list[Triple]]`, Triple = `(head, relation, tail, score)` — score บังคับคืนเสมอ (แกนของ rate-once-slice-many)
- **rate-once-slice-many** — รัน inference ครั้งเดียว/model เก็บ raw scores แล้ว precision@threshold คำนวณจากการ rate ทุก unique triple ครั้งเดียว sweep ≥3 จุดได้ฟรี
- **pin ตาม memory**: `transformers==4.52.4`, `huggingface_hub<1.0` (glirel ต้องใช้, pin ผูกทั้ง env) — ถ้าชนกับ pyrheads/relik ให้แยก `uv sync --group <model>` แทนการรวม env เดียว
- **smoke-first**: ทุก model ผ่าน smoke (1 ประโยคเป้า LoRA/hallucination + ตรวจ score scale) ก่อนเขียน adapter เต็ม

## Task List

### Phase 1: Foundation

- [x] **Task 1: uv project + deps** — `bake-off/pyproject.toml` (deps: torch cu128, transformers==4.52.4, huggingface_hub<1.0, gliner, glirel, relik, spacy; `[project.scripts] bakeoff = "scripts.bakeoff:main"`) + `uv sync` + spacy `en_core_web_sm`

  **Acceptance criteria:**
  - [x] `uv run python -c "import torch, gliner, glirel, relik, spacy"` ผ่าน *(ปรับ: pin ชนตามคาด → แยก dependency-groups ต่อ model ตามที่ human ยืนยัน — ตรวจ import แยก per group: gliner group = torch/gliner/glirel/spacy ✓, relik group = torch/relik ✓)*
  - [x] `torch.cuda.is_available()` เป็น True บนเครื่อง

  **Verification:**
  - [x] `cd bake-off && uv sync` จบไม่ error
  - [x] `uv run python -m spacy download en_core_web_sm` จบไม่ error

  **Dependencies:** None · **Files:** `bake-off/pyproject.toml` · **Scope:** S

  > ถ้า transformers pin ชนกับ relik/pyrheads → หยุดถามก่อน แล้วแยก dependency-groups ต่อ model ตาม spec

- [x] **Task 2: ประโยค + seed schema (data)** — `schema/seed.json` (relation hints 10 อัน + entity labels จาก r1) + ชุดประโยคจาก `docs/bakeoff-results.md` แมป category (`alias`/`comparison`/`training`/`benchmark`/`effect`/`multi-rel`/`hard`) ครบทุกประโยค แต่ละ category ≥2 (เติม/ตัดได้) — ประโยคเป้า `(LoRA, reduces, hallucination)` คงไว้เป็น `alias`+`effect`

  **Acceptance criteria:**
  - [x] ทุกประโยคมี category กำกับ
  - [x] ทุก category มี ≥2 ประโยค *(r1 มีประโยค alias แค่อันเดียว — เติม 1 ประโยค CoT/CoT alias รวม 28)*
  - [x] script โหลด seed.json ได้ (relation hints ครบ 10) *(entity labels: เอกสาร r1 ไม่ได้บันทึกค่าเป๊ะ สคริปต์ r1 ไม่ได้กู้มา — derive จากผล r1 และกำกับ note ไว้ใน seed.json)*

  **Verification:**
  - [x] `uv run python -c "from scripts.bakeoff import load_data; ..."` พิมพ์นับ per-category แล้วครบเงื่อนไข

  **Dependencies:** Task 1 · **Files:** `bake-off/schema/seed.json`, ประโยคฝังใน `scripts/bakeoff.py` · **Scope:** S

### Checkpoint: Foundation
- [x] `uv sync` + import ครบ + CUDA พร้อม
- [x] data โหลดได้, category ครบ

### Phase 2: Framework + Smoke (fail fast ตัวเสี่ยงก่อน)

- [x] **Task 3: skeleton + adapter contract** — `Triple` dataclass, `GPU guard` (fail ถ้าไม่มี CUDA), helper warm-up + จับเวลา ms/ประโยค, dedupe triple ซ้ำเป๊ะต่อประโยค, CLI (`--smoke <model>`, `--only <model>`, default ทุก model) — adapter 5 ตัวเป็น stub

  **Acceptance criteria:**
  - [x] `uv run bakeoff` รัน stub ได้โดย fail ที่ GPU guard เมื่อไม่มี CUDA
  - [x] Triple บังคับมี score (dataclass field ไม่มี default)

  **Verification:**
  - [x] `uv run bakeoff --smoke gliner-relex` ไม่ crash (stub คืน [])

  **Dependencies:** Task 2 · **Files:** `scripts/bakeoff.py` · **Scope:** M

- [x] **Task 4: adapter gliner-relex** (known-working จาก r1 — ใช้ validate ทั้ง framework: sweep, timing, report path ก่อนแตะ model ที่เสี่ยง)

  **Acceptance criteria:**
  - [x] smoke 1 ประโยคเป้า: ได้ `(LoRA|Low-Rank Adaptation, reduces, hallucination)` + score
  - [x] `uv run bakeoff --only gliner-relex` จบ → เขียน report ชุดแรกออกมา *(ผ่าน stdout: 233 triples · 16.9 ms/ประโยค — ไฟล์ report/ มาที่ Task 8–9 ตามแผนเดิม)*

  **Verification:**
  - [x] ผล smoke ตรงกับ r1 (คล้าย ไม่จำเป็นต้องเป๊ะ) *(ได้ทั้ง Low-Rank Adaptation→hallucination 0.917 และ LoRA→hallucination 0.801, ms ตรง r1 ~16)*

  **Dependencies:** Task 3 · **Files:** `scripts/bakeoff.py` · **Scope:** S

- [x] **Task 5: smoke NuExtract 2.0** (ตัวเสี่ยงสุด — fp16 ~3GB บน 6GB ที่ใช้เป็นจอด้วย · ทำก่อนพวก encoder เพราะล้มได้ทั้งรอบ)

  **Acceptance criteria:**
  - [x] โหลด `numind/NuExtract-2.0-1.5B` fp16 บน GPU ได้ + จับ VRAM peak บันทึกไว้ *(checkpoint เปลี่ยนเป็น NuExtract-2.0-2B — 1.5B ไม่มีบน HF, human อนุมัติ · VRAM peak 4.39 GiB)*
  - [x] smoke คืน triple + score *(score = token prob เฉลี่ยเฉพาะ JSON object ของ triple, scale สูง 0.89–0.98)*

  **Verification:**
  - [x] `nvidia-smi` ระหว่างรัน — VRAM peak ไม่ชน headroom *(4.39 GiB / 6 GiB เหลือ ~1.6 GiB)*

  **Dependencies:** Task 4 · **Files:** `scripts/bakeoff.py` · **Scope:** S

- [ ] **Task 6: smoke ReLiK** — ทดสอบว่ายัด seed schema เป็น custom relation ได้ไหม

  **Acceptance criteria:**
  - [ ] ถ้าได้ → smoke คืน triple ด้วย relation จาก seed schema + score
  - [ ] ถ้าไม่ได้ → บันทึก "ปิด schema" ลง report เป็นข้อมูลของแผนที่ (ไม่ใช่ความล้มเหลว ไม่ต้องถาม)

  **Verification:**
  - [ ] ผลสรุป custom-relation support ถูกเก็บในโครงสร้างที่ report อ่านได้

  **Dependencies:** Task 3 · **Files:** `scripts/bakeoff.py` · **Scope:** S

- [ ] **Task 7: smoke GLiNER→GLiREL + GLiNER 2.x** — GLiREL ใช้ GLiNER checkpoint สายเดียวกับ Relex (ล็อกตั้งแต่ config); pyrheads อาจต้อง lib/install ต่างจาก `gliner` — smoke เป็นตัวตัดสิน

  **Acceptance criteria:**
  - [ ] glirel smoke คืน triple + score (จาก NER ต้นทางเดียวกับ Relex)
  - [ ] pyrheads smoke คืน triple + score หรือ error ชัดเจนว่าต้อง install อะไร → ถามก่อนเพิ่ม dependency

  **Verification:**
  - [ ] `uv run bakeoff --smoke glirel && uv run bakeoff --smoke gliner-pyrheads`

  **Dependencies:** Task 4 · **Files:** `scripts/bakeoff.py` · **Scope:** M

### Checkpoint: Smoke ครบ
- [ ] 5 model ผ่าน smoke (ReLiK ปิด schema ถือว่าผ่านในรูป "บันทึกข้อมูล")
- [ ] score scale ต่อ model ถูกตรวจด้วยตาแล้ว → ตั้งช่วง sweep ต่อ model
- [ ] ถ้ามี model ที่ต้องเพิ่ม dependency / ตัดออก → ถาม human ก่อน Phase 3

### Phase 3: Sweep + Reports

- [ ] **Task 8: threshold sweep (rate-once-slice-many)** — ต่อ model: รัน inference ครั้งเดียว เก็บ unique triples+scores, ตั้ง ≥3 threshold จุดต่อ model ตาม score scale ของมัน, นับ precision@threshold, เลือกจุด best

  **Acceptance criteria:**
  - [ ] inference ต่อ model 1 ครั้ง (ไม่รันซ้ำต่อ threshold)
  - [ ] sweep ≥3 จุด/model + จุด best ถูกเลือกด้วยมาตราของ model เอง
  - [ ] triple ซ้ำเป๊ะในประโยคถูก dedupe ก่อนนับ

  **Verification:**
  - [ ] `uv run bakeoff --only gliner-relex` → report มีตาราง sweep ≥3 แถว

  **Dependencies:** Tasks 4–7 · **Files:** `scripts/bakeoff.py` · **Scope:** M

- [ ] **Task 9: เขียน report** — `report/bakeoff-r2-results.md` (ตาราง Step 4: triples | ms/ประโยค | precision ~ | noise, sweep + best, แผนที่ 2 มิติ precision × ms, แตกผลตาม category) + `report/distinct-relations.md` (distinct relation count/model) — สร้างโดย script ตอนรัน

  **Acceptance criteria:**
  - [ ] ทั้ง 2 ไฟล์ถูกสร้าง/เขียนทับโดย `uv run bakeoff`
  - [ ] report แตกผลตาม category ได้ (ไม่รวมก้อนเดียว)
  - [ ] ms/ประโยค = ค่าหลัง warm-up

  **Verification:**
  - [ ] `uv run bakeoff` ครบ 5 model → เปิด report ตรวจด้วยตาตาม `docs/bakeoff-evaluation-steps.md` Step 3
  - [ ] distinct relation count ≥1 ทุก model (ReLiK ปิด schema ยกเว้น รูปแบบบันทึกข้อมูล)

  **Dependencies:** Task 8 · **Files:** `scripts/bakeoff.py`, `report/*.md` (generated) · **Scope:** M

### Checkpoint: Complete
- [ ] Success criteria 1–5 ของ spec ครบ
- [ ] ตรวจตัวเลขด้วยตา (Step 3) แล้ว
- [ ] ไม่มีไฟล์ใหม่หลุดออกนอก `bake-off/`

## Parallelization

- **Sequential เป็นหลัก** — GPU ตัวเดียว, env เดียว และทุก task แตะ `scripts/bakeoff.py` ไฟล์เดียวกัน
- Task 6 (ReLiK) กับ Task 7 (GLiREL/pyrheads) อ่าน-เขียนไม่ทับกัน แต่ VRAM/GPU แชร์กัน จึงรันตามลำดับง่ายกว่า

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NuExtract 2.0 ไม่ลงใน 6GB (จอใช้ VRAM ร่วม) | High | Task 5 ย้ายขึ้นก่อน — ล้มเร็ว, ถามก่อนตัด model |
| transformers pin ชนกับ relik/pyrheads | High | แยก `uv sync --group <model>` ตาม spec — ถามก่อนเปลี่ยน pin |
| pyrheads ต้อง lib ต่างจาก `gliner` | Med | smoke ตัดสิน, เพิ่ม dependency ต้องถามก่อน |
| ReLiK ไม่รับ custom relation | Low | ไม่ใช่ความล้มเหลว — บันทึก "ปิด schema" เป็นข้อมูล |
| encoder scores ไม่ deterministic พอ | Low | rate ทุก triple จาก run เดียว (rate-once) — ไม่มีจุดให้เพี้ยนข้าม run |

## Open Questions (ถามเมื่อถึงจุด)

- ถ้า NuExtract/ชน pin → ตัด model หรือแยก env group?
- จุด threshold sweep เอาเท่าไรต่อ model — ตั้งหลังเห็น score scale จาก smoke (ไม่ต้องถาม ถ้าเห็นชัด)
