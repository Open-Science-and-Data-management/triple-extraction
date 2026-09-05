# Spec: Extraction API MVP — GLiNER-Relex as a Service

อ้างอิง idea: `docs/ideas/extraction-api-mvp.md` (Lean Job API — ตัดสินใจครบแล้วตามหัวข้อ Decision ด้านล่าง)

## Objective

Async API รับ academic papers (batch ≤ 20 ไฟล์, JSON field text) รัน GLiNER-Relex
(`knowledgator/gliner-relex-multi-v1.0`) เบื้องหลังบน GPU, คืน `job_id`,
ดึงผลด้วย threshold ที่เลือกภายหลัง **โดยไม่ rerun** (rate once, slice many —
เก็บ raw score ทุก triple, ตัดตอน read ด้วย `WHERE score >= threshold`)

- ผู้ใช้: consumer ภายใน network เดียวกัน (ยังไม่มี auth)
- สำเร็จ = POST job → poll/detect done → GET triples ที่ threshold ต่าง ๆ ได้จาก inference ครั้งเดียว, restart service แล้ว pending job ไม่หาย

## Decision Log (สิ่งที่ตกลงแล้ว)

| หัวข้อ | ตัดสินใจ |
|---|---|
| venv / deps | uv — root เป็น uv project ใหม่ แยกจาก bake-off (ไม่แตะ bake-off) |
| ที่อยู่โค้ด | `src/extraction_api/` ใน root ของ triple-extraction |
| model checkpoint | เก็บใน project ที่ `models/gliner-relex-multi-v1.0/` (ดาวน์โหลดครั้งเดียวด้วย `make download-model` / `uv run python -m extraction_api.download_model`) — `from_pretrained` อ่านจาก path นี้เสมอ ไม่ยิง HF ตอน runtime |
| ตั้งค่า | ไฟล์ `.env` (commit `.env.example` เป็นตัวอย่าง) — ทั้งค่าตัวเลขและ feature toggle เปิด/ปิด function (เช่น `WEBHOOK_ENABLED`) |
| **ที่เก็บผลลัพธ์** | **SQLite เก็บแค่ job id + status เท่านั้น (queue/track) — ผล triple เก็บเป็นไฟล์ JSON ต่อ job ที่ `data/results/{job_id}.json`** เพราะ job DB ไม่ใช่ store หลักสำหรับข้อมูลจำนวนมากจาก extraction; ตอนย้ายไป store จริง (downstream) ก็ย้ายแค่ไฟล์ — เหตุผลเต็มดู [ทำไมไฟล์ ไม่ใช่ DB](#ทำไมไฟล์-ไม่ใช่-db) ด้านล่าง |
| sentence split | pysbd |
| retention | **TTL เป็นหลัก + ขนาดรวมเป็นตัวสำรอง**: env `RETENTION_DAYS` (default 7, 0 = ปิด — ลบผลของ `done` ที่ `finished_at` เก่ากว่า) + `MAX_RESULTS_MB` (default 500 — เกินแล้วลบไฟล์ผลของ `done` เก่าสุดจนต่ำกว่าเพดาน); ลบได้เฉพาะ `done` เท่านั้น; `DELETE /jobs/{id}` ให้ consumer ลบเองเมื่อดึงแล้ว; prune วิ่งหลัง job แต่ละ job เสร็จใน worker loop (ไม่มี timer thread) |
| job limits | 20 ไฟล์ / 5 MB รวมต่อ job (422 ถ้าเกิน) |
| default schema | `bake-off/schema/seed.json` (10 relation hints + entity labels); ส่ง `seed_relations` มาทับได้ต่อ job |
| default threshold | 0.9 (query param) |

## Tech Stack

- Python >=3.12, จัดการด้วย uv (lock ด้วย `uv.lock` ที่ root)
- FastAPI + uvicorn — process เดียว
- SQLite (`data/jobs.db`) เก็บ **แค่ job id + status + callback_url** — queue/track เท่านั้น
- ไฟล์ JSON (`data/results/{job_id}.json`) เก็บผล raw triples ต่อ job — read แล้ว filter ด้วย threshold ใน memory
- in-process worker thread — โหลด model จาก `models/` บน CUDA ตอน startup, ดึง job ทีละอัน
- gliner + pins ชุดเดียวกับ bake-off group `gliner` (transformers==4.52.4, huggingface-hub<1.0, loguru, sentencepiece, protobuf, tiktoken — ตาม memory glirel-setup) + pysbd, pydantic-settings (.env), httpx (webhook), ruff, pytest
- torch cu128 wheel ตาม index config ใน bake-off/pyproject.toml (sm_89 + CUDA 13.3)

## Commands

```
Setup:    uv sync
Model:    uv run python -m extraction_api.download_model        # ดาวน์โหลด checkpoint → models/gliner-relex-multi-v1.0/ ครั้งเดียว
Dev:      uv run uvicorn extraction_api.main:app --host 0.0.0.0 --port 8000 --reload   # reload โหลด model ใหม่ทุกครั้ง — ใช้ตอนแก่ non-model code เท่านั้น
Prod:     uv run uvicorn extraction_api.main:app --host 0.0.0.0 --port 8000
Test:     uv run pytest                     # ไม่แตะ GPU — mock extractor
Test GPU: uv run pytest -m gpu              # smoke ผ่าน model จริง
Lint:     uv run ruff check src tests --fix
Format:   uv run ruff format src tests
```

## Project Structure

```
triple-extraction/
├── pyproject.toml              # uv project ของ API (root) — bake-off คงของเดิม
├── .env                        # gitignore — ตั้งค่าจริง
├── .env.example                # commit — ตัวอย่างค่าพร้อม comment อธิบายทุกตัวแปร
├── src/
│   └── extraction_api/
│       ├── main.py             # FastAPI app, lifespan: สร้าง DB → โหลด model จาก models/ → spawn worker thread
│       ├── settings.py         # pydantic-settings อ่าน .env (ดูตารางด้านล่าง)
│       ├── schemas.py          # Pydantic: POST /jobs body (documents[{field, content}], callback_url, seed_relations), status response, triples response
│       ├── db.py               # SQLite: table jobs เดียว — enqueue/claim/finish/prune (ไม่เก็บ triples)
│       ├── results.py          # เขียน (atomic: .tmp → rename)/อ่าน/ลบ data/results/{job_id}.json + filter ตาม threshold + prune ตาม TTL/ขนาด (อายุวัดจาก finished_at ใน DB ไม่ใช่ mtime)
│       ├── extractor.py        # port make_gliner_relex จาก bake-off: split ด้วย pysbd → inference @ threshold 0.3 → raw triples + provenance
│       ├── download_model.py   # snapshot_download → models/gliner-relex-multi-v1.0/
│       ├── worker.py           # loop: claim pending → status running → extract → เขียนไฟล์ผล → status done/failed → webhook
│       └── __init__.py
├── tests/
│   ├── test_api.py             # endpoint + validation ที่ trust boundary (FakeExtractor)
│   ├── test_results.py         # เขียน/อ่านผล JSON + filter-by-threshold (model-free)
│   ├── test_worker.py          # claim→done lifecycle, failed path, prune, webhook (FakeExtractor)
│   └── test_gpu_smoke.py       # marker `gpu`: POST job จริง 10 ย่อหน้า → done → triples ไม่ว่าง (วัด wall-clock ต่อ job ด้วย — assumption #1 ใน idea doc)
├── models/                     # gitignore — checkpoint ที่ดาวน์โหลดมา (~1.29 GiB)
├── data/                       # gitignore — jobs.db + results/*.json
├── bake-off/                   # ไม่แตะ (seed.json ที่อ่านต่ออยู่ที่นี่)
└── docs/
```

### ตัวแปรใน `.env` (ทั้งหมดมี default ใน settings.py — ไม่ตั้งก็รันได้)

| ตัวแปร | default | ความหมาย |
|---|---|---|
| `RETENTION_DAYS` | 7 | อายุผลลัพธ์ (วันหลัง `finished_at`) — เกินแล้วลบไฟล์ผล + แถว job (0 = ปิด TTL) |
| `MAX_RESULTS_MB` | 500 | ขนาดรวมของ `data/results/` — เกินแล้วลบผลของ `done` เก่าสุดจนต่ำกว่า (ตัวสำรองกันดิสก์เต็ม) |
| `WEBHOOK_ENABLED` | true | เปิด/ปิด callback ไป `callback_url` (toggle function) |
| `CALLBACK_TIMEOUT` | 10 | วินาทีรอ webhook ต่อครั้ง |
| `MAX_FILES` | 20 | จำนวนไฟล์สูงสุดต่อ job |
| `MAX_BYTES` | 5242880 | ขนาดรวม content สูงสุดต่อ job (5 MB) |
| `DEFAULT_THRESHOLD` | 0.9 | threshold เมื่อ GET /triples ไม่ส่ง query param |
| `MODEL_DIR` | models/gliner-relex-multi-v1.0 | path checkpoint ในเครื่อง |
| `DB_PATH` / `RESULTS_DIR` | data/jobs.db / data/results | ที่เก็บของ service |
| `DEVICE` | cuda | อ่านโมเดลบนอะไร (cpu สำหรับ debug ไม่มีการ์ด) |

### Storage

**SQLite (`jobs` table เดียว — id + status เท่านั้น):**

```sql
CREATE TABLE jobs (
  id TEXT PRIMARY KEY,             -- uuid4
  status TEXT NOT NULL DEFAULT 'pending',   -- pending|running|done|failed
  error TEXT,
  callback_url TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
```

**ผลลัพธ์ (ไฟล์ JSON ต่อ job — store หลักของ triples อยู่นอก DB):**

```json
{ "job_id": "...", "model_version": "knowledgator/gliner-relex-multi-v1.0",
  "seed_schema_hash": "sha256:...",
  "documents": [{"field": "text", "content": "..."}],
  "triples": [
    { "source_file": 0, "field": "text", "sentence": "LoRA reduces ...",
      "head": "LoRA", "head_type": "method", "tail": "hallucination",
      "tail_type": "concept", "relation": "reduces", "score": 0.93 }
  ] }
```

`GET /jobs/{id}/triples?threshold=0.9` → อ่านไฟล์ผลของ job → filter `[t for t in triples if t["score"] >= threshold]` — inference จบไปแล้วตอน worker เจอ job, ที่นี่เป็นแค่การอ่านไฟล์มากรอง

### ทำไมไฟล์ ไม่ใช่ DB

ทางเลือกเดียวที่จริงจังคือยัด triples ลง SQLite (มีอยู่แล้ว ไม่เพิ่ม dep) ให้ `WHERE job_id = ? AND score >= ?` กรองที่ index — แต่เลือกไฟล์เพราะ:

- **retention ถูกออกแบบเป็นหน่วยไฟล์**: `DELETE /jobs/{id}`, `RETENTION_DAYS`, `MAX_RESULTS_MB` ลบไฟล์ผลของ `done` เก่าสุด — ถ้าเก็บใน DB ทุก prune path กลายเป็น bulk DELETE หลายพันแถว + ไฟล์ DB บวมถ้าไม่ VACUUM
- **rate once, slice many ไม่ต้องการ index**: ผลต่อ job อยู่ในระดับ MB เดียว (≤ 20 ไฟล์ / 5 MB) — อ่านทั้งไฟล์มา filter ใน memory เร็วพอ index lookup ไม่ได้ชนะอะไร
- **แยก concern ตรงธรรมชาติของข้อมูล**: SQLite = queue/track (transactional, claim job), ไฟล์ = payload (เขียนครั้งเดียว atomic, อ่านเยอะ, downstream ย้ายเป็นไฟล์)
- retention ทั้งกฎคือ `rm data/results/{id}.json` — ไม่มี ORM, ไม่มี migration

**เงื่อนไขเปลี่ยน**: ผลต่อ job ใหญ่ระดับหลายสิบ MB หรือต้อง query ข้าม job — ตอนนั้นค่อยพิจารณา DuckDB ต่อ shard ก่อน SQLite ธรรมดา

## Code Style

- ไทยใน comment อธิบาย "ทำไม", อังกฤษใน identifier — ตามของเดิมใน repo
- type hints ครบ (`from __future__ import annotations`)
- ฟังก์ชันเล็ก flat, ไม่สร้าง class นอกจากที่ framework บังคับ (Pydantic models, app object)
- ตัวอย่างสไตล์ (extractor.py):

```python
def split_sentences(text: str) -> list[str]:
    """pysbd — academic text, ไม่พึ่ง model download"""
    return [s.strip() for s in Segmenter(language="en", clean=False).segment(text) if s.strip()]

def extract_raw(model, sentences: list[str], schema: SeedSchema) -> list[RawTriple]:
    """rate once: threshold ต่ำ 0.3 ตาม bake-off — เก็บทุก triple พร้อม score, slice ทีหลัง"""
    return [
        RawTriple(head=r["head"], head_type=r["head_type"], tail=r["tail"],
                  tail_type=r["tail_type"], relation=r["relation"], score=r["score"],
                  sentence=sent)
        for sent, rels in zip(sentences, batch_inference(model, sentences, schema))
        for r in rels
    ]
```

## Testing Strategy

- pytest, tests อยู่ `tests/` — ทุก test ที่ไม่มี marker `gpu` ต้องรันได้โดยไม่มี GPU/model (FakeExtractor inject เข้า app ผ่าน dependency)
- ระดับ:
  - unit: validation, filter-by-threshold จากไฟล์ผล, prune logic
  - integration (mock): POST→GET lifecycle ครบผ่าน `TestClient`
  - gpu smoke (`-m gpu`): 1 job จริง + วัด wall-clock — ใช้ตอบ assumption "เซิร์ฟเวอร์จะหนักไหม" ก่อนแตะ infra
- coverage: ไม่ตั้งเป้าเปอร์เซ็นต์ — ครอบทุก endpoint, ทุก status transition, และ trust-boundary validation ให้ครบพอ

## Boundaries

- **Always:** validation ที่ trust boundary ทุก endpoint (field รู้จัก, content string ไม่ว่าง, ≤ MAX_FILES, ≤ MAX_BYTES); เก็บ provenance ครบทุก triple (source_file, sentence, score, model_version, seed_schema_hash); อ่านค่า config ผ่าน settings เสมอ — ห้าม hardcode ค่าที่อยู่ใน .env; รัน `uv run pytest` ก่อน declare เสร็จ; inference ครั้งเดียว threshold 0.3 เสมอ — ห้าม rerun เพื่อเปลี่ยน threshold
- **Ask first:** เปลี่ยน schema SQLite หรือรูปแบบไฟล์ผล, เพิ่ม dependency ใหม่, expose ออกนอก network / เพิ่ม auth, แก้ไฟล์ใน `bake-off/`
- **Never:** auto-delete job ที่ยังไม่ done (prune ได้แค่ `done` เก่าสุด); แนบ secret/log ที่มีข้อมูล raw text ของผู้ใช้ออกนอก service (webhook ส่งแค่ job_id + status); commit `data/`, `models/`, `.env`

## Success Criteria

1. `uv sync && uv run pytest` เขียว (ไม่ต้องมี GPU); `uv run pytest -m gpu` เขียวบนเครื่องมี CUDA
2. POST /jobs ด้วย body ผิดรูป/เกิน limit → 422 พร้อมเหตุผลชัด; ถูกรูป → `job_id` ทันที
3. `GET /jobs/{id}` ครบ 4 status; job pending หลัง restart service ถูกหยิบไปทำต่อ
4. `GET /jobs/{id}/triples?threshold=t` กรองจากไฟล์ผลเดิมให้ `score >= t` เสมอ — เปลี่ยน t ได้ลูกเดียวไม่ rerun (ทดสอบด้วย test_results)
5. `WEBHOOK_ENABLED=true` → POST ไป `callback_url` เมื่อ job done/failed; `false` → ไม่ยิง (มี test ทั้งสองทาง)
6. Retention: ผลที่ `finished_at` เก่ากว่า `RETENTION_DAYS` ถูกลบทั้งแถวและไฟล์; รวมไฟล์ผลเกิน `MAX_RESULTS_MB` → ผลของ `done` เก่าสุดถูกลบจนต่ำกว่าเพดาน; pending/running/failed ไม่ถูกแตะทั้งสองกฎ; ไฟล์ผลที่ถูกอ่านได้ต้อง parse สำเร็จเสมอ (atomic write)
7. ทุก triple ที่คืนมี provenance ครบ 5 field
8. ไม่มี network call ไป Hugging Face หลัง startup — model อ่านจาก `models/` ล้วน

## Open Questions (ไม่บล็อก MVP)

- Deploy บน GPU จริงเมื่อไหร่ / การ์ดไหน — ใช้ test_gpu_smoke วัดก่อนตั้ง SLA
- Auth (API key) — เมื่อ expose นอก network ค่อยเพิ่ม
- Downstream จะดึงผลจาก `data/results/` ด้วยวิธีไหน (pull ผ่าน API เท่านั้น หรืออ่านไฟล์ตรง) — ตัดสินตอนมี consumer จริง
