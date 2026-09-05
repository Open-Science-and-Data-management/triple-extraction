# extraction-api

Async API สกัด knowledge-graph triples จากเอกสารด้วย [GLiNER-Relex](https://github.com/urchade/GLiNER) ตัวเดียว — โหลด model ครั้งเดียวตอน startup, slice หลายอันโดยไม่ rerun

- **Python 3.12+ / uv / FastAPI / SQLite** — ไม่มี external broker/redis
- Queue เป็น SQLite (`data/jobs.db`), ผลลัพธ์เป็นไฟล์ JSON ต่อ job (`data/results/`)
- Swagger UI: **http://localhost:8000/docs** (มีให้จาก FastAPI โดย default)

## ติดตั้ง + รัน

```bash
uv sync                                  # deps (torch cu128, gliner + pins)
uv run python -m extraction_api.download_model  # ดึง model ลง models/gliner-relex-multi-v1.0
uv run uvicorn extraction_api.main:app          # device=cuda default
```

## API

| Method | Path | คำอธิบาย |
|---|---|---|
| POST | `/jobs` | ส่ง documents → ได้ `job_id` (201) |
| GET | `/jobs/{id}` | สถานะ `pending / running / done / failed` |
| GET | `/jobs/{id}/triples?threshold=0.9` | ผลลัพธ์ (ต้อง done) — filter ตาม score ได้ทุกครั้ง ไม่ต้อง extract ใหม่ |
| DELETE | `/jobs/{id}` | ลบ job + ไฟล์ผล |

```bash
curl -X POST localhost:8000/jobs -H 'content-type: application/json' -d '{
  "documents": [{"field": "text", "content": "JanusGraph stores data in Cassandra."}]
}'
# → {"job_id": "…"}
curl "localhost:8000/jobs/<id>/triples?threshold=0.9"
```

- `callback_url` (optional) — ยิง POST `{job_id, status}` เมื่อ job เสร็จ ไม่กระทบ status ถ้าพัง
- `seed_relations` (optional) — ทับ `relation_hints` ใน `bake-off/schema/seed.json` เฉพาะ job นั้น
- validation: ≤ `MAX_FILES=20` documents, รวม content ≤ `MAX_BYTES=5MB`, threshold ∈ [0,1] → 422

## Config (.env — มี default ครบ, ไม่มีไฟล์ก็รันได้)

| ตัวแปร | Default | ความหมาย |
|---|---|---|
| `DEVICE` | `cuda` | torch device |
| `MODEL_DIR` | `models/gliner-relex-multi-v1.0` | ที่อยู่ model |
| `DB_PATH` | `data/jobs.db` | SQLite queue |
| `RESULTS_DIR` | `data/results` | ไฟล์ผล JSON |
| `DEFAULT_THRESHOLD` | `0.9` | filter ตอน GET /triples ไม่ส่ง threshold |
| `MAX_FILES` / `MAX_BYTES` | `20` / `5242880` | ลิมิตต่อ request |
| `WEBHOOK_ENABLED` / `CALLBACK_TIMEOUT` | `true` / `10.0` | webhook |
| `RETENTION_DAYS` / `MAX_RESULTS_MB` | `7` / `500` | prune หลังทุก job เสร็จ |

## สถาปัตยกรรม

```
POST /jobs ──▶ SQLite queue ──▶ worker thread (daemon)
                                   claim → extract (GLiNER-Relex, slice ไม่ rerun)
                                   → เขียนไฟล์ผล atomic → done/failed
                                   → webhook → prune (retention + size cap)
GET /triples ◀── ไฟล์ผล JSON (filter threshold ที่หน้า endpoint)
```

- job ไหนพัง → `failed` + error, worker วนต่อ
- restart ระหว่างมี pending/running ค้าง → worker กลับมา claim ต่อเองตอน startup
- โค้ด: `src/extraction_api/` — `main.py` (app/endpoints), `db.py` (queue), `worker.py` (loop), `extractor.py` (model), `results.py` (ไฟล์ผล + prune), `schemas.py`, `settings.py`

## Test

```bash
uv run pytest            # ยูนิต/API — fake extractor ไม่แตะ GPU
uv run pytest -m gpu     # smoke แตะ model จริง (~1.5s/job / 10 ย่อหน้า)
uv run ruff check .
```
