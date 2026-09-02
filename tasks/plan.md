# Plan: REBEL Triple Extraction API

> Spec: `docs/specs/SPEC-rebel-triple-api.md` · อนุมัติแล้ว 2026-09-02

## สถาปัตยกรรมและลำดับการสร้าง

```
settings ──→ db ──→ extractor/rebel ──→ worker ──→ api ──→ smoke/gpu-verify
   (1)       (2)         (3)             (4)       (5)         (6)
```

ทุก slice เดินจากลึกไปตื้น: config → storage → brain → loop → HTTP → ปลายทางจริง
แต่ละขั้นมี test ของตัวเองและไม่ต้องโหลดโมเดลจริง (mock จนถึงขั้นสุดท้าย)

## ความเสี่ยงและการรับมือ

| ความเสี่ยง | ผลกระทบ | การรับมือ |
|---|---|---|
| torch release ล่าสุดไม่รองรับ sm_120 (RTX 5060 Ti) | `uv sync` แล้ว cuda ใช้ไม่ได้ | pin torch cu128 wheel จาก index `pytorch-cu128`; ตรวจด้วย `torch.cuda.is_available()` ทันทีหลัง sync |
| REBEL decoder output รูปแบบเบี่ยงเบน (คอลัมน์ขาด/เกิน) | parse พัง ทิ้ง triples | parser ต้อง tolerate: เก็บ raw ใน `unparsed`, unit test ครบ edge case |
| spaCy split text-from-PDF พัง (line-break/หัวข้อ) | triples แปลก ๆ / offset เพี้ยน | gpu smoke test กับเอกสารจริง 1 ฉบับ + บันทึกผลเป็นหลักฐาน (assumption #1 ใน idea doc) |
| SQLite lock ระหว่าง worker thread ↔ API thread | `database is locked` | WAL mode + `busy_timeout` + ผูก connection ต่อ thread |
| โมเดลโหลดช้า/กิน VRAM ตอนยังโหลดซ้ำ | server สตาร์ทช้า, OOM | โหลดโมเดล 1 ครั้งใน worker, serialize inference (worker เดียวอยู่แล้ว) |

## การตัดสินใจเชิงเทคนิค

- **Worker เป็น background thread ใน process เดียวกับ uvicorn** (ไม่ใช่ process แยก)
  — ง่าย, แชร์ SQLite file ผ่าน WAL, ตอบโจทย์ "worker เดียวพอ" ของ idea doc
- **Job recovery ตอนบูต:** mark ทุก job `processing` ที่เหลือค้างเป็น `failed`
  (ยอมตาม idea doc — ไม่ re-queue เพื่อความเรียบง่าย)
- **Extractor protocol แบบบาง:** `Protocol` เดียวมีเมธอด `extract(text) -> list[triple]`
  — เตรียม backend อื่นโดยไม่สร้าง abstraction หลายชั้น (Not-doing list ใน spec)
- **โมเดล lazy-load ใน worker thread** ไม่ใช่ตอน import — test ไม่พัง, startup ไม่ค้าง

## Checkpoints ระหว่างทาง

1. หลังขั้น (1)–(2): `uv run pytest tests/test_db.py` เขียว — schema/recovery ถูก
2. หลังขั้น (3): parser unit test เขียว — จัดการ edge case ครบ
3. หลังขั้น (4)–(5): API integration test เขียวทั้ง lifecycle + error cases
4. ขั้น (6): startup พิมพ์ device, `-m gpu` ผ่าน, ยิงเอกสารจริงได้ triples จริง

## ขนานได้ / ต้องตามลำดับ

- ต้องตามลำดับ: (1)→(2)→(3)→(4)→(5) (dependency ตรง ๆ)
- ขนานได้: test parser edge cases เขียนคู่กับ implementation ตาม TDD
