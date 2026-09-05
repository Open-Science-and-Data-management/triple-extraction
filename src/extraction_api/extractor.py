"""GLiNER-Relex inference — rate once @ threshold 0.3, slice ทีหลังด้วย threshold ใดก็ได้"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# อ่าน model จาก models/ ล้วน — ห้ามยิง HF ตอน runtime (success criteria 8)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# threshold 0.3 ทั้งชุดตาม bake-off r1 "lo" — ค่าคงที่ของ design (rate once) ไม่ใช่ config
EXTRACTION_THRESHOLD = 0.3
BATCH_SIZE = 8

SEED_SCHEMA_PATH = Path("bake-off/schema/seed.json")
MODEL_VERSION = "knowledgator/gliner-relex-multi-v1.0"


def load_seed_schema(path: Path = SEED_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def schema_hash(schema: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def split_sentences(text: str) -> list[str]:
    """pysbd — academic text, ไม่พึ่ง model download"""
    from pysbd import Segmenter

    return [s.strip() for s in Segmenter(language="en", clean=False).segment(text) if s.strip()]


def load_extractor(model_dir: Path | None = None, device: str | None = None):
    from gliner import GLiNER

    from extraction_api.settings import Settings

    settings = Settings()
    model = GLiNER.from_pretrained(str(model_dir or settings.model_dir))
    return model.to(device or settings.device).eval()


def extract_raw(
    model: Any,
    documents: list[dict[str, str]],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """rate once: เก็บทุก triple พร้อม score + provenance — slice ทีหลังที่ GET /triples"""
    labels, relations = schema["entity_labels"], schema["relation_hints"]

    # ประโยคทั้ง job ยัด batch เดียว — batch_size 8 ใน inference จัดการเอง
    flat: list[tuple[int, str, str]] = [
        (i, doc["field"], sent) for i, doc in enumerate(documents) for sent in split_sentences(doc["content"])
    ]
    if not flat:
        return []

    _, rels = model.inference(
        [s for _, _, s in flat],
        labels,
        relations=relations,
        threshold=EXTRACTION_THRESHOLD,
        adjacency_threshold=EXTRACTION_THRESHOLD,
        relation_threshold=EXTRACTION_THRESHOLD,
        batch_size=BATCH_SIZE,
        return_relations=True,
    )

    return [
        {
            "source_file": src_idx,
            "field": field,
            "sentence": sent,
            "head": r["head"]["text"],
            "head_type": r["head"].get("type", ""),
            "tail": r["tail"]["text"],
            "tail_type": r["tail"].get("type", ""),
            "relation": r["relation"],
            "score": float(r["score"]),
        }
        for (src_idx, field, sent), rs in zip(flat, rels)
        for r in rs
    ]
