"""GLiNER-Relex inference — rate once @ threshold 0.3, slice ทีหลังด้วย threshold ใดก็ได้"""

from __future__ import annotations

import hashlib
import json
import os
from html.parser import HTMLParser
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


class _TableParser(HTMLParser):
    """เก็บ <tr> เป็นประโยค — cell ทั้งแถว join ด้วยช่องว่าง"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None:
            text = " ".join("".join(c).strip() for c in self._row).strip()
            if text:
                self.rows.append(text)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append("".join(self._cell))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _strip_table_rows(html: str) -> list[str]:
    # ponytail: ไม่ prefix ชื่อ column — GLiNER จับ relation จากแถวล้วนพอหรือไม่ รอ GPU test (Task 5)
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows and html.strip():
        return [html.strip()]  # ไม่ใช่ table tag — เก็บทั้งก้อนไว้ อย่าทำ triple หาย
    return parser.rows


# GLiNER extract เฉพาะ 3 field นี้ — latex/image/section pass-through เก็บในไฟล์ผลอย่างเดียว
EXTRACTABLE_FIELDS = {"text", "table", "figure_caption"}


def extract_raw(
    model: Any,
    documents: list[dict[str, str]],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """rate once: เก็บทุก triple พร้อม score + provenance — slice ทีหลังที่ GET /triples"""
    labels, relations = schema["entity_labels"], schema["relation_hints"]

    def sentences_of(doc: dict[str, str]) -> list[str]:
        if doc["field"] == "table":
            return _strip_table_rows(doc["content"])
        return split_sentences(doc["content"])

    # ประโยคทั้ง job ยัด batch เดียว — batch_size 8 ใน inference จัดการเอง
    flat: list[tuple[int, str, str | None, str]] = [
        (i, doc["field"], doc.get("section"), sent)
        for i, doc in enumerate(documents)
        if doc["field"] in EXTRACTABLE_FIELDS
        for sent in sentences_of(doc)
    ]
    if not flat:
        return []

    _, rels = model.inference(
        [s for _, _, _, s in flat],
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
            "section": section,
            "sentence": sent,
            "head": r["head"]["text"],
            "head_type": r["head"].get("type", ""),
            "tail": r["tail"]["text"],
            "tail_type": r["tail"].get("type", ""),
            "relation": r["relation"],
            "score": float(r["score"]),
        }
        for (src_idx, field, section, sent), rs in zip(flat, rels)
        for r in rs
    ]
