"""extractor.py — ยูนิต strip ตาราง + filter field + provenance (stub model ไม่แตะ GPU)"""

from __future__ import annotations

import pytest

from extraction_api.extractor import _strip_table_rows, extract_raw

SCHEMA = {"entity_labels": ["method", "metric"], "relation_hints": ["evaluates"]}

REL = {
    "head": {"text": "H", "type": "method"},
    "tail": {"text": "T", "type": "metric"},
    "relation": "evaluates",
    "score": 0.9,
}


class StubModel:
    """คืน relation 1 ตัวต่อประโยค — ใช้นับว่าประโยคไหนเข้า inference"""

    def __init__(self):
        self.seen: list[str] = []

    def inference(self, sentences, labels, **kw):
        self.seen.extend(sentences)
        return None, [[dict(REL)] for _ in sentences]  # รูปทรงเดียวกับ gliner: relations ต่อประโยค


# --- _strip_table_rows ---
def test_strip_table_rows_basic():
    html = "<table><tr><td>F1</td><td>58.3</td></tr><tr><td>Acc</td><td>91.2</td></tr></table>"
    assert _strip_table_rows(html) == ["F1 58.3", "Acc 91.2"]


def test_strip_table_rows_th_and_empty_cells():
    html = "<table><tr><th>Model</th><th> </th></tr><tr><td> </td><td>7.1</td></tr></table>"
    # แถวที่ join แล้วว่าง = ตัดทิ้ง, cell ว่าง join ไม่เหลือช่องว่างซ้ำ
    assert _strip_table_rows(html) == ["Model", "7.1"]


def test_strip_table_rows_no_table_tag():
    assert _strip_table_rows("plain text") == ["plain text"]


def test_strip_table_rows_empty():
    assert _strip_table_rows("") == []


# --- filter field + provenance ---
def test_extracts_only_text_table_figure_caption():
    docs = [
        {"field": "latex", "content": r"\frac{a}{b}", "section": None},
        {"field": "image", "content": "fig1.png", "section": None},
        {"field": "section", "content": "3 Results", "section": None},
    ]
    model = StubModel()
    assert extract_raw(model, docs, SCHEMA) == []
    assert model.seen == []  # ไม่มีประโยคเหลือให้ inference


def test_provenance_matches_parent_document():
    docs = [
        {"field": "text", "content": "Alpha beta.", "section": "3.1 Setup"},
        {"field": "table", "content": "<table><tr><td>x</td><td>1</td></tr></table>", "section": "3.2 Results"},
    ]
    triples = extract_raw(StubModel(), docs, SCHEMA)
    assert len(triples) == 2
    by_field = {t["field"]: t for t in triples}
    assert by_field["text"]["source_file"] == 0
    assert by_field["text"]["section"] == "3.1 Setup"
    assert by_field["table"]["source_file"] == 1
    assert by_field["table"]["section"] == "3.2 Results"
    assert by_field["table"]["sentence"] == "x 1"  # strip แล้ว ไม่ใช่ HTML


@pytest.mark.parametrize("field", ["text", "table", "figure_caption"])
def test_figure_caption_goes_through(field):
    docs = [{"field": field, "content": "content here", "section": None}]
    triples = extract_raw(StubModel(), docs, SCHEMA)
    assert len(triples) == 1
    assert triples[0]["field"] == field
