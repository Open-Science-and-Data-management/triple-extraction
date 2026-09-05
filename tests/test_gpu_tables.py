"""GPU — GLiNER จับ relation จากตาราง strip เป็นรายแถวได้ไหม (open question: prefix ชื่อ column)

รัน: uv run pytest -m gpu -s tests/test_gpu_tables.py  (skip อัตโนมัติถ้าไม่มี CUDA)

วัด 2 variant: strip ล้วน vs แถวที่ prefix ชื่อ column — พิมพ์จำนวน triple เทียบ
แต่ "ไม่ assert ว่า variant ไหนชนะ" — สรุปตัวเลขให้ผู้ใช้ตัดสินก่อนเปลี่ยน strip logic
"""

from __future__ import annotations

import pytest
import torch

from extraction_api.extractor import extract_raw, load_extractor, load_seed_schema

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="ไม่มี CUDA"),
]

# ตารางจริงจาก bake-off/report/bakeoff-r2-results.md — แปลงเป็น HTML แบบที่ PaddleOCR emit
TABLES = {
    "model-compare": {
        "headers": ["model", "precision", "ms_per_sentence", "vram_peak", "schema"],
        "rows": [
            ["gliner-relex", "~0.88 @ th 0.9", "15.5", "1.29 GiB", "seed schema"],
            ["glirel", "~0.14 @ th 0.25", "31.2", "3.32 GiB", "seed schema"],
            ["gliner-pyrheads", "~0.53 @ th 0.85", "46.7", "2.09 GiB", "seed schema"],
            ["relik", "~0.00 @ th 0.5", "151.0", "1.79 GiB", "closed native NYT"],
            ["nuextract", "~1.00 @ th 0.98", "976.6", "4.39 GiB", "seed schema"],
        ],
    },
    "relex-by-threshold": {
        "headers": ["threshold", "triples", "precision", "empty_sentences"],
        "rows": [
            ["0.3", "233", "~0.39", "0"],
            ["0.5", "139", "~0.52", "0"],
            ["0.7", "92", "~0.66", "1"],
            ["0.9", "33", "~0.88", "9"],
        ],
    },
    "category-effect": {
        "headers": ["category", "triples", "empty_of_total"],
        "rows": [
            ["alias", "4", "0/2"],
            ["comparison", "10", "0/4"],
            ["training", "7", "2/5"],
            ["benchmark", "3", "1/3"],
            ["effect", "8", "1/7"],
            ["multi-rel", "5", "0/4"],
            ["hard", "3", "5/7"],
        ],
    },
}


def to_html(t: dict, prefix_headers: bool = False) -> str:
    """prefix_headers=True → ฝังชื่อ column นำหน้าค่าทุก cell (คง 1 ประโยค/แถว)"""

    def cells(row: list[str]) -> str:
        if prefix_headers:
            row = [f"{h} {c}" for h, c in zip(t["headers"], row)]
        return "".join(f"<td>{c}</td>" for c in row)

    trs = "".join(f"<tr>{cells(row)}</tr>" for row in t["rows"])
    return f"<table>{trs}</table>"


@pytest.fixture(scope="session")
def model():
    return load_extractor().eval()


def test_table_rows_vs_prefixed_headers(model, tmp_path):
    schema = load_seed_schema()
    print()
    total = {"rows-only": 0, "prefixed": 0}
    seen: list[dict] = []
    for name, t in TABLES.items():
        # variant A: strip ล้วน (รวมแถว header) — ตามที่ production ทำอยู่
        html_a = to_html(t)
        docs_a = [{"field": "table", "content": html_a, "section": f"table {name}"}]
        # variant B: 1 ประโยค/แถว เท่าเดิม แต่ทุก cell มีชื่อ column นำหน้า — header แนบ context ไปกับแถว
        docs_b = [{"field": "table", "content": to_html(t, prefix_headers=True), "section": f"table {name}"}]

        triples_a = extract_raw(model, docs_a, schema)
        triples_b = extract_raw(model, docs_b, schema)
        total["rows-only"] += len(triples_a)
        total["prefixed"] += len(triples_b)
        seen.extend(triples_a + triples_b)
        print(f"{name}: rows-only={len(triples_a)}  prefixed={len(triples_b)}")
        for tag, triples in (("A/rows-only", triples_a), ("B/prefixed", triples_b)):
            for tr in triples:
                print(f"  [{tag}] ({tr['score']:.2f}) {tr['head']} --{tr['relation']}--> {tr['tail']}")

    print(f"\nรวม: {total}")
    # assert ระดับต่ำ — provenance ครบ ไม่ตัดสินผู้ชนะ
    assert seen, "สอง variant ไม่ได้ triple เลย — ตรวจ model/ตาราง"
    assert all(tr["field"] == "table" for tr in seen)
    assert all(tr["section"] and tr["section"].startswith("table ") for tr in seen)
