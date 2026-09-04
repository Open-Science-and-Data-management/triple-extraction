"""self-check logic sweep (rate-once-slice-many) — รัน: uv run python -m scripts.test_sweep"""

import tempfile
from pathlib import Path

from scripts.bakeoff import Triple, best_point, collect_rows, slice_at, sweep_table, write_reports

# 3 unique triple: ถูก 1 (2 ประโยค), ผิด 1, ยังไม่ rate 1
ROWS = [
    {"head": "LoRA", "relation": "reduces", "tail": "hallucination", "correct": True,
     "occ": [[0, 0.9], [1, 0.4]]},
    {"head": "GPT-4", "relation": "outperforms", "tail": "GPT-3.5", "correct": False,
     "occ": [[0, 0.6]]},
    {"head": "BERT", "relation": "pretrained on", "tail": "BookCorpus", "correct": None,
     "occ": [[2, 0.2]]},
]


def test_collect_rows_dedupes_and_merges_occurrences():
    per_sent = [
        [Triple("LoRA", "reduces", "hallucination", 0.9), Triple("LoRA", "reduces", "hallucination", 0.7)],
        [Triple("LoRA", "reduces", "hallucination", 0.4)],
    ]
    rows = collect_rows(per_sent)
    assert len(rows) == 1
    assert rows[0]["occ"] == [[0, 0.9], [1, 0.4]]  # เก็บ score ทุกเหตุการณ์ ไม่ merge ทิ้ง


def test_slice_at_counts_occurrences_above_threshold():
    n, n_ok, empty = slice_at(ROWS, 0.5, n_sents=3)
    assert n == 2          # occ 0.9 กับ 0.6 (0.4, 0.2 ตก)
    assert n_ok == 1       # ที่ถูกคืออันเดียว
    assert empty == [1, 2]  # ประโยค 1 (occ 0.4) กับ 2 (occ 0.2) ไม่มี triple ผ่าน threshold


def test_slice_at_precision_none_while_unrated():
    (table,) = sweep_table(ROWS, [0.5], n_sents=3)
    assert table["precision"] is None  # rate ไม่ครบ → ไม่อ้าง precision


def test_best_point_precision_first_then_most_triples():
    table = [
        {"threshold": 0.3, "n": 10, "precision": 0.60, "empty": 1},
        {"threshold": 0.5, "n": 8, "precision": 0.75, "empty": 2},
        {"threshold": 0.9, "n": 5, "precision": 0.75, "empty": 4},
    ]
    assert best_point(table)["threshold"] == 0.5  # เสมอกัน → เอาจุดที่เหลือ triple มากสุด


def test_best_point_none_while_unrated():
    assert best_point(sweep_table(ROWS, [0.5, 0.8], n_sents=3)) is None


SENTS = [
    {"text": "s0", "categories": ["alias", "effect"]},
    {"text": "s1", "categories": ["training"]},
    {"text": "s2", "categories": ["hard"]},
]

RESULT = {"name": "m1", "ms": 15.5, "vram": "VRAM peak: 1.29 GiB",
          "triples": ROWS[:2] + [{"head": "x", "relation": "pretrained on", "tail": "y",
                                  "correct": True, "occ": [[1, 0.8]]}],
          "table": [{"threshold": 0.3, "n": 3, "precision": 0.67, "empty": [2]},
                    {"threshold": 0.5, "n": 2, "precision": 0.50, "empty": [1, 2]}],
          "best": {"threshold": 0.3, "n": 3, "precision": 0.67, "empty": [2]}}


def test_write_reports_creates_both_files_with_sweep_and_category_split():
    with tempfile.TemporaryDirectory() as d:
        write_reports([RESULT], SENTS, report_dir=Path(d))
        results = (Path(d) / "bakeoff-r2-results.md").read_text()
        distinct = (Path(d) / "distinct-relations.md").read_text()
    assert "0.3" in results and "0.5" in results          # ตาราง sweep ≥2 แถว
    assert all(c in results for c in ("alias", "training", "hard"))  # แตกตาม category
    assert "15.5" in results                              # ms หลัง warm-up
    assert "pretrained on" in distinct                    # distinct relation


def test_distinct_relations_counts_unique():
    with tempfile.TemporaryDirectory() as d:
        write_reports([RESULT], SENTS, report_dir=Path(d))
        distinct = (Path(d) / "distinct-relations.md").read_text()
    # ROWS[:2] มี reduces, outperforms + pretrained on = 3 distinct สำหรับ m1
    assert "| m1 | 3 |" in distinct


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
