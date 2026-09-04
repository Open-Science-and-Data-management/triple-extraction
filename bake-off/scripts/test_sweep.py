"""self-check logic sweep (rate-once-slice-many) — รัน: uv run python -m scripts.test_sweep"""

from scripts.bakeoff import best_point, collect_rows, slice_at, sweep_table
from scripts.bakeoff import Triple

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
