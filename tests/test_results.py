"""results.py — เขียน/อ่าน atomic, filter-by-threshold, prune (model-free)"""

from __future__ import annotations

import pytest

from extraction_api.results import filter_triples, prune, read_result, write_result

NOW = "2026-09-05T12:00:00"


def payload(score=0.5):
    return {
        "job_id": "j1",
        "triples": [
            {"head": "a", "relation": "r", "tail": "b", "score": score},
            {"head": "c", "relation": "r", "tail": "d", "score": score + 0.4},
        ],
    }


# --- เขียน/อ่าน atomic ---
def test_write_then_read_roundtrip(tmp_path):
    write_result(tmp_path, "j1", payload())
    # ไฟล์ที่เขียน parse สำเร็จเสมอ (atomic — ไม่มีไฟล์ครึ่ง ๆ กลาง ๆ)
    assert read_result(tmp_path, "j1") == payload()


def test_write_is_atomic_leaves_no_tmp(tmp_path):
    write_result(tmp_path, "j1", payload())
    assert list(tmp_path.glob("*.tmp")) == []
    assert (tmp_path / "j1.json").exists()


# --- filter threshold ---
def test_filter_keeps_only_score_ge_threshold():
    p = payload(0.5)  # scores: 0.5, 0.9
    assert filter_triples(p, 0.91) == []
    assert len(filter_triples(p, 0.9)) == 1
    assert len(filter_triples(p, 0.5)) == 2
    # ผลต่าง threshold ต่างกัน จากไฟล์เดิม
    assert filter_triples(p, 0.9) != filter_triples(p, 0.5)


def test_read_missing_returns_none(tmp_path):
    assert read_result(tmp_path, "nope") is None
    with pytest.raises(TypeError):
        filter_triples({"triples": None}, 0.5)


# --- prune ---
def make_done(tmp_path, job_id, finished_at, pad=0):
    """สร้างผล done ที่คุม finished_at และขนาดไฟล์ (pad = ไบต์ที่โปะเพิ่ม)"""
    p = payload()
    if pad:
        p["pad"] = "x" * pad
    write_result(tmp_path, job_id, p)
    return {"id": job_id, "status": "done", "finished_at": finished_at}


def test_prune_ttl_removes_only_old_done(tmp_path):
    jobs = [
        make_done(tmp_path, "old", "2026-08-01T00:00:00"),
        make_done(tmp_path, "fresh", "2026-09-05T00:00:00"),
        {"id": "pend", "status": "pending", "finished_at": "2026-08-01T00:00:00"},
        {"id": "fail", "status": "failed", "finished_at": "2026-08-01T00:00:00"},
    ]
    removed = prune(tmp_path, jobs, now=NOW, retention_days=7, max_results_mb=0)
    assert (tmp_path / "old.json").exists() is False
    assert (tmp_path / "fresh.json").exists()
    assert removed == ["old"]
    # pending/failed เก่ากว่า TTL ก็ไม่ถูกแตะ (ไม่มีไฟล์อยู่แล้ว แต่ต้องไม่ติดใน removed)
    assert "pend" not in removed and "fail" not in removed


def test_prune_ttl_disabled(tmp_path):
    jobs = [make_done(tmp_path, "old", "2020-01-01T00:00:00")]
    assert prune(tmp_path, jobs, now=NOW, retention_days=0, max_results_mb=0) == []
    assert (tmp_path / "old.json").exists()


def test_prune_size_caps_removes_oldest_done_first(tmp_path):
    jobs = [
        make_done(tmp_path, "d1", "2026-09-01T00:00:00", pad=600_000),
        make_done(tmp_path, "d2", "2026-09-02T00:00:00", pad=600_000),
        make_done(tmp_path, "d3", "2026-09-03T00:00:00", pad=600_000),
    ]
    # รวม ~1.8MB เกินเพดาน 1MB → ลบจนต่ำกว่า: d1 หายก่อน, ยังเกิน → d2 ตาม, d3 รอด
    removed = prune(tmp_path, jobs, now=NOW, retention_days=0, max_results_mb=1)
    assert removed == ["d1", "d2"]  # เก่าสุดก่อน จนต่ำกว่าเพดาน
    assert (tmp_path / "d3.json").exists()


def test_prune_size_cap_keeps_fresh_when_under_ceiling(tmp_path):
    jobs = [
        make_done(tmp_path, "a", "2026-09-01T00:00:00"),
        make_done(tmp_path, "b", "2026-09-02T00:00:00"),
    ]
    # ไฟล์เล็กมาก — เพดาน 500MB ไม่ควรลบอะไร
    assert prune(tmp_path, jobs, now=NOW, retention_days=0, max_results_mb=500) == []
