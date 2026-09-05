"""gpu smoke — POST จริงผ่าน model จริง, วัดวินาที/job, slice หลาย threshold ไม่ rerun

รัน: uv run pytest -m gpu  (skip อัตโนมัติถ้าไม่มี CUDA)
"""

from __future__ import annotations

import time

import pytest
import torch
from fastapi.testclient import TestClient

from extraction_api.main import create_app
from extraction_api.settings import Settings

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="ไม่มี CUDA"),
]

PARAGRAPHS = [
    f"Paragraph {i}. Low-Rank Adaptation, or LoRA, reduces hallucination in large language models. "
    "We fine-tune LLaMA-7B on the Alpaca dataset using LoRA and evaluate on MMLU. "
    "Chain-of-thought prompting improves reasoning accuracy on multi-step problems."
    for i in range(10)
]


def body():
    return {"documents": [{"field": "text", "content": p} for p in PARAGRAPHS]}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpu")
    settings = Settings(db_path=tmp / "jobs.db", results_dir=tmp / "results")
    app = create_app(settings)  # spawn_worker=True — โหลด model ครั้งเดียวที่ lifespan
    with TestClient(app) as c:
        yield c, settings


def test_real_job_end_to_end_and_slice(client):
    c, _ = client
    t0 = time.perf_counter()
    jid = c.post("/jobs", json=body()).json()["job_id"]
    for _ in range(600):
        status = c.get(f"/jobs/{jid}").json()["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.5)
    elapsed = time.perf_counter() - t0
    print(f"\nwall-clock/job: {elapsed:.1f}s (10 paragraphs)")

    assert status == "done", f"job จบด้วย {status}"
    hi = c.get(f"/jobs/{jid}/triples").json()  # default threshold (0.9)
    assert hi["triples"], "threshold 0.9 ไม่ได้ triple เลย — ตรวจ model/threshold"

    # rate once, slice many — ไฟล์ผลเดิม ตัดที่ 0.5 ได้มากกว่า โดยไม่ rerun
    lo = c.get(f"/jobs/{jid}/triples", params={"threshold": 0.5}).json()
    assert len(lo["triples"]) >= len(hi["triples"])
    assert all(t["score"] >= 0.9 for t in hi["triples"])
    assert all(t["score"] >= 0.5 for t in lo["triples"])
