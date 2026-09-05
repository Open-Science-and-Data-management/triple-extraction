"""Pydantic request/response — validation ที่ trust boundary ทุก endpoint"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# field enum — output ของ PaddleOCR ครบ 6 field; GLiNER extract เฉพาะ 3 แรก ที่เหลือ pass-through
KnownField = Literal["text", "table", "figure_caption", "latex", "image", "section"]


class Document(BaseModel):
    field: KnownField
    content: str = Field(min_length=1)
    section: str | None = None


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "documents": [
                        {
                            "field": "text",
                            "content": (
                                "We evaluate GLiNER-Relex on the SciERC benchmark and find that "
                                "joint entity–relation extraction outperforms the pipeline baseline, "
                                "achieving an F1 score of 58.3."
                            ),
                            "section": "1 Introduction",
                        },
                        {
                            "field": "table",
                            "content": (
                                "<table><tr><td>Model</td><td>F1</td></tr>"
                                "<tr><td>GLiNER-Relex</td><td>58.3</td></tr></table>"
                            ),
                            "section": "4 Results",
                        },
                        {
                            "field": "figure_caption",
                            "content": "Figure 1: Architecture of the joint entity–relation model.",
                            "section": "3 Method",
                        },
                        {"field": "latex", "content": r"\mathcal{L} = \mathcal{L}_{ent} + \mathcal{L}_{rel}", "section": None},
                        {"field": "image", "content": "figures/architecture.png", "section": None},
                        {"field": "section", "content": "5 Conclusion and Future Work", "section": None},
                    ],
                    "callback_url": None,
                    "seed_relations": None,
                }
            ]
        }
    )

    documents: list[Document]
    callback_url: str | None = None
    # ไม่ส่ง = ใช้ default schema จาก bake-off/schema/seed.json
    seed_relations: list[str] | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    error: str | None = None


class TripleOut(BaseModel):
    source_file: int
    field: str
    section: str | None = None
    sentence: str
    head: str
    head_type: str
    tail: str
    tail_type: str
    relation: str
    score: float


class TriplesResponse(BaseModel):
    job_id: str
    threshold: float
    model_version: str
    seed_schema_hash: str
    triples: list[TripleOut]


def validate_job_request(req: CreateJobRequest, max_files: int, max_bytes: int) -> None:
    """limit เป็นค่าจาก settings — endpoint ต้องส่งมา, 422 ที่ HTTP layer"""
    if len(req.documents) > max_files:
        raise ValueError(f"documents เกิน MAX_FILES={max_files} (ได้ {len(req.documents)})")
    total = sum(len(d.content.encode()) for d in req.documents)
    if total > max_bytes:
        raise ValueError(f"content รวมเกิน MAX_BYTES={max_bytes} (ได้ {total} ไบต์)")


def triples_payload_dict(payload: dict, threshold: float) -> dict[str, Any]:
    """ตัด triples ตาม threshold จากไฟล์ผลเดิม — ใช้ตอบ GET /triples"""
    return {**payload, "threshold": threshold}
