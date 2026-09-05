"""schemas.py — trust-boundary validation (model-free) + default seed schema"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from extraction_api.schemas import CreateJobRequest, validate_job_request


def body(**kw):
    doc = {"field": "text", "content": "LoRA reduces hallucination."}
    return {"documents": [doc], **kw}


# --- รูปร่าง JSON (Pydantic) ---
def test_accepts_valid_body():
    req = CreateJobRequest.model_validate(body(callback_url="http://x/cb"))
    assert req.documents[0].field == "text"
    assert req.callback_url == "http://x/cb"
    assert req.seed_relations is None  # ไม่ส่ง = ใช้ default schema


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        CreateJobRequest.model_validate(body(documents=[{"field": "table", "content": "x"}]))


def test_rejects_empty_content():
    with pytest.raises(ValidationError):
        CreateJobRequest.model_validate(body(documents=[{"field": "text", "content": ""}]))


def test_rejects_non_string_content():
    with pytest.raises(ValidationError):
        CreateJobRequest.model_validate(body(documents=[{"field": "text", "content": 123}]))


# --- limits (จาก settings — inject ได้เพื่อ test) ---
def test_rejects_too_many_files():
    docs = [{"field": "text", "content": "x"}] * 3
    with pytest.raises(ValueError, match="MAX_FILES"):
        validate_job_request(
            CreateJobRequest.model_validate(body(documents=docs)), max_files=2, max_bytes=1000
        )


def test_rejects_too_many_bytes():
    docs = [{"field": "text", "content": "y" * 100}]
    with pytest.raises(ValueError, match="MAX_BYTES"):
        validate_job_request(CreateJobRequest.model_validate(body(documents=docs)), max_files=5, max_bytes=50)


def test_accepts_within_limits():
    validate_job_request(CreateJobRequest.model_validate(body()), max_files=20, max_bytes=1000)


def test_seed_relations_override_shape():
    req = CreateJobRequest.model_validate(body(seed_relations=["reduces", "improves"]))
    assert req.seed_relations == ["reduces", "improves"]


# --- default schema จาก bake-off/schema/seed.json ---
def test_default_seed_schema():
    from extraction_api.extractor import load_seed_schema

    schema = load_seed_schema()
    assert len(schema["relation_hints"]) == 10
    assert schema["entity_labels"]
