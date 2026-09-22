# pydantic 数据模型的单元测试。

from typing import List

import pytest
from pydantic import ValidationError

from rag_data.models import BASE_FIELD_NAMES, DocumentChunk, MemoryRecord, QueryHit


def _record(**overrides):
    base = {
        "id": "m1",
        "text_payload": "文本",
        "vector": [0.1, 0.2],
        "entities": ["A"],
        "created_at": 1.0,
    }
    base.update(overrides)
    return MemoryRecord(**base)


def test_memory_record_base_fields():
    record = _record()
    assert record.id == "m1"
    assert record.extra_fields == {}
    assert set(record.to_row()) == set(BASE_FIELD_NAMES)


def test_memory_record_dropped_legacy_fields():
    fields = set(MemoryRecord.model_fields)
    assert "memory_id" not in fields
    assert "user_id" not in fields


def test_memory_record_allows_extra_fields():
    record = _record(user_id="u1", tags=["x"])
    assert record.user_id == "u1"
    assert record.extra_fields == {"user_id": "u1", "tags": ["x"]}
    assert record.to_row()["user_id"] == "u1"


def test_memory_record_subclass_adds_typed_field():
    class TenantRecord(MemoryRecord):
        user_id: str
        tags: List[str] = []

    record = TenantRecord(
        id="m1", text_payload="t", vector=[0.1], created_at=1.0, user_id="u1", tags=["a"]
    )
    assert record.extra_fields == {"user_id": "u1", "tags": ["a"]}
    assert set(record.to_row()) >= set(BASE_FIELD_NAMES)


def test_memory_record_subclass_validates_its_field():
    class TenantRecord(MemoryRecord):
        user_id: str

    with pytest.raises(ValidationError):
        TenantRecord(id="m1", text_payload="t", vector=[0.1], created_at=1.0)


def test_memory_record_rejects_empty_vector():
    with pytest.raises(ValidationError):
        _record(vector=[])


def test_memory_record_rejects_empty_text():
    with pytest.raises(ValidationError):
        _record(text_payload="")


def test_memory_record_rejects_empty_id():
    with pytest.raises(ValidationError):
        _record(id="")


def test_base_vector_still_validated_on_subclass():
    class TenantRecord(MemoryRecord):
        user_id: str

    with pytest.raises(ValidationError):
        TenantRecord(id="m1", text_payload="t", vector=[], created_at=1.0, user_id="u1")


def test_query_hit_uses_id():
    hit = QueryHit(id="m1", text_payload="t", score=0.5, created_at=1.0)
    assert hit.id == "m1"
    assert "memory_id" not in set(QueryHit.model_fields)


def test_query_hit_score_must_be_in_range():
    with pytest.raises(ValidationError):
        QueryHit(id="m1", text_payload="t", score=1.5, created_at=1.0)


def test_document_chunk_is_frozen():
    chunk = DocumentChunk(chunk_id="c1", user_id="u1", source_path="a.md", text="t", created_at=0.0)
    with pytest.raises(ValidationError):
        chunk.text = "x"
