# 数据模型单元测试：DocumentChunk 与字段常量。

import pytest
from pydantic import ValidationError

from rag_data.models import DocumentChunk
from rag_data.storage.milvus_store import BASE_FIELD_NAMES, VECTOR_FIELD


def test_document_chunk_is_frozen():
    chunk = DocumentChunk(chunk_id="c1", user_id="u1", source_path="a.md", text="t", created_at=0.0)
    with pytest.raises(ValidationError):
        chunk.text = "x"


def test_document_chunk_requires_fields():
    with pytest.raises(ValidationError):
        DocumentChunk(chunk_id="c1", user_id="u1", source_path="a.md")


def test_base_field_names_cover_record_columns():
    assert list(BASE_FIELD_NAMES) == ["id", "text_payload", "vector", "entities", "created_at"]
    assert VECTOR_FIELD in BASE_FIELD_NAMES
