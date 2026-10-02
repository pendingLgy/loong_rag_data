# 数据模型单元测试：DocumentChunk 及其向量字段。

import pytest
from pydantic import ValidationError

from rag_data.models import DocumentChunk


def _chunk(vector=None):
    fields = dict(chunk_id="c1", user_id="u1", source_path="a.md", text="t", created_at=0.0)
    if vector is not None:
        fields["vector"] = vector
    return DocumentChunk(**fields)


def test_document_chunk_is_frozen():
    chunk = _chunk()
    with pytest.raises(ValidationError):
        chunk.text = "x"


def test_document_chunk_requires_fields():
    with pytest.raises(ValidationError):
        DocumentChunk(chunk_id="c1", user_id="u1", source_path="a.md")


def test_vector_defaults_to_empty():
    assert _chunk().vector == []


def test_vector_is_a_plain_field():
    assert _chunk(vector=[0.1, 0.2]).vector == [0.1, 0.2]


def test_vector_can_be_bound_with_model_copy():
    bound = _chunk().model_copy(update={"vector": [0.3, 0.4]})
    assert bound.vector == [0.3, 0.4]
    assert bound.chunk_id == "c1"
