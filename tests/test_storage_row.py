# 存储行编解码测试：编解码由 MilvusRecord 提供。

from rag_data.storage.milvus_store import BASE_FIELD_NAMES, VECTOR_FIELD, MilvusRecord


def _record(**overrides):
    base = {"id": "m1", "text_payload": "文本", "vector": [0.1, 0.2], "entities": [], "created_at": 1.0}
    base.update(overrides)
    return MilvusRecord(**base)


def test_storage_fields_are_base_fields():
    assert MilvusRecord.storage_fields() == list(BASE_FIELD_NAMES)


def test_to_storage_row_flattens_base_fields():
    row = _record().to_storage_row()
    assert set(row) == set(BASE_FIELD_NAMES)
    assert row[VECTOR_FIELD] == [0.1, 0.2]


def test_undeclared_fields_are_not_stored():
    """未声明的扩展字段不落库，存储行只含平铺列。"""
    row = _record(user_id="u1", tags=["a"]).to_storage_row()
    assert set(row) == set(BASE_FIELD_NAMES)


def test_roundtrip_preserves_base_fields():
    original = _record()
    restored = MilvusRecord.from_storage_row(original.to_storage_row())
    assert restored.id == original.id
    assert restored.text_payload == original.text_payload
    assert restored.vector == original.vector
    assert restored.created_at == original.created_at
