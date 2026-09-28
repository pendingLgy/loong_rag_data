# 存储行编解码测试：编解码由记录类自身提供（MemoryRecord 定义，MilvusRecord 继承）。

import json

from rag_data.models import (
    BASE_FIELD_NAMES,
    METADATA_FIELD,
    VECTOR_FIELD,
    MemoryRecord,
    MilvusRecord,
    decode_metadata,
    encode_metadata,
)


def _record(**overrides):
    base = {"id": "m1", "text_payload": "文本", "vector": [0.1, 0.2], "entities": [], "created_at": 1.0}
    base.update(overrides)
    return MemoryRecord(**base)


def test_storage_fields_include_metadata():
    assert MemoryRecord.storage_fields() == list(BASE_FIELD_NAMES) + [METADATA_FIELD]


def test_storage_columns_dedups_promoted():
    assert MemoryRecord.storage_columns() == list(BASE_FIELD_NAMES)
    assert MemoryRecord.storage_columns(["user_id", "id"]) == list(BASE_FIELD_NAMES) + ["user_id"]


def test_to_storage_row_flattens_base_and_serializes_extras():
    row = _record(user_id="u1", tags=["a"]).to_storage_row()
    assert set(row) == set(MemoryRecord.storage_fields())
    assert row[VECTOR_FIELD] == [0.1, 0.2]
    assert decode_metadata(row[METADATA_FIELD]) == {"user_id": "u1", "tags": ["a"]}


def test_to_storage_row_without_extras_uses_empty_object():
    assert _record().to_storage_row()[METADATA_FIELD] == "{}"


def test_roundtrip_preserves_base_and_extra_fields():
    original = _record(user_id="u1", tags=["a"], weight=0.5)
    restored = MemoryRecord.from_storage_row(original.to_storage_row())
    assert restored.id == original.id
    assert restored.vector == original.vector
    assert restored.extra_fields == {"user_id": "u1", "tags": ["a"], "weight": 0.5}


def test_decode_and_encode_metadata():
    assert decode_metadata(None) == {}
    assert decode_metadata("") == {}
    assert decode_metadata({"a": 1}) == {"a": 1}
    assert decode_metadata(json.dumps({"a": 1})) == {"a": 1}
    assert encode_metadata({"b": 1, "a": 2}) == encode_metadata({"a": 2, "b": 1})


def test_promoted_column_is_not_stored_in_metadata():
    row = _record(user_id="u1", tags=["a"]).to_storage_row(["user_id"])
    assert row["user_id"] == "u1"
    assert decode_metadata(row[METADATA_FIELD]) == {"tags": ["a"]}


def test_promoted_column_roundtrip():
    record = _record(user_id="u1", tags=["a"])
    row = record.to_storage_row(["user_id"])
    assert MemoryRecord.from_storage_row(row, ["user_id"]).extra_fields == {"user_id": "u1", "tags": ["a"]}


def test_flatten_row_and_row_matches():
    row = _record(user_id="u1").to_storage_row()
    assert MemoryRecord.flatten_row(row)["user_id"] == "u1"
    assert MemoryRecord.row_matches(row, {"user_id": "u1"})
    assert not MemoryRecord.row_matches(row, {"user_id": "u2"})
    assert MemoryRecord.row_matches(row, {})
    assert not MemoryRecord.row_matches(row, {"missing": 1})


def test_milvus_record_inherits_storage_codec():
    record = MilvusRecord(id="m1", text_payload="t", vector=[0.1], created_at=1.0, user_id="u1")
    assert MilvusRecord.from_storage_row(record.to_storage_row()).extra_fields == {"user_id": "u1"}
