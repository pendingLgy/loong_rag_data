# 存储 Schema 与存储行编解码的单元测试。

import json

import pytest

from rag_data.models import MemoryRecord
from rag_data.exceptions import ConfigError
from rag_data.storage import schema


def _record(**overrides):
    base = {
        "id": "m1",
        "text_payload": "文本",
        "vector": [0.1, 0.2],
        "entities": [],
        "created_at": 1.0,
    }
    base.update(overrides)
    return MemoryRecord(**base)


def test_field_names_complete():
    assert schema.field_names() == [
        "id",
        "text_payload",
        "vector",
        "entities",
        "created_at",
        "metadata",
    ]


def test_removed_fields_are_absent():
    names = schema.field_names()
    assert "memory_id" not in names
    assert "user_id" not in names
    assert "last_accessed_at" not in names
    assert "recall_count" not in names


def test_scalar_indexes_subset_of_fields():
    names = schema.field_names()
    for index_name in schema.SCALAR_INDEXES:
        assert index_name in names


def test_vector_field_and_defaults():
    assert schema.VECTOR_FIELD == "vector"
    assert schema.DEFAULT_METRIC == "COSINE"
    assert schema.DEFAULT_INDEX_TYPE == "HNSW"


def test_primary_key_is_id():
    assert schema.PRIMARY_FIELD == "id"
    assert "id" in schema.BASE_COLUMNS
    assert "id" not in schema.SCALAR_INDEXES


def test_base_columns_exclude_metadata():
    assert schema.METADATA_FIELD == "metadata"
    assert schema.METADATA_FIELD not in schema.BASE_COLUMNS
    assert schema.BASE_COLUMNS == ["id", "text_payload", "vector", "entities", "created_at"]


def test_is_base_field():
    assert schema.is_base_field("id")
    assert schema.is_base_field("vector")
    assert not schema.is_base_field("metadata")
    assert not schema.is_base_field("user_id")


def test_metadata_fields_extracts_extras_only():
    assert schema.metadata_fields(_record()) == {}
    extras = schema.metadata_fields(_record(user_id="u1", tags=["a"]))
    assert extras == {"user_id": "u1", "tags": ["a"]}


# ---------- 编解码 ----------


def test_to_storage_row_layout():
    row = schema.to_storage_row(_record(user_id="u1"))
    assert set(row) == set(schema.field_names())
    assert row[schema.PRIMARY_FIELD] == "m1"
    assert row[schema.VECTOR_FIELD] == [0.1, 0.2]
    assert isinstance(row[schema.METADATA_FIELD], str)


def test_to_storage_row_serializes_extras_into_metadata():
    row = schema.to_storage_row(_record(user_id="u1", tags=["a"]))
    assert json.loads(row[schema.METADATA_FIELD]) == {"user_id": "u1", "tags": ["a"]}


def test_to_storage_row_without_extras_uses_empty_object():
    row = schema.to_storage_row(_record())
    assert row[schema.METADATA_FIELD] == "{}"


def test_roundtrip_preserves_base_and_extra_fields():
    original = _record(user_id="u1", tags=["a"], weight=0.5)
    restored = schema.from_storage_row(schema.to_storage_row(original))
    assert restored.id == original.id
    assert restored.text_payload == original.text_payload
    assert restored.vector == original.vector
    assert restored.created_at == original.created_at
    assert restored.extra_fields == {"user_id": "u1", "tags": ["a"], "weight": 0.5}


def test_decode_metadata_accepts_multiple_shapes():
    assert schema.decode_metadata(None) == {}
    assert schema.decode_metadata("") == {}
    assert schema.decode_metadata({"a": 1}) == {"a": 1}
    assert schema.decode_metadata(json.dumps({"a": 1})) == {"a": 1}


def test_encode_metadata_is_deterministic():
    assert schema.encode_metadata({"b": 1, "a": 2}) == schema.encode_metadata({"a": 2, "b": 1})


def test_flatten_row_merges_extras():
    row = schema.to_storage_row(_record(user_id="u1"))
    flat = schema.flatten_row(row)
    assert flat["user_id"] == "u1"
    assert flat[schema.PRIMARY_FIELD] == "m1"


def test_row_matches_base_and_extra():
    row = schema.to_storage_row(_record(user_id="u1"))
    assert schema.row_matches(row, {})
    assert schema.row_matches(row, {"text_payload": "文本"})
    assert schema.row_matches(row, {"user_id": "u1"})
    assert not schema.row_matches(row, {"user_id": "u2"})
    assert not schema.row_matches(row, {"missing": 1})


# ---------- 提升列 ----------


def test_promoted_names_helper():
    assert schema.promoted_names([{"name": "user_id"}]) == ["user_id"]
    assert schema.promoted_names(None) == []


def test_promoted_field_is_physical_column_not_in_metadata():
    row = schema.to_storage_row(_record(user_id="u1", tags=["a"]), promoted=["user_id"])
    assert row["user_id"] == "u1"
    assert schema.decode_metadata(row[schema.METADATA_FIELD]) == {"tags": ["a"]}


def test_promoted_field_roundtrip_keeps_all_extras():
    record = _record(user_id="u1", tags=["a"])
    promoted = ["user_id"]
    row = schema.to_storage_row(record, promoted)
    restored = schema.from_storage_row(row, None, promoted)
    assert restored.extra_fields == {"user_id": "u1", "tags": ["a"]}


def test_flatten_row_includes_promoted_column():
    row = schema.to_storage_row(_record(user_id="u1"), promoted=["user_id"])
    assert schema.flatten_row(row, ["user_id"])["user_id"] == "u1"


def test_row_matches_with_promoted_column():
    row = schema.to_storage_row(_record(user_id="u1"), promoted=["user_id"])
    assert schema.row_matches(row, {"user_id": "u1"}, ["user_id"])
    assert not schema.row_matches(row, {"user_id": "u2"}, ["user_id"])


# ---------- 建表字段生成 ----------


def test_build_scalar_index_specs():
    specs = schema.build_scalar_index_specs(["id", "entities", "created_at"])
    assert [spec["field_name"] for spec in specs] == ["entities", "created_at"]


def test_build_vector_index_params_hnsw():
    params = schema.build_vector_index_params("HNSW", "COSINE")
    assert params["index_type"] == "HNSW"
    assert params["params"]["M"] == 16


def test_build_vector_index_params_ivflat():
    params = schema.build_vector_index_params("IVFLAT", "L2")
    assert params["params"]["nlist"] == 1024


def test_build_vector_index_params_rejects_unknown():
    with pytest.raises(ConfigError):
        schema.build_vector_index_params("IVF_PQ", "COSINE")

