# BaseVectorStore 接口契约测试，使用内存实现。
#
# 存储行编解码由记录类自身提供，这里同时覆盖
# 扩展字段能否真正落库与还原。

import json

import pytest

from rag_data.models import METADATA_FIELD, PRIMARY_FIELD, MemoryRecord
from rag_data.storage.base import BaseVectorStore
from rag_data.storage.memory_store import InMemoryVectorStore, cosine_similarity


def _record(record_id, vector, **extra):
    """构造记录；extra 中的键作为扩展字段一并存储。"""
    return MemoryRecord(
        id=record_id,
        text_payload=record_id,
        vector=vector,
        entities=[],
        created_at=1.0,
        **extra,
    )


def test_in_memory_implements_base_interface():
    assert isinstance(InMemoryVectorStore(), BaseVectorStore)


def test_upsert_and_count():
    store = InMemoryVectorStore()
    store.ensure_collection()
    assert store.upsert([_record("m1", [1.0, 0.0]), _record("m2", [0.0, 1.0])]) == 2
    assert store.count() == 2


def test_upsert_is_idempotent_by_id():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0])])
    store.upsert([_record("m1", [1.0, 0.0])])
    assert store.count() == 1


def test_query_orders_by_similarity():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0]), _record("m2", [0.0, 1.0])])
    hits = store.query([1.0, 0.0], top_n=2)
    assert [hit.id for hit in hits] == ["m1", "m2"]
    assert hits[0].score == pytest.approx(1.0)


def test_query_without_filters_returns_all():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0], user_id="u1"), _record("m2", [1.0, 0.0], user_id="u2")])
    assert len(store.query([1.0, 0.0], top_n=10)) == 2


def test_query_filters_by_extra_field():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0], user_id="u1"), _record("m2", [1.0, 0.0], user_id="u2")])
    hits = store.query([1.0, 0.0], top_n=10, filters={"user_id": "u2"})
    assert [hit.id for hit in hits] == ["m2"]


def test_query_filters_by_base_field():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0], user_id="u1")])
    hits = store.query([1.0, 0.0], top_n=10, filters={"text_payload": "m1"})
    assert [hit.id for hit in hits] == ["m1"]


def test_query_filters_unknown_field_matches_nothing():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0])])
    assert store.query([1.0, 0.0], top_n=10, filters={"unknown": 1}) == []


# ---------- 扩展字段落库 ----------


def test_row_layout_matches_schema():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0], user_id="u1")])
    row = store.get_row("m1")
    assert set(row) == set(MemoryRecord.storage_fields())
    assert row[PRIMARY_FIELD] == "m1"
    assert isinstance(row[METADATA_FIELD], str)


def test_extra_fields_are_persisted_and_restored():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0], user_id="u1", tags=["a"])])
    row = store.get_row("m1")
    assert json.loads(row[METADATA_FIELD]) == {"user_id": "u1", "tags": ["a"]}
    restored = store.get_record("m1")
    assert restored is not None
    assert restored.extra_fields == {"user_id": "u1", "tags": ["a"]}


def test_record_without_extras_has_empty_metadata():
    store = InMemoryVectorStore()
    store.upsert([_record("m1", [1.0, 0.0])])
    assert store.get_row("m1")[METADATA_FIELD] == "{}"


def test_get_record_missing_returns_none():
    assert InMemoryVectorStore().get_record("nope") is None


def test_subclass_extras_are_persisted():
    class TenantRecord(MemoryRecord):
        user_id: str

    store = InMemoryVectorStore()
    store.upsert([TenantRecord(id="m1", text_payload="t", vector=[1.0], created_at=1.0, user_id="u1")])
    assert store.get_record("m1").extra_fields == {"user_id": "u1"}


def test_cosine_similarity_bounds():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0
