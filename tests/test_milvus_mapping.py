# Milvus 实现的映射、过滤表达式与建表流程测试（用假 pymilvus，不依赖真实服务）。

import sys
import types

import pytest

from rag_data.config import Settings
from rag_data.exceptions import OptionalDependencyError
from rag_data.storage.milvus_store import (
    BASE_FIELD_NAMES,
    SCALAR_INDEX_TYPES,
    VECTOR_FIELD,
    MilvusRecord,
    MilvusVectorStore,
    import_pymilvus,
)


class _StubLogger:
    def bind(self, **fields):
        return self

    def debug(self, *a, **k):
        return None

    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None

    def exception(self, *a, **k):
        return None


def _record(**overrides):
    base = {
        "id": "m1",
        "text_payload": "文本",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "entities": ["E"],
        "created_at": 1.0,
    }
    base.update(overrides)
    return MilvusRecord(**base)


def _settings(**overrides):
    storage = {"vector_dim": 4, "collection_name": "test_coll"}
    storage.update(overrides.pop("storage", {}))
    return Settings(storage=storage, **overrides)


def _store(settings=None, **kwargs):
    return MilvusVectorStore(settings or _settings(), _StubLogger(), **kwargs)


# ---------- 记录与行的映射 ----------


def test_record_to_row_matches_schema():
    row = _store().record_to_row(_record(user_id="u1"))
    assert set(row) == set(MilvusRecord.storage_fields())


# ---------- 假 pymilvus，用于验证建表流程 ----------


class _FakeDataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    ARRAY = "ARRAY"
    DOUBLE = "DOUBLE"
    INT64 = "INT64"
    BOOL = "BOOL"
    JSON = "JSON"


class _FakeFieldSchema:
    """假 FieldSchema，构造时复现真实 Milvus 的必要参数校验。"""

    # 真实服务会拒绝缺少必要参数的 schema；若此处不校验，缺陷会一直漏到
    # 集成测试才暴露，故在构造期即按 Milvus 规则拦截。

    def __init__(self, name, dtype, description="", **kwargs):
        self.name = name
        self.dtype = dtype
        self.params = dict(kwargs)
        self._require_type_params()

    def _require_type_params(self):
        data_type = _FakeDataType
        if self.dtype == data_type.VARCHAR:
            self._require("max_length")
        elif self.dtype == data_type.FLOAT_VECTOR:
            self._require("dim")
        elif self.dtype == data_type.ARRAY:
            # ARRAY 需要元素个数上限；元素为 VARCHAR 时还需元素级最大长度。
            self._require("max_capacity")
            if self.params.get("element_type") == data_type.VARCHAR:
                self._require("max_length")

    def _require(self, param):
        if param in self.params:
            return
        raise ValueError(
            "type param(" + param + ") should be specified for the field("
            + self.name + "): missing parameter"
        )


class _FakeCollectionSchema:
    def __init__(self, fields, description=""):
        self.fields = fields
        self.description = description


class _FakeCollection:
    def __init__(self, name, schema=None, using=None):
        self.name = name
        self.schema = schema or _FakeCollectionSchema([])
        self.using = using
        self.indexes = []
        self.loaded = False
        self.upserted = []
        self.flush_count = 0

    def create_index(self, field_name, index_params):
        self.indexes.append((field_name, index_params))

    def load(self):
        self.loaded = True

    def upsert(self, columns):
        self.upserted.append(columns)

    def flush(self):
        self.flush_count += 1

    def search(self, **kwargs):
        return []


class _FakeUtility:
    def __init__(self, existing=None):
        self.existing = set(existing or [])

    def has_collection(self, name, using=None):
        return name in self.existing


class _FakeConnections:
    def __init__(self):
        self.connected = []
        self.disconnected = []

    def connect(self, alias=None, uri=None, db_name=None):
        self.connected.append((alias, uri, db_name))

    def disconnect(self, alias=None):
        self.disconnected.append(alias)


def _install_fake_pymilvus(monkeypatch, existing=None):
    module = types.ModuleType("pymilvus")
    module.DataType = _FakeDataType
    module.FieldSchema = _FakeFieldSchema
    module.CollectionSchema = _FakeCollectionSchema
    module.Collection = _FakeCollection
    module.utility = _FakeUtility(existing)
    module.connections = _FakeConnections()
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    return module


# ---------- 表结构（建表委托给记录类） ----------


def test_build_collection_schema_uses_milvus_record_fields(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    collection_schema = _store()._build_collection_schema(module)
    names = [field.name for field in collection_schema.fields]
    assert names == ["id", "text_payload", "vector", "entities", "created_at"]


def test_build_collection_schema_injects_vector_dim(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    collection_schema = _store()._build_collection_schema(module)
    by_name = {field.name: field for field in collection_schema.fields}
    assert by_name["vector"].params["dim"] == 4
    assert by_name["id"].params["is_primary"] is True
    assert by_name["created_at"].dtype == module.DataType.DOUBLE


def test_build_collection_schema_marks_array_element_type(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    collection_schema = _store()._build_collection_schema(module)
    by_name = {field.name: field for field in collection_schema.fields}
    assert by_name["entities"].dtype == module.DataType.ARRAY
    assert by_name["entities"].params["element_type"] == module.DataType.VARCHAR
    # ARRAY 的两项必要参数：元素个数上限与元素级最大长度
    assert by_name["entities"].params["max_capacity"] == 64
    assert by_name["entities"].params["max_length"] == 256


def test_column_names_before_create_are_base_fields():
    # 未建表时只有基类字段可知；扩展列需建表后从集合 schema 读取。
    assert _store().column_names() == list(BASE_FIELD_NAMES)


def test_vector_index_params_follow_settings():
    params = _store().vector_index_params()
    assert params["index_type"] == "HNSW"
    assert params["metric_type"] == "COSINE"
    assert params["params"]["M"] == 16


def test_scalar_index_specs_cover_entities_and_created_at():
    names = [name for name, _ in _store().scalar_index_specs()]
    assert names == list(SCALAR_INDEX_TYPES)


# ---------- 建表 ----------


def test_ensure_collection_creates_table_and_indexes(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    store = _store()
    store.ensure_collection()
    collection = store._collection
    assert collection is not None and collection.loaded
    indexed = [name for name, _ in collection.indexes]
    assert VECTOR_FIELD in indexed
    for name in list(SCALAR_INDEX_TYPES):
        assert name in indexed
    assert module.connections.connected

def test_connect_passes_db_name_from_settings(monkeypatch):
    """建连时带上配置的库名，集合读写才落在目标库。"""
    module = _install_fake_pymilvus(monkeypatch)
    store = _store(_settings(storage={"milvus_db": "rag_data_test"}))
    store.ensure_collection()
    _, uri, db_name = module.connections.connected[0]
    assert db_name == "rag_data_test"
    assert uri == store._settings.storage.milvus_uri


def test_created_field_schema_marks_primary_key(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    store = _store()
    store.ensure_collection()
    by_name = {field.name: field for field in store._collection.schema.fields}
    assert by_name["id"].params.get("is_primary") is True
    assert by_name["vector"].params.get("dim") == 4
    assert by_name["entities"].params.get("element_type") == module.DataType.VARCHAR


def test_ensure_collection_is_idempotent_when_exists(monkeypatch):
    _install_fake_pymilvus(monkeypatch, existing={"test_coll"})
    store = _store()
    store.ensure_collection()
    assert store._collection.loaded
    assert store._collection.indexes == []


def test_close_disconnects(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    store = _store()
    store.ensure_collection()
    store.close()
    assert module.connections.disconnected


def test_missing_pymilvus_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(OptionalDependencyError):
        import_pymilvus()
