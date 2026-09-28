# Milvus 实现的映射、过滤表达式与建表流程测试（用假 pymilvus，不依赖真实服务）。

import json
import sys
import types

import pytest

from rag_data.config import Settings
from rag_data.exceptions import OptionalDependencyError, SchemaMismatchError
from rag_data.models import BASE_FIELD_NAMES, METADATA_FIELD, VECTOR_FIELD, MilvusRecord, decode_metadata
from rag_data.storage.milvus_store import (
    METADATA_PATH,
    SCALAR_INDEXES,
    VECTOR_INDEX_PARAMS,
    MilvusVectorStore,
    ScalarIndexSpec,
    build_filter_expr,
    build_scalar_index_specs,
    build_vector_index_params,
    format_literal,
    import_pymilvus,
    register_vector_index_params,
)

QUOTE = chr(34)
LBRACKET = chr(91)
RBRACKET = chr(93)


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


# ---------- 字面量 ----------


def test_format_literal_scalars():
    assert format_literal(3) == "3"
    assert format_literal(3.5) == "3.5"
    assert format_literal(True) == "true"
    assert format_literal(False) == "false"


def test_format_literal_string_is_quoted():
    assert format_literal("abc") == json.dumps("abc")


def test_format_literal_list():
    expected = LBRACKET + json.dumps("a") + RBRACKET
    assert format_literal(["a"]) == expected


def test_format_literal_rejects_unsupported_type():
    with pytest.raises(TypeError):
        format_literal(object())


# ---------- 过滤表达式 ----------


def test_build_filter_expr_empty():
    assert build_filter_expr(None) == ""
    assert build_filter_expr({}) == ""


def test_build_filter_expr_base_field():
    expected = "text_payload == " + json.dumps("t")
    assert build_filter_expr({"text_payload": "t"}) == expected


def test_build_filter_expr_extra_field_uses_metadata_path():
    expected = METADATA_PATH.format("user_id") + " == " + json.dumps("u1")
    assert build_filter_expr({"user_id": "u1"}) == expected


def test_build_filter_expr_promoted_column_uses_plain_name():
    expected = "user_id == " + json.dumps("u1")
    expr = build_filter_expr({"user_id": "u1"}, column_names=["id", "user_id"])
    assert expr == expected


def test_build_filter_expr_combines_clauses():
    expr = build_filter_expr({"text_payload": "t", "user_id": "u1"})
    assert " and " in expr
    assert expr.count("==") == 2


# ---------- 记录与行的映射 ----------


def test_record_to_row_matches_schema():
    row = _store().record_to_row(_record(user_id="u1"))
    assert set(row) == set(MilvusRecord.storage_fields())


def test_record_to_row_puts_extras_in_metadata():
    row = _store().record_to_row(_record(user_id="u1"))
    assert decode_metadata(row[METADATA_FIELD]) == {"user_id": "u1"}


def test_row_to_record_restores_extras():
    row = _store().record_to_row(_record(user_id="u1"))
    assert _store().row_to_record(row).extra_fields == {"user_id": "u1"}


def test_row_to_hit_maps_base_columns():
    row = _store().record_to_row(_record())
    hit = _store().row_to_hit(row, 0.75)
    assert hit.id == "m1" and hit.score == 0.75


def test_normalize_score_cosine():
    assert _store().normalize_score(1.0, "COSINE") == pytest.approx(1.0)
    assert _store().normalize_score(-1.0, "COSINE") == pytest.approx(0.0)


def test_normalize_score_l2_is_bounded():
    value = _store().normalize_score(0.0, "L2")
    assert 0.0 <= value <= 1.0


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


class _ExtendedRecord(MilvusRecord):
    """Milvus 记录子类通过覆盖 build_collection_schema 扩展表结构。"""

    user_id: str

    @classmethod
    def build_collection_schema(cls, pymilvus, vector_dim):
        collection_schema = super().build_collection_schema(pymilvus, vector_dim)
        collection_schema.fields.append(
            pymilvus.FieldSchema(name="user_id", dtype=pymilvus.DataType.VARCHAR, max_length=128)
        )
        return collection_schema


def test_build_collection_schema_uses_milvus_record_fields(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    collection_schema = _store()._build_collection_schema(module)
    names = [field.name for field in collection_schema.fields]
    assert names == ["id", "text_payload", "vector", "entities", "created_at", "metadata"]


def test_build_collection_schema_injects_vector_dim(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    collection_schema = _store()._build_collection_schema(module)
    by_name = {field.name: field for field in collection_schema.fields}
    assert by_name["vector"].params["dim"] == 4
    assert by_name["id"].params["is_primary"] is True
    assert by_name["metadata"].dtype == module.DataType.JSON


def test_build_collection_schema_marks_array_element_type(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    collection_schema = _store()._build_collection_schema(module)
    by_name = {field.name: field for field in collection_schema.fields}
    assert by_name["entities"].dtype == module.DataType.ARRAY
    assert by_name["entities"].params["element_type"] == module.DataType.VARCHAR
    # ARRAY 的两项必要参数：元素个数上限与元素级最大长度
    assert by_name["entities"].params["max_capacity"] == 64
    assert by_name["entities"].params["max_length"] == 256


def test_subclass_override_extends_collection_schema(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    store = _store(record_class=_ExtendedRecord)
    collection_schema = store._build_collection_schema(module)
    names = [field.name for field in collection_schema.fields]
    assert names == ["id", "text_payload", "vector", "entities", "created_at", "metadata", "user_id"]


def test_column_names_exclude_metadata_before_create():
    # 未建表时只有基类字段可知，metadata 不计入；扩展列需建表后从集合 schema 读取。
    assert _store().column_names() == list(BASE_FIELD_NAMES)


def test_vector_index_params_follow_settings():
    params = _store().vector_index_params()
    assert params["index_type"] == "HNSW"
    assert params["metric_type"] == "COSINE"
    assert params["params"]["M"] == 16


def test_scalar_index_specs_cover_entities_and_created_at():
    names = [spec.field_name for spec in _store().scalar_index_specs()]
    assert names == SCALAR_INDEXES


# ---------- 建表 ----------


def test_ensure_collection_creates_table_and_indexes(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch)
    store = _store()
    store.ensure_collection()
    collection = store._collection
    assert collection is not None and collection.loaded
    indexed = [name for name, _ in collection.indexes]
    assert VECTOR_FIELD in indexed
    for name in SCALAR_INDEXES:
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


def test_ensure_collection_rejects_dim_mismatch(monkeypatch):
    module = _install_fake_pymilvus(monkeypatch, existing={"test_coll"})
    mismatch = _FakeCollection("test_coll", _FakeCollectionSchema([_FakeFieldSchema("vector", _FakeDataType.FLOAT_VECTOR, dim=1024)]))
    module.Collection = lambda name, schema=None, using=None: mismatch
    with pytest.raises(SchemaMismatchError):
        _store().ensure_collection()


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


# ---------- 子类覆盖建表后，列名与写入以实际集合为准 ----------


def test_column_names_follow_actual_collection_after_create(monkeypatch):
    _install_fake_pymilvus(monkeypatch)
    store = _store(record_class=_ExtendedRecord)
    store.ensure_collection()
    assert "user_id" in store.column_names()
    assert store.promoted_columns() == ["user_id"]


def test_upsert_flattens_column_added_by_subclass(monkeypatch):
    _install_fake_pymilvus(monkeypatch)
    store = _store(record_class=_ExtendedRecord)
    store.ensure_collection()
    record = _ExtendedRecord(
        id="m1", text_payload="t", vector=[0.1, 0.2, 0.3, 0.4], created_at=1.0, user_id="u1"
    )
    store.upsert([record])
    columns = store._collection.upserted[0]
    assert columns["user_id"][0] == "u1"
    assert columns[METADATA_FIELD][0] == {}


# ---------- 索引参数（Milvus 专有，随实现存放） ----------


def test_build_scalar_index_specs_default():
    specs = build_scalar_index_specs(["id", "entities", "created_at"])
    assert [spec.field_name for spec in specs] == ["entities", "created_at"]
    assert [spec.index_params for spec in specs] == [
        {"index_type": "INVERTED"},
        {"index_type": "STL_SORT"},
    ]


def test_scalar_index_spec_is_named_tuple():
    spec = ScalarIndexSpec("entities", {"index_type": "INVERTED"})
    name, params = spec
    assert (name, params) == ("entities", {"index_type": "INVERTED"})
    assert isinstance(spec, tuple)
    with pytest.raises(AttributeError):
        spec.field_name = "other"


def test_build_vector_index_params_defaults_and_returns_copy():
    params = build_vector_index_params("HNSW", "COSINE")
    assert params["index_type"] == "HNSW"
    assert params["params"]["M"] == 16
    params["params"]["M"] = 0
    assert VECTOR_INDEX_PARAMS["HNSW"]["M"] == 16


def test_build_vector_index_params_accepts_unknown_type():
    assert build_vector_index_params("IVF_PQ", "COSINE")["params"] == {}


def test_build_vector_index_params_accepts_explicit_params():
    assert build_vector_index_params("HNSW", "L2", {"M": 32})["params"] == {"M": 32}


def test_register_vector_index_params_extends_table():
    original = dict(VECTOR_INDEX_PARAMS)
    try:
        register_vector_index_params("SCANN", {"nlist": 128})
        assert build_vector_index_params("SCANN", "COSINE")["params"] == {"nlist": 128}
    finally:
        VECTOR_INDEX_PARAMS.clear()
        VECTOR_INDEX_PARAMS.update(original)


def test_build_scalar_index_specs_accepts_custom_types():
    specs = build_scalar_index_specs(["id", "user_id"], {"user_id": "INVERTED"})
    assert [spec.field_name for spec in specs] == ["user_id"]
