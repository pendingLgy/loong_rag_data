# 记录类路径解析与自定义继承生效的测试。

import json
import os
import sys
import textwrap

import pytest

from rag_data import (
    DEFAULT_RECORD_CLASS_PATH,
    InMemoryVectorStore,
    MemoryRecord,
    RagData,
    Settings,
    build_record_class,
    build_store,
    resolve_record_class,
)
from rag_data.config import ModelSettings
from rag_data.exceptions import ConfigError
from rag_data.ingestion.pipeline import IngestionPipeline
from rag_data.storage import schema

CUSTOM_TEMPLATE = textwrap.dedent(chr(10).join([
    "from typing import List",
    "",
    "from rag_data import MemoryRecord",
    "",
    "",
    "class TenantRecord(MemoryRecord):",
    "    user_id: str",
    "    tags: List[str] = []",
])).strip() + chr(10)


def _write_module(tmp_path, name, body):
    """在临时目录写入模块并加入 sys.path，返回模块名。"""
    directory = os.path.join(str(tmp_path), name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + ".py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    sys.path.insert(0, directory)
    return name


# ---------- 配置 ----------


def test_model_settings_default():
    assert ModelSettings().record_class == DEFAULT_RECORD_CLASS_PATH


def test_settings_has_models_section():
    assert Settings().models.record_class == DEFAULT_RECORD_CLASS_PATH


def test_settings_accepts_models_section():
    config = Settings(models={"record_class": "demo.Record"})
    assert config.models.record_class == "demo.Record"


def test_model_settings_rejects_unknown_field():
    with pytest.raises(Exception):
        ModelSettings(unknown=1)


# ---------- 路径解析 ----------


def test_resolve_default_class():
    assert resolve_record_class(DEFAULT_RECORD_CLASS_PATH) is MemoryRecord


def test_resolve_custom_subclass(tmp_path):
    module = _write_module(tmp_path, "myrecords_a", CUSTOM_TEMPLATE)
    cls = resolve_record_class(module + ".TenantRecord")
    assert issubclass(cls, MemoryRecord)
    assert cls is not MemoryRecord


def test_resolve_rejects_non_dotted_path():
    with pytest.raises(ConfigError):
        resolve_record_class("MemoryRecord")


def test_resolve_rejects_empty_path():
    with pytest.raises(ConfigError):
        resolve_record_class("")


def test_resolve_rejects_missing_module():
    with pytest.raises(ConfigError):
        resolve_record_class("no_such_module_xyz.Record")


def test_resolve_rejects_missing_attribute():
    with pytest.raises(ConfigError):
        resolve_record_class("rag_data.models.NotExist")


def test_resolve_rejects_non_record_class():
    with pytest.raises(ConfigError):
        resolve_record_class("rag_data.models.QueryHit")


# ---------- 配置驱动的全链路 ----------


def test_build_record_class_from_settings(tmp_path):
    module = _write_module(tmp_path, "myrecords_b", CUSTOM_TEMPLATE)
    settings = Settings(models={"record_class": module + ".TenantRecord"})
    cls = build_record_class(settings)
    assert cls.__name__ == "TenantRecord"


def test_store_uses_custom_class(tmp_path):
    module = _write_module(tmp_path, "myrecords_c", CUSTOM_TEMPLATE)
    settings = Settings(models={"record_class": module + ".TenantRecord"})
    store = build_store(settings, None)
    record = MemoryRecord(
        id="m1", text_payload="t", vector=[1.0, 0.0], created_at=1.0, user_id="u1"
    )
    store.upsert([record])
    restored = store.get_record("m1")
    assert isinstance(restored, build_record_class(settings))
    assert restored.user_id == "u1"


def test_pipeline_writes_custom_class(tmp_path, logger):
    module = _write_module(tmp_path, "myrecords_d", CUSTOM_TEMPLATE)
    settings = Settings(models={"record_class": module + ".TenantRecord"})
    cls = build_record_class(settings)

    class _FakeEmbedder:
        def encode(self, texts):
            return [[0.1] * settings.storage.vector_dim for _ in texts]

    store = InMemoryVectorStore(record_class=cls)
    pipeline = IngestionPipeline(store, _FakeEmbedder(), settings, logger, record_class=cls)
    path = os.path.join(str(tmp_path), "doc.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("第一句。第二句。")
    written = pipeline.run([path], user_id="tenant-a")
    assert written > 0
    row = next(iter(store._rows.values()))
    assert json.loads(row[schema.METADATA_FIELD])["user_id"] == "tenant-a"


def test_facade_exposes_record_class(tmp_path, settings=None):
    module = _write_module(tmp_path, "myrecords_e", CUSTOM_TEMPLATE)
    config = Settings(models={"record_class": module + ".TenantRecord"})
    app = RagData(config, logger=_NullLogger(), nlp=None, embedder=_NullEmbedder(config))
    assert app.record_class.__name__ == "TenantRecord"


def test_custom_class_field_is_validated(tmp_path):
    module = _write_module(tmp_path, "myrecords_f", CUSTOM_TEMPLATE)
    cls = resolve_record_class(module + ".TenantRecord")
    with pytest.raises(Exception):
        cls(id="m1", text_payload="t", vector=[0.1], created_at=1.0)


class _NullLogger:
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


class _NullEmbedder:
    def __init__(self, settings):
        self._d = settings.storage.vector_dim

    def encode(self, texts):
        return [[0.0] * self._d for _ in texts]
