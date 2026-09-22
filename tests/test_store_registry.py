# 存储后端注册表与配置驱动切换的测试。

import pytest

import rag_data
from rag_data import (
    BaseVectorStore,
    InMemoryVectorStore,
    RagData,
    Settings,
    available_backends,
    build_store,
    create_store,
    is_registered,
    register_backend,
    register_store,
    resolve_store,
)
from rag_data.exceptions import ConfigError
from rag_data.storage.registry import BUILTIN_BACKENDS


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


@pytest.fixture(autouse=True)
def _restore_registry():
    """用例结束还原注册表，避免相互污染。"""
    from rag_data.storage import registry
    snapshot = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


# ---------- 内置后端 ----------


def test_builtin_backends_are_registered():
    assert set(BUILTIN_BACKENDS) <= set(available_backends())
    assert is_registered("memory") and is_registered("milvus")


def test_resolve_builtin_classes():
    assert resolve_store("memory") is InMemoryVectorStore
    assert resolve_store("milvus").__name__ == "MilvusVectorStore"


def test_resolved_classes_inherit_base():
    for name in available_backends():
        assert issubclass(resolve_store(name), BaseVectorStore)


def test_unknown_backend_raises_with_hint():
    with pytest.raises(ConfigError) as excinfo:
        resolve_store("nope")
    message = str(excinfo.value)
    assert "nope" in message and "memory" in message


# ---------- 继承即注册 ----------


def test_subclass_with_backend_is_auto_registered():
    @register_backend("custom_marker")
    class _Marked(InMemoryVectorStore):
        backend = "custom_marker"

    assert is_registered("custom_marker")
    assert resolve_store("custom_marker") is _Marked


def test_subclass_conflicting_name_overrides():
    class _Override(InMemoryVectorStore):
        backend = "memory"

    assert resolve_store("memory") is _Override


def test_subclass_without_backend_is_not_registered():
    before = set(available_backends())

    class _Plain(InMemoryVectorStore):
        pass

    assert set(available_backends()) == before


def test_register_store_accepts_dotted_path():
    register_store("external", "rag_data.storage.memory_store.InMemoryVectorStore")
    assert resolve_store("external") is InMemoryVectorStore


def test_register_store_accepts_class_object():
    register_store("byclass", InMemoryVectorStore)
    assert resolve_store("byclass") is InMemoryVectorStore


# ---------- 配置驱动 ----------


def test_config_backend_defaults_to_memory():
    assert Settings().storage.backend == "memory"


def test_config_backend_accepts_custom_name():
    assert Settings(storage={"backend": "custom"}).storage.backend == "custom"


def test_build_store_follows_config_backend():
    settings = Settings(storage={"backend": "memory"})
    assert isinstance(build_store(settings, _StubLogger()), InMemoryVectorStore)


def test_create_store_passes_keyword_arguments():
    settings = Settings()
    store = create_store("memory", settings, _StubLogger())
    assert isinstance(store, InMemoryVectorStore)
    assert store._settings is settings


def test_facade_uses_config_backend():
    settings = Settings(storage={"backend": "memory"})
    app = RagData(settings, logger=_StubLogger(), nlp=None)
    assert isinstance(app.store, InMemoryVectorStore)


def test_package_root_exports_registry_helpers():
    for name in ("register_store", "register_backend", "available_backends", "resolve_store", "create_store"):
        assert hasattr(rag_data, name)
