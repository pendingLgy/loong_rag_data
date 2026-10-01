# 存储后端注册表与配置驱动切换的测试。

import os

import pytest

import rag_data
from fake_store import FakeStore
from rag_data import (
    RagData,
    Settings,
    available_backends,
    build_store,
    create_store,
    is_registered,
    load_store_modules,
    register_backend,
    register_store,
    resolve_store,
)
from rag_data.exceptions import ConfigError
from rag_data.storage.milvus_store import MilvusVectorStore
from rag_data.store_registry import BUILTIN_BACKENDS


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
    from rag_data import store_registry
    snapshot = dict(store_registry._REGISTRY)
    yield
    store_registry._REGISTRY.clear()
    store_registry._REGISTRY.update(snapshot)


# ---------- 内置后端 ----------


def test_builtin_backends_are_registered():
    assert set(BUILTIN_BACKENDS) <= set(available_backends())
    assert is_registered("milvus")


def test_resolve_builtin_classes():
    assert resolve_store("milvus") is MilvusVectorStore


def test_unknown_backend_raises_with_hint():
    with pytest.raises(ConfigError) as excinfo:
        resolve_store("nope")
    message = str(excinfo.value)
    assert "nope" in message and "milvus" in message


# ---------- 显式登记 ----------


def test_register_backend_decorator_registers_class():
    @register_backend("custom_marker")
    class _Marked(FakeStore):
        backend = "custom_marker"

    assert is_registered("custom_marker")
    assert resolve_store("custom_marker") is _Marked


def test_unregistered_class_is_not_auto_registered():
    before = set(available_backends())

    class _Plain(FakeStore):
        backend = "plain"

    assert set(available_backends()) == before


def test_register_store_accepts_dotted_path():
    register_store("external", "fake_store.FakeStore")
    assert resolve_store("external") is FakeStore


def test_register_store_accepts_class_object():
    register_store("byclass", FakeStore)
    assert resolve_store("byclass") is FakeStore


# ---------- 配置驱动 ----------


def test_config_backend_defaults_to_milvus():
    assert Settings().storage.backend == "milvus"


def test_config_backend_accepts_custom_name():
    assert Settings(storage={"backend": "custom"}).storage.backend == "custom"


def test_build_store_follows_config_backend():
    register_store("fake", FakeStore)
    settings = Settings(storage={"backend": "fake"})
    assert isinstance(build_store(settings, _StubLogger()), FakeStore)


def test_create_store_passes_keyword_arguments():
    register_store("fake", FakeStore)
    settings = Settings()
    store = create_store("fake", settings, _StubLogger())
    assert isinstance(store, FakeStore)
    assert store._settings is settings


def test_facade_uses_config_backend():
    register_store("fake", FakeStore)
    settings = Settings(storage={"backend": "fake"})
    app = RagData(settings, logger=_StubLogger(), nlp=None)
    assert isinstance(app.store, FakeStore)


def test_package_root_exports_registry_helpers():
    for name in ("register_store", "register_backend", "available_backends", "resolve_store", "create_store", "load_store_modules"):
        assert hasattr(rag_data, name)


# ---------- 用户模块（插件） ----------


def _write_py(directory, name, lines):
    """把若干行写入目录下的 py 文件，返回文件路径。"""
    path = os.path.join(str(directory), name + ".py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(chr(10).join(lines))
    return path


def test_store_modules_file_path_is_imported(tmp_path):
    path = _write_py(tmp_path, "my_stores", [
        'from fake_store import FakeStore',
        'from rag_data.store_registry import register_store',
        '',
        'register_store("plugged", FakeStore)',
    ])
    settings = Settings(storage={"backend": "plugged", "store_modules": [path]})
    store = build_store(settings, _StubLogger())
    assert isinstance(store, FakeStore)
    assert store._settings is settings
    assert resolve_store("plugged") is FakeStore


def test_store_modules_dotted_path_is_imported(tmp_path, monkeypatch):
    _write_py(tmp_path, "my_plugins", [
        'from fake_store import FakeStore',
        'from rag_data.store_registry import register_store',
        '',
        'register_store("dotted", FakeStore)',
    ])
    monkeypatch.syspath_prepend(str(tmp_path))
    settings = Settings(storage={"backend": "dotted", "store_modules": ["my_plugins"]})
    assert isinstance(build_store(settings, _StubLogger()), FakeStore)


def test_store_modules_missing_file_raises():
    with pytest.raises(ConfigError):
        load_store_modules(["no_such_stores.py"])


def test_store_module_is_executed_once(tmp_path):
    path = _write_py(tmp_path, "once_stores", [
        'from rag_data.store_registry import register_store',
        '',
        'class OnceStore:',
        '    def __init__(self, settings=None, logger=None):',
        '        pass',
        '',
        'register_store("once", OnceStore)',
    ])
    load_store_modules([path])
    first = resolve_store("once")
    load_store_modules([path])
    assert resolve_store("once") is first
