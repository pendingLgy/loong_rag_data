# 门面单元测试：MemoryServer 的四类组件装配、流程代理与启用门控，以及构造签名契约。
#
# 用假向量模型与假解析器隔离网络与 spaCy；默认装配分支由 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import MemorySettings
from rag_data.exceptions import ConfigError

# ---------------- MemoryServer ----------------


class _FakeMemoryHandler:
    # 受控记忆处理器替身：记录四条流程的入参并返回固定结果。
    def __init__(self):
        self.calls = []

    def add(self, messages, **kwargs):
        self.calls.append(('add', messages, kwargs))
        return 'ADD'

    def search(self, query, **kwargs):
        self.calls.append(('search', query, kwargs))
        return 'SEARCH'

    def update(self, memory_id, **kwargs):
        self.calls.append(('update', memory_id, kwargs))
        return 'UPDATE'

    def delete(self, memory_id):
        self.calls.append(('delete', memory_id, {}))
        return 'DELETE'


def _install_memory_fakes(monkeypatch):
    # 打桩 MemoryServer 的四类组件装配，避免建立真实数据库与向量库连接。
    monkeypatch.setattr(facade, '_build_embedder', lambda settings: 'EMB')
    monkeypatch.setattr(facade, '_build_llm', lambda settings: 'LLM')
    monkeypatch.setattr(facade, '_build_db', lambda settings: 'DB')
    monkeypatch.setattr(facade, '_build_vector', lambda settings: 'VEC')


@pytest.fixture()
def memory_settings():
    # 记忆分区独立于 RagSetting：直接构造 MemorySettings，开启四类依赖子分区。
    return MemorySettings(
        enable=True,
        embedding={'enable': True},
        db={'enable': True, 'sqlite': {'path': './test_facade.db'}},
        vector={'enable': True},
        llm={'enable': True},
    )


def test_memory_server_builds_all_components(monkeypatch, memory_settings):
    # 四类依赖由门面内部按分区装配。
    _install_memory_fakes(monkeypatch)
    captured = {}

    def fake_memory(settings, embedder, llm, db, vector):
        captured['args'] = (embedder, llm, db, vector)
        return 'HANDLER'

    monkeypatch.setattr(facade, '_build_memory', fake_memory)
    server = facade.MemoryServer(memory_settings)

    assert server.settings is memory_settings
    assert server.embedder == 'EMB'
    assert server.llm == 'LLM'
    assert server.db == 'DB'
    assert server.vector == 'VEC'
    assert server.memory == 'HANDLER'
    assert captured['args'] == ('EMB', 'LLM', 'DB', 'VEC')


def test_memory_server_passes_section_settings(monkeypatch, memory_settings):
    # 各装配方法收到的应是 memory 分区下对应的子设置对象。
    captured = {}

    def record(name):
        def inner(settings):
            captured[name] = settings
            return name
        return inner

    monkeypatch.setattr(facade, '_build_embedder', lambda settings: 'EMB')
    monkeypatch.setattr(facade, '_build_llm', record('LLM'))
    monkeypatch.setattr(facade, '_build_db', record('DB'))
    monkeypatch.setattr(facade, '_build_vector', record('VEC'))
    monkeypatch.setattr(facade, '_build_memory', lambda *args, **kwargs: None)
    facade.MemoryServer(memory_settings)

    assert captured['LLM'] is memory_settings.llm
    assert captured['DB'] is memory_settings.db
    assert captured['VEC'] is memory_settings.vector


def test_memory_server_delegates_flows(monkeypatch, memory_settings):
    # 四条流程逐条转发给 MemoryHandler，服务层不做业务加工。
    handler = _FakeMemoryHandler()
    _install_memory_fakes(monkeypatch)
    monkeypatch.setattr(facade, '_build_memory', lambda *args, **kwargs: handler)
    server = facade.MemoryServer(memory_settings)

    assert server.add('hi', user_id='u1') == 'ADD'
    assert server.search('q', top_k=3) == 'SEARCH'
    assert server.update(1, text='t') == 'UPDATE'
    assert server.delete(1) == 'DELETE'
    assert [call[0] for call in handler.calls] == ['add', 'search', 'update', 'delete']


def test_memory_server_requires_enabled_memory():
    # 默认配置关闭记忆层：各组件为 None，四条流程统一抛 ConfigError。
    server = facade.MemoryServer(MemorySettings())
    assert server.memory is None
    assert server.embedder is None
    assert server.db is None
    assert server.vector is None
    assert server.llm is None
    for call in (lambda: server.add('hi'), lambda: server.search('q'), lambda: server.update(1), lambda: server.delete(1)):
        with pytest.raises(ConfigError):
            call()


def test_memory_server_signature_accepts_settings():
    # 构造签名只保留 settings，组件装配全部下沉到门面内部。
    import inspect

    parameters = list(inspect.signature(facade.MemoryServer.__init__).parameters)
    assert parameters == ['self', 'settings']

