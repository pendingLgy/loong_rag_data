# 门面单元测试：模块级私有装配方法 _build_* 的门控与参数透传。
# 两个门面的实例用例分别在 test_facade_ragdata.py 与 test_facade_memoryserver.py。
#
# 用假向量模型与假数据库隔离网络与真实连接；分支装配由 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import MemorySettings, RagSetting
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ConfigError


# ---------------- 测试替身 ----------------


class _FakeModel:
    # 受控向量模型：记录每次 encode 的入参，返回固定维度向量。
    def __init__(self, dim, max_batch_size=0):
        self.dim = dim
        self.max_batch_size = max_batch_size
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * self.dim for _ in texts]


# ---------------- 夹具与构造 ----------------


@pytest.fixture()
def det_settings():
    # 不读取环境变量，保证配置可复现；各分区默认关闭，这里统一打开以便测试装配分支。
    return RagSetting.load_mapping(
        parsing={'enable': True},
        embedding={'enable': True},
    )


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


# ---------------- _build_nlp ----------------


def test_build_nlp_delegates_model(monkeypatch, det_settings):
    captured = {}
    sentinel = object()

    def fake_load(model_name):
        captured['model'] = model_name
        return sentinel

    monkeypatch.setattr(facade, 'load_nlp', fake_load)
    assert facade._build_nlp(det_settings.parsing) is sentinel
    assert captured['model'] == det_settings.parsing.spacy_model


def test_build_nlp_returns_handle_or_none(det_settings):
    result = facade._build_nlp(det_settings.parsing)
    assert result is None or callable(result)


# ---------------- _build_embedder ----------------


def test_build_embedder_injected_model(det_settings):
    model = _FakeModel(dim=det_settings.embedding.dim)
    result = facade._build_embedder(det_settings.embedding, model=model)
    assert isinstance(result, EmbedderHandler)
    assert result.provider is model


def test_build_embedder_without_model_is_lazy(det_settings):
    result = facade._build_embedder(det_settings.embedding)
    assert isinstance(result, EmbedderHandler)


def test_facade_all_symbols():
    assert set(facade.__all__) == {'RagData', 'MemoryServer'}


# ---------------- _build_db ----------------


class _FakeDB:
    # 受控数据库替身：记录被装配的配置与日志适配器。
    def __init__(self, settings):
        self.settings = settings


def _install_fake_db(monkeypatch):
    # 打桩 facade.build_database，捕获 DBSettings。
    captured = {}

    def fake_build_database(settings):
        captured['settings'] = settings
        return _FakeDB(settings)

    monkeypatch.setattr(facade, 'build_database', fake_build_database)
    return captured


def test_build_db_delegates_settings(monkeypatch, memory_settings):
    captured = _install_fake_db(monkeypatch)
    result = facade._build_db(memory_settings.db)
    assert isinstance(result, _FakeDB)
    assert captured['settings'] is memory_settings.db


def test_build_db_sqlite_uses_configured_path(monkeypatch, memory_settings):
    captured = _install_fake_db(monkeypatch)
    db = memory_settings.db.model_copy(update={
        'backend': 'sqlite',
        'sqlite': memory_settings.db.sqlite.model_copy(update={'path': 'store.db'}),
    })
    facade._build_db(db)
    assert captured['settings'].backend == 'sqlite'
    assert captured['settings'].sqlite.path == 'store.db'


def test_build_db_sqlite_requires_path():
    # 未配置 sqlite.path 时由装配层抛出明确配置错误。
    settings = MemorySettings(db={'enable': True}).db
    with pytest.raises(ConfigError):
        facade._build_db(settings)


def test_build_db_reads_sqlite_path_from_env(monkeypatch, tmp_path):
    captured = _install_fake_db(monkeypatch)
    path = str(tmp_path / 'env.db')
    monkeypatch.setenv('RAG_MEMORY__DB__BACKEND', 'sqlite')
    monkeypatch.setenv('RAG_MEMORY__DB__ENABLE', 'true')
    monkeypatch.setenv('RAG_MEMORY__DB__SQLITE__PATH', path)
    facade._build_db(MemorySettings().db)
    assert captured['settings'].sqlite.path == path


# ---------------- embedding 模块命名 ----------------


def test_embedding_modules_renamed():
    import rag_data.embedding.embedder_handler as handler_module
    import rag_data.embedding.openai_embedder as openai_module
    import rag_data.embedding.qwen_embedder as qwen_module
    import rag_data.embedding.registry_embedder as registry_module

    assert handler_module.__name__.endswith('.embedder_handler')
    assert registry_module.__name__.endswith('.registry_embedder')
    assert openai_module.__name__.endswith('.openai_embedder')
    assert qwen_module.__name__.endswith('.qwen_embedder')


def test_embedding_exports_handler_only():
    import rag_data.embedding as embedding_package

    assert 'EmbedderHandler' in embedding_package.__all__
    assert 'Embedder' not in embedding_package.__all__
    assert not hasattr(embedding_package, 'Embedder')


def test_embedding_old_module_paths_are_gone():
    import importlib

    for path in (
        'rag_data.embedding.embedder',
        'rag_data.embedding.registry',
        'rag_data.embedding.openai_provider',
        'rag_data.embedding.qwen_provider',
        'rag_data.embedding.registry_embedding',
    ):
        with pytest.raises(ImportError):
            importlib.import_module(path)


def test_embedding_registry_has_no_factory_api():
    import rag_data.embedding.registry_embedder as registry_module

    assert not hasattr(registry_module, 'create_embedding_provider')
    assert not hasattr(registry_module, 'create')


# ---------------- _build_vector ----------------


def test_build_vector_delegates_settings(monkeypatch, memory_settings):
    captured = {}

    def fake_builder(settings, **kwargs):
        captured['settings'] = settings
        return 'ENGINE'

    monkeypatch.setattr(facade, 'build_vector_engine', fake_builder)
    result = facade._build_vector(memory_settings.vector)
    assert result == 'ENGINE'
    assert captured['settings'] is memory_settings.vector


# ---------------- enable 门控 ----------------


def test_builders_return_none_when_disabled():
    # 各分区默认关闭时不初始化任何组件。
    settings = RagSetting.load_mapping()
    assert facade._build_nlp(settings.parsing) is None
    assert facade._build_embedder(settings.embedding) is None
    assert facade._build_db(MemorySettings().db) is None
    assert facade._build_vector(MemorySettings().vector) is None


def test_build_memory_returns_none_when_disabled():
    # memory.enable 为 False 时不构造 MemoryHandler。
    assert facade._build_memory(MemorySettings(), None, None, None, None) is None


# ---------------- MemoryServer ----------------

