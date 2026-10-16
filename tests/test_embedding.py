# 向量化模块单元测试：纯注册表、内置 embedder 与 EmbedderHandler 处理器。
#
# 不触网络：内置 embedder 仅校验类属性与配置解析；调用路径用注入的假模型或假 embedder 隔离。

import pytest

from rag_data.config import EmbeddingSettings
from rag_data.embedding import (
    BUILTIN_EMBEDDERS,
    BaseEmbedder,
    EmbedderHandler,
    OpenAIEmbedder,
    QwenEmbedder,
    available_embedders,
    is_embedder_registered,
    register_embedder,
    register_embedder_provider,
    resolve_embedder,
)
from rag_data.exceptions import ConfigError, EmbeddingError


# ---------------- 测试替身 ----------------


class _FakeModel:
    # 受控向量模型：记录每次 encode 的入参，返回固定维度向量。
    def __init__(self, dim=4, max_batch_size=0):
        self.dim = dim
        self.max_batch_size = max_batch_size
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[0.5] * self.dim for _ in texts]


# ---------------- 注册表 ----------------


def test_builtin_embedders_declared():
    assert set(BUILTIN_EMBEDDERS) == {'openai', 'qwen'}


def test_available_embedders_sorted():
    names = available_embedders()
    assert names == sorted(names)
    for name in ('openai', 'qwen'):
        assert is_embedder_registered(name)


def test_resolve_embedder_returns_class():
    assert resolve_embedder('openai') is OpenAIEmbedder
    assert resolve_embedder('qwen') is QwenEmbedder


def test_resolve_unknown_embedder_raises():
    with pytest.raises(ConfigError):
        resolve_embedder('nope')


def test_register_embedder_by_class_object():
    class _Local(BaseEmbedder):
        backend = None
        default_dim = 3

        def encode(self, texts): return [[0.0] * 3 for _ in texts]

    register_embedder('local-emb', _Local)
    assert is_embedder_registered('local-emb')
    assert resolve_embedder('local-emb') is _Local
    assert 'local-emb' in available_embedders()


def test_register_embedder_by_decorator():
    name = 'deco-emb'

    @register_embedder_provider(name)
    class _Dummy(BaseEmbedder):
        backend = None

        def encode(self, texts): return []

    assert is_embedder_registered(name)
    assert resolve_embedder(name) is _Dummy


def test_backend_declaration_auto_registers():
    class _Auto(BaseEmbedder):
        backend = 'auto-emb'

        def encode(self, texts): return []

    # 声明 backend 即在类定义时自动注册。
    assert is_embedder_registered('auto-emb')
    assert resolve_embedder('auto-emb') is _Auto


# ---------------- 内置 embedder ----------------


def test_openai_embedder_defaults():
    assert OpenAIEmbedder.backend == 'openai'
    assert OpenAIEmbedder.default_model == 'text-embedding-3-small'
    assert OpenAIEmbedder.api_key_env == 'OPENAI_API_KEY'


def test_qwen_embedder_overrides_openai():
    assert issubclass(QwenEmbedder, OpenAIEmbedder)
    assert QwenEmbedder.backend == 'qwen'
    assert QwenEmbedder.default_model == 'qwen3.7-text-embedding'
    assert QwenEmbedder.api_key_env == 'DASHSCOPE_API_KEY'
    assert QwenEmbedder.max_batch_size == 10


def test_base_embedder_is_abstract():
    # 抽象基类不可直接实例化。
    with pytest.raises(TypeError):
        BaseEmbedder(EmbeddingSettings(), None)


def test_class_names_use_embedder_suffix():
    # 命名规范：实现类以 Embedder 结尾，不出现 Provider 字样的类。
    for cls in (BaseEmbedder, OpenAIEmbedder, QwenEmbedder, EmbedderHandler):
        assert cls.__name__.endswith('Embedder') or cls.__name__.endswith('EmbedderHandler')


# ---------------- EmbedderHandler 处理器 ----------------


def test_handler_uses_injected_model():
    model = _FakeModel(dim=4)
    handler = EmbedderHandler(EmbeddingSettings(dim=4), None, model=model)
    assert handler.provider is model
    assert handler.encode(['a', 'b']) == [[0.5] * 4, [0.5] * 4]
    assert model.calls == [['a', 'b']]


def test_handler_empty_input_returns_empty():
    handler = EmbedderHandler(EmbeddingSettings(dim=4), None, model=_FakeModel(dim=4))
    assert handler.encode([]) == []


def test_handler_defers_instantiation():
    created = []

    class _Lazy(BaseEmbedder):
        backend = None
        default_dim = 2

        def __init__(self, settings, logger):
            super().__init__(settings, logger)
            created.append(self)

        def encode(self, texts): return [[0.0] * 2 for _ in texts]

    register_embedder('lazy-emb', _Lazy)
    handler = EmbedderHandler(EmbeddingSettings(provider='lazy-emb', dim=2), None)
    assert created == []
    handler.encode(['x'])
    assert len(created) == 1


def test_handler_caches_provider():
    created = []

    class _Counting(BaseEmbedder):
        backend = None
        default_dim = 2

        def __init__(self, settings, logger):
            super().__init__(settings, logger)
            created.append(self)

        def encode(self, texts): return [[0.0] * 2 for _ in texts]

    register_embedder('counting-emb', _Counting)
    handler = EmbedderHandler(EmbeddingSettings(provider='counting-emb', dim=2), None)
    assert handler.provider is handler.provider
    assert len(created) == 1


def test_handler_rejects_dim_mismatch():
    handler = EmbedderHandler(EmbeddingSettings(dim=8), None, model=_FakeModel(dim=4))
    with pytest.raises(EmbeddingError):
        handler.encode(['x'])


def test_handler_propagates_unknown_provider():
    handler = EmbedderHandler(EmbeddingSettings(provider='definitely-missing'), None)
    with pytest.raises(ConfigError):
        handler.encode(['x'])


def test_handler_respects_provider_batch_limit():
    class _Limited(BaseEmbedder):
        backend = None
        default_dim = 2
        max_batch_size = 1

        def encode(self, texts): return [[0.0] * 2 for _ in texts]

    register_embedder('limited-emb', _Limited)
    handler = EmbedderHandler(EmbeddingSettings(provider='limited-emb', dim=2, batch_size=10), None)
    # 取配置值与 embedder 上限的较小者，单批最多 1 条。
    target = handler.provider
    assert handler._batch_size(target) == 1


def test_handler_batch_size_falls_back_to_config():
    handler = EmbedderHandler(EmbeddingSettings(dim=2, batch_size=7), None, model=_FakeModel(dim=2))
    assert handler._batch_size(handler.provider) == 7


def test_old_registry_names_are_gone():
    import importlib

    module = importlib.import_module('rag_data.embedding.registry_embedder')
    for old in (
        'register_embedding',
        'register_embedding_provider',
        'is_embedding_registered',
        'available_embedding_providers',
        'resolve_embedding_provider',
        'BUILTIN_PROVIDERS',
    ):
        assert not hasattr(module, old)


def test_new_registry_names_are_exported():
    import importlib

    module = importlib.import_module('rag_data.embedding.registry_embedder')
    for name in (
        'register_embedder',
        'register_embedder_provider',
        'is_embedder_registered',
        'available_embedders',
        'resolve_embedder',
        'BUILTIN_EMBEDDERS',
    ):
        assert hasattr(module, name)

