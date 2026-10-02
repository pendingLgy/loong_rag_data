# 嵌入模型注册表与 provider 的单元测试：配置驱动切换、请求构造与响应解析。

import urllib.error
import urllib.request

import pytest

from rag_data.config import Settings
from rag_data.embedding.base import BaseEmbeddingProvider
from rag_data.embedding.embedder import Embedder
from rag_data.embedding.openai_provider import OpenAIEmbeddingProvider
from rag_data.embedding.qwen_provider import QwenEmbeddingProvider
from rag_data.embedding.registry import (
    available_embedding_providers,
    register_embedding_provider,
    resolve_embedding_provider,
)
from rag_data.exceptions import ConfigError, EmbeddingError


class _RecordingProvider(OpenAIEmbeddingProvider):
    """记录请求内容并返回固定维度向量，用于隔离网络。"""

    # 不声明 backend，避免覆盖内置的 openai 注册项。

    def __init__(self, settings, logger):
        super().__init__(settings, logger)
        self.requests = []

    def _post_json(self, url, payload):
        self.requests.append((url, payload))
        dim = self.dim
        return {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(len(payload["input"]))]}


class _BatchyProvider(OpenAIEmbeddingProvider):
    """批量上限为 2 的假 provider，用于验证批次切分。"""

    default_model = "fake-model"
    default_dim = 4
    default_base_url = "http://example.invalid/v1"
    api_key_env = "FAKE_API_KEY"
    max_batch_size = 2

    def __init__(self, settings, logger):
        super().__init__(settings, logger)
        self.calls = []

    def _post_json(self, url, payload):
        self.calls.append(len(payload["input"]))
        return {"data": [{"index": i, "embedding": [0.5] * 4} for i in range(len(payload["input"]))]}


class _FakeModel:
    """自定义注入模型：只需实现 encode。"""

    def __init__(self, dim):
        self._dim = dim

    def encode(self, texts):
        return [[0.2] * self._dim for _ in texts]


@pytest.fixture(autouse=True)
def _restore_registry():
    """用例结束还原嵌入模型注册表，避免相互污染。"""

    from rag_data.embedding import registry

    snapshot = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


@pytest.fixture()
def batchy():
    """把假 provider 注册为 batchy；注册表由 autouse fixture 还原。"""

    from rag_data.embedding.registry import register_embedding

    register_embedding("batchy", _BatchyProvider)
    return "batchy"

# ---------- 注册表与内置 provider ----------


def test_builtin_providers_registered():
    assert available_embedding_providers() == ["openai", "qwen"]


def test_resolve_builtin_providers():
    assert resolve_embedding_provider("openai") is OpenAIEmbeddingProvider
    assert resolve_embedding_provider("qwen") is QwenEmbeddingProvider


def test_qwen_reuses_openai_implementation():
    assert issubclass(QwenEmbeddingProvider, OpenAIEmbeddingProvider)


def test_provider_defaults():
    assert OpenAIEmbeddingProvider.default_model == "text-embedding-3-small"
    assert OpenAIEmbeddingProvider.default_dim == 1536
    assert OpenAIEmbeddingProvider.api_key_env == "OPENAI_API_KEY"
    # 默认模型由 provider 自行声明；校验其存在与配套不变量，避免锁死具体模型名。
    assert QwenEmbeddingProvider.default_model
    assert QwenEmbeddingProvider.default_dim == 1024
    assert QwenEmbeddingProvider.api_key_env == "DASHSCOPE_API_KEY"
    assert QwenEmbeddingProvider.max_batch_size == 10
    assert QwenEmbeddingProvider.default_dim == 1024
    assert QwenEmbeddingProvider.api_key_env == "DASHSCOPE_API_KEY"
    assert QwenEmbeddingProvider.max_batch_size == 10


def test_unknown_provider_raises_with_hint(logger):
    embedder = Embedder(Settings(embedding={"provider": "nope"}), logger)
    with pytest.raises(ConfigError) as excinfo:
        embedder.encode(["a"])
    message = str(excinfo.value)
    assert "nope" in message and "openai" in message


def test_register_custom_provider_via_decorator(logger):
    @register_embedding_provider("mine")
    class _Mine(BaseEmbeddingProvider):
        def encode(self, texts):
            return [[0.0] for _ in texts]

    assert "mine" in available_embedding_providers()
    assert resolve_embedding_provider("mine") is _Mine


def test_provider_without_backend_is_not_auto_registered():
    before = available_embedding_providers()

    class _Plain(OpenAIEmbeddingProvider):
        pass

    assert available_embedding_providers() == before


# ---------- 配置解析 ----------


def test_config_defaults_to_openai():
    config = Settings()
    assert config.embedding.provider == "openai"
    assert config.embedding.model == ""
    assert config.embedding.dim == 1024
    assert config.embedding.api_key == ""
    assert config.embedding.base_url == ""
    assert config.embedding.batch_size == 128
    assert config.embedding.timeout == 60.0


def test_provider_falls_back_to_defaults(logger):
    provider = QwenEmbeddingProvider(Settings(), logger)
    assert provider.model == QwenEmbeddingProvider.default_model
    assert provider.dim == 1024
    assert provider.base_url.startswith("https:")


def test_provider_resolves_config_overrides(logger):
    settings = Settings(
        embedding={
            "provider": "qwen",
            "model": "text-embedding-v2",
            "dim": 512,
            "base_url": "http://gateway.internal/v1",
            "api_key": "cfg-key",
        }
    )
    provider = QwenEmbeddingProvider(settings, logger)
    assert provider.model == "text-embedding-v2"
    assert provider.dim == 512
    assert provider.base_url == "http://gateway.internal/v1"
    assert provider.api_key == "cfg-key"


def test_api_key_from_env(logger, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    assert QwenEmbeddingProvider(Settings(), logger).api_key == "env-key"


def test_api_key_config_wins_over_env(logger, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    settings = Settings(embedding={"api_key": "cfg-key"})
    assert QwenEmbeddingProvider(settings, logger).api_key == "cfg-key"


def test_api_key_missing_raises(logger, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIEmbeddingProvider(Settings(), logger)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.api_key
    assert "OPENAI_API_KEY" in str(excinfo.value)

# ---------- 请求构造与响应解析 ----------


def test_payload_uses_default_model(logger):
    provider = _RecordingProvider(Settings(embedding={"api_key": "k", "dim": 1536}), logger)
    provider.encode(["a", "b"])
    url, payload = provider.requests[0]
    assert url.endswith("/embeddings")
    assert payload["model"] == "text-embedding-3-small"
    assert payload["input"] == ["a", "b"]
    assert "dimensions" not in payload


def test_payload_includes_dimensions_when_overridden(logger):
    settings = Settings(embedding={"api_key": "k", "dim": 512})
    provider = _RecordingProvider(settings, logger)
    provider.encode(["a"])
    assert provider.requests[0][1]["dimensions"] == 512


def test_headers_carry_bearer_token(logger):
    provider = _RecordingProvider(Settings(embedding={"api_key": "k"}), logger)
    assert provider._headers()["Authorization"] == "Bearer k"


def test_parse_sorts_by_index(logger):
    provider = _RecordingProvider(Settings(embedding={"api_key": "k"}), logger)
    data = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    assert provider._parse(data) == [[1.0], [2.0]]


def test_parse_without_data_raises(logger):
    provider = _RecordingProvider(Settings(embedding={"api_key": "k"}), logger)
    with pytest.raises(EmbeddingError):
        provider._parse({})


def test_http_error_is_wrapped(logger, monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.HTTPError("http://example.invalid", 401, "unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    provider = OpenAIEmbeddingProvider(Settings(embedding={"api_key": "k"}), logger)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.encode(["a"])
    assert "401" in str(excinfo.value)


def test_url_error_is_wrapped(logger, monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    provider = OpenAIEmbeddingProvider(Settings(embedding={"api_key": "k"}), logger)
    with pytest.raises(EmbeddingError) as excinfo:
        provider.encode(["a"])
    assert "无法连接" in str(excinfo.value)

# ---------- Embedder 配置驱动 ----------


def test_embedder_uses_openai_by_default(logger, monkeypatch):
    seen = {}

    def fake_post(self, url, payload):
        seen["url"] = url
        seen["model"] = payload["model"]
        return {"data": [{"index": 0, "embedding": [0.1] * 4}]}

    monkeypatch.setattr(OpenAIEmbeddingProvider, "_post_json", fake_post)
    settings = Settings(embedding={"dim": 4, "api_key": "k"})
    assert len(Embedder(settings, logger).encode(["a"])) == 1
    assert seen["model"] == "text-embedding-3-small"
    assert "api.openai.com" in seen["url"]


def test_embedder_uses_qwen_when_configured(logger, monkeypatch):
    seen = {}

    def fake_post(self, url, payload):
        seen["url"] = url
        seen["model"] = payload["model"]
        return {"data": [{"index": 0, "embedding": [0.1] * 4}]}

    monkeypatch.setattr(QwenEmbeddingProvider, "_post_json", fake_post)
    settings = Settings(embedding={"dim": 4, "provider": "qwen", "api_key": "k"})
    Embedder(settings, logger).encode(["a"])
    assert seen["model"] == QwenEmbeddingProvider.default_model
    # 端点取自 provider 自述的 base_url，不锁死具体域名。
    assert QwenEmbeddingProvider.default_base_url in seen["url"]


def test_embedder_honors_provider_batch_limit(logger, batchy):
    settings = Settings(embedding={"dim": 4, "provider": batchy, "api_key": "k", "batch_size": 10})
    embedder = Embedder(settings, logger)
    assert len(embedder.encode(["1", "2", "3", "4", "5"])) == 5
    # provider 单次上限为 2，故 5 条被切成 2 + 2 + 1
    assert getattr(embedder.provider, "calls") == [2, 2, 1]


def test_embedder_injected_model_wins_over_provider(logger):
    settings = Settings(embedding={"dim": 3, "provider": "qwen"})
    embedder = Embedder(settings, logger, model=_FakeModel(3))
    assert len(embedder.encode(["a"])[0]) == 3


def test_embedder_validates_dimension(logger, batchy):
    # batchy 输出 4 维，与 embedding.dim 不一致时应报错
    settings = Settings(embedding={"dim": 3, "provider": batchy, "api_key": "k"})
    with pytest.raises(EmbeddingError):
        Embedder(settings, logger).encode(["a"])


def test_embedder_empty_input_skips_provider(logger):
    settings = Settings(embedding={"dim": 3, "provider": "missing"})
    assert Embedder(settings, logger).encode([]) == []
