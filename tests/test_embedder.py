# 向量化封装的单元测试。

import pytest

from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.exceptions import EmbeddingError


class _FakeModel:
    def __init__(self, dim, bad_dim=False):
        self._dim = dim
        self._bad_dim = bad_dim

    def encode(self, texts):
        dim = self._dim - 1 if self._bad_dim else self._dim
        return [[0.1] * dim for _ in texts]


def test_encode_empty_returns_empty(logger):
    settings = Settings()
    embedder = Embedder(settings, logger, model=_FakeModel(settings.storage.vector_dim))
    assert embedder.encode([]) == []


def test_encode_batches_and_validates_dim(logger):
    settings = Settings(embedding={"batch_size": 2})
    embedder = Embedder(settings, logger, model=_FakeModel(settings.storage.vector_dim))
    vectors = embedder.encode(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(vectors[0]) == settings.storage.vector_dim


def test_provider_without_api_key_raises(logger, monkeypatch):
    # 默认 provider 为 openai，缺少 API Key 时应在发起请求前即报错
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = Embedder(Settings(), logger)
    with pytest.raises(EmbeddingError):
        embedder.encode(["a"])


def test_wrong_dimension_raises(logger):
    settings = Settings()
    embedder = Embedder(settings, logger, model=_FakeModel(settings.storage.vector_dim, bad_dim=True))
    with pytest.raises(EmbeddingError):
        embedder.encode(["a"])
