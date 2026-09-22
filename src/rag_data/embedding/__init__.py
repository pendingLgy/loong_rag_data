# 文本向量化：抽象接口、注册表与内置 provider。

from rag_data.embedding.base import BaseEmbeddingProvider
from rag_data.embedding.embedder import Embedder
from rag_data.embedding.openai_provider import OpenAIEmbeddingProvider
from rag_data.embedding.qwen_provider import QwenEmbeddingProvider
from rag_data.embedding.registry import (
    BUILTIN_PROVIDERS,
    available_embedding_providers,
    create_embedding_provider,
    is_embedding_registered,
    register_embedding,
    register_embedding_provider,
    resolve_embedding_provider,
)

__all__ = [
    "BaseEmbeddingProvider",
    "Embedder",
    "OpenAIEmbeddingProvider",
    "QwenEmbeddingProvider",
    "register_embedding",
    "register_embedding_provider",
    "available_embedding_providers",
    "is_embedding_registered",
    "resolve_embedding_provider",
    "create_embedding_provider",
    "BUILTIN_PROVIDERS",
]
