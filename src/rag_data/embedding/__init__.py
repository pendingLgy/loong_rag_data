# 文本向量化：抽象接口、纯注册表（供用户注册自定义 embedder）与内置实现；实例化在适配器调用时进行。

from rag_data.embedding.base import BaseEmbedder
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.embedding.openai_embedder import OpenAIEmbedder
from rag_data.embedding.qwen_embedder import QwenEmbedder
from rag_data.embedding.registry_embedder import (
    BUILTIN_EMBEDDERS,
    available_embedders,
    is_embedder_registered,
    register_embedder,
    register_embedder_provider,
    resolve_embedder,
)

__all__ = [
    "BaseEmbedder",
    "EmbedderHandler",
    "OpenAIEmbedder",
    "QwenEmbedder",
    "register_embedder",
    "register_embedder_provider",
    "available_embedders",
    "is_embedder_registered",
    "resolve_embedder",
    "BUILTIN_EMBEDDERS",
]
