# 向量库抽象、注册表与内置实现。

from rag_data.storage.base import BaseVectorStore
from rag_data.storage.memory_store import InMemoryVectorStore
from rag_data.storage.registry import (
    BUILTIN_BACKENDS,
    available_backends,
    create_store,
    is_registered,
    register_backend,
    register_store,
    resolve_store,
)

__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "register_store",
    "register_backend",
    "available_backends",
    "is_registered",
    "resolve_store",
    "create_store",
    "BUILTIN_BACKENDS",
]
