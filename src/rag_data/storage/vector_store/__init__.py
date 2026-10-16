# 向量库存储：抽象接口、纯注册表、向量引擎与内置实现。

from rag_data.storage.vector_store.base import MetricType, VectorStoreBase, VectorStoreResult
from rag_data.storage.vector_store.registry_vector import (
    BUILTIN_STORES,
    available_vector_stores,
    is_vector_store_registered,
    register_vector_store,
    register_vector_store_provider,
    resolve_vector_store,
)
from rag_data.storage.vector_store.vector_engine import (
    VectorEngine,
    build_vector_store,
    create_vector_engine,
)


__all__ = [
    "MetricType",
    "VectorStoreBase",
    "VectorStoreResult",
    "register_vector_store",
    "register_vector_store_provider",
    "available_vector_stores",
    "is_vector_store_registered",
    "resolve_vector_store",
    "BUILTIN_STORES",
    "VectorEngine",
    "build_vector_store",
    "create_vector_engine",
]

