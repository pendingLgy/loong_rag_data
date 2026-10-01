# rag-data 公共入口文件（Public API）。
#
# 外部调用只需引入本包，无需关心内部子包路径：
#
#   import rag_data
#   from rag_data import Settings, IngestionPipeline
#
# 本文件仅做统一导出，不承载任何业务逻辑；内部实现仍按模块分布在各自子包中。

from rag_data.__about__ import __version__
from rag_data.config import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    ChunkingSettings,
    EmbeddingSettings,
    LoggingSettings,
    NLPSettings,
    Settings,
    StorageSettings,
    load_env_overrides,
)
from rag_data.embedding.base import BaseEmbeddingProvider
from rag_data.embedding.embedder import Embedder
from rag_data.embedding.openai_provider import OpenAIEmbeddingProvider
from rag_data.embedding.qwen_provider import QwenEmbeddingProvider
from rag_data.embedding.registry import (
    available_embedding_providers,
    create_embedding_provider,
    is_embedding_registered,
    register_embedding,
    register_embedding_provider,
    resolve_embedding_provider,
)
from rag_data.exceptions import (
    ConfigError,
    DataError,
    EmbeddingError,
    OptionalDependencyError,
    ParseError,
    ParserDependencyError,
    RagDataError,
    SchemaMismatchError,
    StoreError,
)
from rag_data.facade import (
    RagData,
    build_embedder,
    build_logger,
    build_nlp,
    build_pipeline,
    build_settings,
    build_store,
    ingest,
    init_collection,
)
from rag_data.ingestion.chunking import build_semantic_chunks
from rag_data.ingestion.entities import extract_entities
from rag_data.ingestion.parsers import parse_document
from rag_data.ingestion.pipeline import IngestionPipeline
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import configure_logging, get_logger
from rag_data.models import DocumentChunk
from rag_data.storage.milvus_store import MilvusRecord
from rag_data.store_registry import (
    BUILTIN_BACKENDS,
    available_backends,
    create_store,
    is_registered,
    load_store_modules,
    register_backend,
    register_store,
    resolve_store,
)

__all__ = [
    "config",
    "models",
    "ingestion",
    "embedding",
    "storage",
    "logging",
    # 流程门面
    "RagData",
    "init_collection",
    "ingest",
    "build_settings",
    "build_logger",
    "build_store",
    "build_nlp",
    "build_embedder",
    "build_pipeline",
    # 版本
    "__version__",
    # 配置
    "Settings",
    "StorageSettings",
    "ChunkingSettings",
    "EmbeddingSettings",
    "NLPSettings",
    "LoggingSettings",
    "load_env_overrides",
    "ENV_PREFIX",
    "ENV_NESTED_DELIMITER",
    # 数据模型
    "DocumentChunk",
    "MilvusRecord",
    # 导入管道
    "IngestionPipeline",
    "parse_document",
    "build_semantic_chunks",
    "extract_entities",
    # 向量化
    "Embedder",
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "QwenEmbeddingProvider",
    "register_embedding",
    "register_embedding_provider",
    "available_embedding_providers",
    "is_embedding_registered",
    "resolve_embedding_provider",
    "create_embedding_provider",
    # 存储
    "register_store",
    "register_backend",
    "available_backends",
    "is_registered",
    "resolve_store",
    "create_store",
    "load_store_modules",
    "BUILTIN_BACKENDS",
    # 日志适配层
    "LoggerAdapter",
    "configure_logging",
    "get_logger",
    # 异常
    "RagDataError",
    "ConfigError",
    "DataError",
    "OptionalDependencyError",
    "ParserDependencyError",
    "ParseError",
    "SchemaMismatchError",
    "StoreError",
    "EmbeddingError",
]
