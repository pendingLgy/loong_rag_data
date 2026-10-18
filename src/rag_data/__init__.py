# rag-data 包根：仅导出当前目录（直接模块）的公共符号，不含子包与子目录内容。
#
# 子包 parsers、embedding、llm、logging 的内容请按具体模块路径取用：
#
#   from rag_data.parsers import ParserHandler
#   from rag_data.embedding import EmbedderAdapter
#   from rag_data.logging import LoggerFactory

from rag_data.__about__ import __version__
from rag_data.config import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    MEMORY_ENV_PREFIX,
    SECTION_EMBEDDING,
    SECTION_LOGGING,
    SECTION_PARSING,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    MemorySettings,
    ParsingSettings,
    RagSetting,
)
from rag_data.exceptions import (
    ConfigError,
    DataError,
    EmbeddingError,
    LLMError,
    OptionalDependencyError,
    ParseError,
    ParserDependencyError,
    RagDataError,
)
from rag_data.facade import MemoryServer, RagData
from rag_data.registry import Ref, Registry

__all__ = [
    "__version__",
    # 配置
    "RagSetting",
    "ParsingSettings",
    "EmbeddingSettings",
    "LoggingSettings",
    "LLMSettings",
    "MemorySettings",
    "ENV_PREFIX",
    "ENV_NESTED_DELIMITER",
    "MEMORY_ENV_PREFIX",
    "SECTION_PARSING",
    "SECTION_EMBEDDING",
    "SECTION_LOGGING",
    # 契约异常
    "RagDataError",
    "ConfigError",
    "DataError",
    "OptionalDependencyError",
    "ParserDependencyError",
    "ParseError",
    "EmbeddingError",
    "LLMError",
    # 通用设施
    "Registry",
    "Ref",
    # 流程门面
    "MemoryServer",
    "RagData",
]
