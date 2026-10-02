# 领域异常基类，统一上层捕获入口。

from __future__ import annotations


class RagDataError(Exception):
    """所有领域异常的基类。"""


class ConfigError(RagDataError):
    """配置缺失或非法。"""


class DataError(RagDataError):
    """数据校验失败。"""


class OptionalDependencyError(RagDataError):
    """可选依赖未安装。"""


class ParserDependencyError(OptionalDependencyError):
    """文档解析依赖未安装。"""


class ParseError(RagDataError):
    """文档损坏或格式不支持。"""


class EmbeddingError(RagDataError):
    """向量化失败。"""
