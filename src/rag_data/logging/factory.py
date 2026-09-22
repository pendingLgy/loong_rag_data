# 日志工厂：按配置装配后端，并对外提供 get_logger。

from __future__ import annotations

import importlib.util
import logging as _stdlib_logging
import sys
from typing import Optional

from rag_data.config import Settings
from rag_data.exceptions import OptionalDependencyError
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.formatters import HumanFormatter, JsonFormatter
from rag_data.logging.loguru_adapter import LoguruAdapter, import_loguru
from rag_data.logging.stdlib_adapter import StdlibAdapter
from rag_data.logging.structlog_adapter import StructlogAdapter, import_structlog

ROOT_LOGGER_NAME = "rag_data"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_BACKEND: str = "stdlib"
_ROOT: Optional[LoggerAdapter] = None


def level_value(level: str) -> int:
    """将级别名称转换为标准库级别数值。"""
    value = _stdlib_logging.getLevelName(level.upper())
    return value if isinstance(value, int) else _stdlib_logging.INFO


def is_available(module: str) -> bool:
    """判断可选模块是否可导入。"""
    return importlib.util.find_spec(module) is not None


def resolve_backend(settings: Settings) -> str:
    """确定实际使用的后端，auto 时按可用性择优选。"""
    configured = settings.logging.backend
    if configured != "auto":
        return configured
    # TODO: 自动探测优先级可按部署环境调整；当前为 structlog 优先、其次 loguru、最后 stdlib。
    if is_available("structlog"):
        return "structlog"
    if is_available("loguru"):
        return "loguru"
    return "stdlib"


def configure_logging(settings: Settings) -> LoggerAdapter:
    """初始化日志系统并返回根适配器。"""
    global _BACKEND, _ROOT
    backend = resolve_backend(settings)
    try:
        _ROOT = _build(backend, settings)
    except OptionalDependencyError:
        _ROOT = _build_stdlib(settings)
        backend = "stdlib"
        _ROOT.warning("指定日志后端不可用，已降级为标准库 logging")
    _BACKEND = backend
    return _ROOT


def get_logger(name: str) -> LoggerAdapter:
    """获取绑定到指定命名空间的日志适配器。"""
    global _ROOT
    if _ROOT is None:
        _ROOT = _build_stdlib(Settings())
    return _ROOT.bind(logger=name)


def _build(backend: str, settings: Settings) -> LoggerAdapter:
    if backend == "loguru":
        _configure_loguru(settings)
        return LoguruAdapter.create()
    if backend == "structlog":
        _configure_structlog(settings)
        return StructlogAdapter.create()
    return _build_stdlib(settings)


def _build_stdlib(settings: Settings) -> LoggerAdapter:
    root = _stdlib_logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level_value(settings.logging.level))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = _stdlib_logging.StreamHandler()
    if settings.logging.json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanFormatter(DEFAULT_FORMAT))
    root.addHandler(handler)
    root.propagate = False
    return StdlibAdapter(root)


def _configure_loguru(settings: Settings) -> None:
    # TODO: 可按需追加文件轮转与序列化等 sink 配置。
    logger = import_loguru()
    logger.remove()
    logger.add(sys.stderr, level=settings.logging.level.upper(), serialize=settings.logging.json_output)


def _configure_structlog(settings: Settings) -> None:
    # TODO: 可按需补充 contextvars 与 trace 等处理器。
    structlog = import_structlog()
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if settings.logging.json_output:
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_value(settings.logging.level)),
    )
