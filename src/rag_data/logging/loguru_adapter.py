# loguru 适配器，惰性导入以支持可选依赖。

from __future__ import annotations

from typing import Any, Dict, Optional

from rag_data.exceptions import OptionalDependencyError
from rag_data.logging.base import LoggerAdapter


def import_loguru() -> Any:
    """导入 loguru，未安装时抛出可读异常。"""
    try:
        from loguru import logger
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "loguru 未安装，请执行 pip install loguru，或将 log_backend 切换为 stdlib"
        ) from exc
    return logger


class LoguruAdapter(LoggerAdapter):
    """基于 loguru 的实现。"""

    def __init__(self, logger: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra: Dict[str, Any] = dict(extra) if extra else {}

    @classmethod
    def create(cls) -> "LoguruAdapter":
        return cls(import_loguru())

    def bind(self, **fields: Any) -> "LoguruAdapter":
        merged: Dict[str, Any] = dict(self._extra)
        merged.update(fields)
        return LoguruAdapter(self._logger, merged)

    def _emit(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        merged: Dict[str, Any] = dict(self._extra)
        merged.update(fields)
        bound = self._logger.bind(**merged) if merged else self._logger
        bound.log(level, msg)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("DEBUG", msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("INFO", msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit("WARNING", msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("ERROR", msg, fields)

    def exception(self, msg: str, **fields: Any) -> None:
        merged: Dict[str, Any] = dict(self._extra)
        merged.update(fields)
        bound = self._logger.bind(**merged) if merged else self._logger
        bound.exception(msg)
