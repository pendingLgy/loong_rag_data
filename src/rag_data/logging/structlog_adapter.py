# structlog 适配器，惰性导入以支持可选依赖。

from __future__ import annotations

from typing import Any, Dict, Optional

from rag_data.exceptions import OptionalDependencyError
from rag_data.logging.base import LoggerAdapter


def import_structlog() -> Any:
    """导入 structlog，未安装时抛出可读异常。"""
    try:
        import structlog
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "structlog 未安装，请执行 pip install structlog，或将 log_backend 切换为 stdlib"
        ) from exc
    return structlog


class StructlogAdapter(LoggerAdapter):
    """基于 structlog 的实现。"""

    def __init__(self, logger: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra: Dict[str, Any] = dict(extra) if extra else {}

    @classmethod
    def create(cls) -> "StructlogAdapter":
        structlog = import_structlog()
        return cls(structlog.get_logger())

    def bind(self, **fields: Any) -> "StructlogAdapter":
        merged: Dict[str, Any] = dict(self._extra)
        merged.update(fields)
        return StructlogAdapter(self._logger.bind(**merged))

    def _emit(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        method = getattr(self._logger, level)
        if fields:
            method(msg, **fields)
        else:
            method(msg)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit("debug", msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit("info", msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit("warning", msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit("error", msg, fields)

    def exception(self, msg: str, **fields: Any) -> None:
        self._emit("exception", msg, fields)
