from __future__ import annotations

from typing import Any, Dict, Optional

from rag_data.exceptions import OptionalDependencyError
from rag_data.logging.base import LoggerAdapter


def import_structlog() -> Any:
    """惰性导入 structlog"""
    try:
        import structlog
    except ImportError as exc:
        raise OptionalDependencyError(
            "structlog 未安装，请执行 pip install structlog"
        ) from exc
    return structlog


class StructlogAdapter(LoggerAdapter):
    """structlog 适配器"""

    def __init__(self, logger: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra: Dict[str, Any] = dict(extra) if extra else {}

    @classmethod
    def create(cls, name: str = "rag_data") -> StructlogAdapter:
        structlog = import_structlog()
        return cls(structlog.get_logger(name))

    def bind(self, **fields: Any) -> StructlogAdapter:
        merged = {**self._extra, **fields}
        return StructlogAdapter(self._logger.bind(**merged))

    def _emit(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        method = getattr(self._logger, level)
        merged = {**self._extra, **fields}
        # 纯净调用，不再传递 _stacklevel，靠 structlog 的 additional_ignores 自动过滤
        method(msg, **merged)

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