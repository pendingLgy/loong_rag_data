from __future__ import annotations

from typing import Any, Dict, Optional

from rag_data.exceptions import OptionalDependencyError
from rag_data.logging.base import LoggerAdapter


def import_loguru() -> Any:
    """惰性导入 loguru"""
    try:
        from loguru import logger
    except ImportError as exc:
        raise OptionalDependencyError(
            "loguru 未安装，请执行 pip install loguru"
        ) from exc
    return logger


class LoguruAdapter(LoggerAdapter):
    """loguru 适配器"""

    def __init__(self, logger: Any, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra: Dict[str, Any] = dict(extra) if extra else {}

    @classmethod
    def create(cls) -> LoguruAdapter:
        logger = import_loguru()
        return cls(logger)

    def bind(self, **fields: Any) -> LoguruAdapter:
        merged = {**self._extra, **fields}
        return LoguruAdapter(self._logger.bind(**merged))

    def _emit(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        merged = {**self._extra, **fields}
        # opt(depth=2) 跳过: 业务代码 -> adapter.info() -> adapter._emit()
        logger_bound = self._logger.opt(depth=2)
        if merged:
            logger_bound = logger_bound.bind(**merged)
        getattr(logger_bound, level)(msg)

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