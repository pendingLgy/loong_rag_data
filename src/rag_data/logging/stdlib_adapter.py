# 标准库 logging 适配器，零额外依赖，作为默认回退。

from __future__ import annotations

import logging as _stdlib_logging
from typing import Any, Dict, Optional

from rag_data.logging.base import LoggerAdapter

RAG_FIELDS_ATTR = "rag_fields"


class StdlibAdapter(LoggerAdapter):
    """基于标准库 logging 的实现。"""

    def __init__(self, logger: _stdlib_logging.Logger, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra: Dict[str, Any] = dict(extra) if extra else {}

    def bind(self, **fields: Any) -> "StdlibAdapter":
        merged: Dict[str, Any] = dict(self._extra)
        merged.update(fields)
        return StdlibAdapter(self._logger, merged)

    def _emit(self, level: int, msg: str, fields: Dict[str, Any], exc_info: bool = False) -> None:
        merged: Dict[str, Any] = dict(self._extra)
        merged.update(fields)
        self._logger.log(level, msg, extra={RAG_FIELDS_ATTR: merged}, exc_info=exc_info)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit(_stdlib_logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(_stdlib_logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit(_stdlib_logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit(_stdlib_logging.ERROR, msg, fields)

    def exception(self, msg: str, **fields: Any) -> None:
        self._emit(_stdlib_logging.ERROR, msg, fields, exc_info=True)
