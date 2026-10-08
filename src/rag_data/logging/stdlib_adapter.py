from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from rag_data.logging.base import LoggerAdapter


class StdlibTimezoneFormatter(logging.Formatter):
    """支持自定义时区与完整路径格式化的 Standard Logging Formatter"""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        tz: Optional[ZoneInfo] = None,
    ):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


class StdlibLogAdapter(LoggerAdapter):
    """标准库 logging 适配器"""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None) -> None:
        self._logger = logger
        self._extra: Dict[str, Any] = dict(extra) if extra else {}

    def bind(self, **fields: Any) -> StdlibLogAdapter:
        merged = {**self._extra, **fields}
        return StdlibLogAdapter(self._logger, extra=merged)

    def _emit(self, level: int, msg: str, fields: Dict[str, Any], exc_info: bool = False) -> None:
        merged = {**self._extra, **fields}
        extra_str = f"  {' '.join(f'{k}={v}' for k, v in merged.items())}" if merged else ""
        full_msg = f"{msg}{extra_str}"
        # stacklevel=3 跳过: 业务代码 -> adapter.info() -> adapter._emit()
        self._logger.log(level, full_msg, exc_info=exc_info, stacklevel=3)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, fields)

    def exception(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, fields, exc_info=True)