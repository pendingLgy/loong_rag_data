# 日志抽象层：业务代码只依赖此接口，屏蔽具体框架。

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Dict, Optional, Type


class LoggerAdapter(ABC):
    """统一的日志接口。"""

    @abstractmethod
    def bind(self, **fields: Any) -> "LoggerAdapter":
        """返回绑定了额外字段的新实例，不修改当前实例。"""

    @abstractmethod
    def debug(self, msg: str, **fields: Any) -> None:
        """DEBUG 级别。"""

    @abstractmethod
    def info(self, msg: str, **fields: Any) -> None:
        """INFO 级别。"""

    @abstractmethod
    def warning(self, msg: str, **fields: Any) -> None:
        """WARNING 级别。"""

    @abstractmethod
    def error(self, msg: str, **fields: Any) -> None:
        """ERROR 级别。"""

    @abstractmethod
    def exception(self, msg: str, **fields: Any) -> None:
        """ERROR 级别并附带当前异常堆栈。"""

    def context(self, **fields: Any) -> "LogContext":
        """返回上下文管理器，在 with 块内临时绑定字段。"""
        return LogContext(self, fields)


class LogContext:
    """在 with 块内临时绑定字段的上下文管理器。"""

    def __init__(self, adapter: LoggerAdapter, fields: Dict[str, Any]) -> None:
        self._adapter = adapter
        self._fields = dict(fields)

    def __enter__(self) -> LoggerAdapter:
        return self._adapter.bind(**self._fields)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        return None
