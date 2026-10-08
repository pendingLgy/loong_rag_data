from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LoggerAdapter(ABC):
    """统一日志适配器基类"""

    @abstractmethod
    def bind(self, **fields: Any) -> LoggerAdapter:
        """绑定上下文字段，返回新的 Adapter 实例"""
        pass

    @abstractmethod
    def debug(self, msg: str, **fields: Any) -> None:
        pass

    @abstractmethod
    def info(self, msg: str, **fields: Any) -> None:
        pass

    @abstractmethod
    def warning(self, msg: str, **fields: Any) -> None:
        pass

    @abstractmethod
    def error(self, msg: str, **fields: Any) -> None:
        pass

    @abstractmethod
    def exception(self, msg: str, **fields: Any) -> None:
        pass