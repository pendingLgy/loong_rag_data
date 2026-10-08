from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Literal, Optional
from zoneinfo import ZoneInfo

# 1. 兼容 Python 3.11+ 标准库 tomllib 与低版本 tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # 未安装 tomli，将优雅降级

from rag_data.logging.base import LoggerAdapter
from rag_data.logging.loguru_adapter import LoguruAdapter, import_loguru
from rag_data.logging.stdlib_adapter import StdlibLogAdapter, StdlibTimezoneFormatter
from rag_data.logging.structlog_adapter import StructlogAdapter, import_structlog

BackendType = Literal["stdliblog", "structlog", "loguru"]


def _load_pytest_log_format_from_toml() -> Optional[str]:
    """尝试读取 pyproject.toml 中的 [tool.pytest.ini_options].log_cli_format"""
    if tomllib is None:
        return None

    # 从当前文件向上查找项目根目录的 pyproject.toml
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        toml_path = parent / "pyproject.toml"
        if toml_path.exists():
            try:
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                return (
                    data.get("tool", {})
                    .get("pytest", {})
                    .get("ini_options", {})
                    .get("log_cli_format")
                )
            except Exception:
                return None
    return None


class LoggerFactory:
    """全局日志工厂与配置中心"""

    _backend: BackendType = "stdliblog"
    _timezone: ZoneInfo = datetime.now().astimezone().tzinfo or ZoneInfo("UTC")

    @classmethod
    def setup(
        cls,
        backend: BackendType = "stdliblog",
        timezone_name: Optional[str] = None,
        level: str = "INFO",
        fmt: Optional[str] = None,
    ) -> None:
        """全局初始化配置

        :param backend: 选择激活的后端 ("stdliblog" | "structlog" | "loguru")
        :param timezone_name: 时区字符串。为 None 时默认系统当前时区。
        :param level: 最低日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        :param fmt: 自定义日志格式。若为 None，优先读取 pyproject.toml 中的配置。
        """
        cls._backend = backend

        if timezone_name:
            cls._timezone = ZoneInfo(timezone_name)
        else:
            cls._timezone = datetime.now().astimezone().tzinfo or ZoneInfo("UTC")

        # 2. 格式字符串优先级：参数 fmt > pyproject.toml 中的 log_cli_format > None
        target_fmt = fmt or _load_pytest_log_format_from_toml()

        if backend == "stdliblog":
            cls._init_stdliblog(level, target_fmt)
        elif backend == "structlog":
            cls._init_structlog(level, target_fmt)
        elif backend == "loguru":
            cls._init_loguru(level, target_fmt)

    @classmethod
    def get_logger(cls, name: str = "rag_data") -> LoggerAdapter:
        """获取适配器实例"""
        if cls._backend == "stdliblog":
            return StdlibLogAdapter(logging.getLogger(name))
        elif cls._backend == "structlog":
            return StructlogAdapter.create(name)
        elif cls._backend == "loguru":
            return LoguruAdapter.create()

        raise ValueError(f"不支持的 Backend: {cls._backend}")

    # --- 后端配置初始化私有方法 ---

    @classmethod
    def _init_stdliblog(cls, level: str, fmt: Optional[str] = None) -> None:
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        root_logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)

        # 默认回退格式
        default_fmt = "%(asctime)s [%(levelname)s] [%(pathname)s:%(lineno)d %(funcName)s()] %(message)s"
        fmt_str = fmt if fmt else default_fmt

        formatter = StdlibTimezoneFormatter(fmt=fmt_str, datefmt="%Y-%m-%dT%H:%M:%S%z", tz=cls._timezone)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    @classmethod
    def _init_structlog(cls, level: str, fmt: Optional[str] = None) -> None:
        structlog = import_structlog()
        from structlog.processors import CallsiteParameter, CallsiteParameterAdder

        tz = cls._timezone

        def tz_timestamper(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
            event_dict["timestamp"] = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            return event_dict

        processors = [
            structlog.contextvars.merge_contextvars,
            tz_timestamper,
            # 核心：自动忽略 rag_data.logging 包下的所有适配器文件
            CallsiteParameterAdder(
                [
                    CallsiteParameter.PATHNAME,
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                ],
                additional_ignores=["rag_data.logging"],  # 👈 增加这一行！
            ),
            structlog.processors.add_log_level,
        ]

        if fmt:
            processors.append(structlog.processors.KeyValueRenderer(sort_keys=False))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))

        structlog.configure(
            processors=processors,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

    @classmethod
    def _init_loguru(cls, level: str, fmt: Optional[str] = None) -> None:
        logger = import_loguru()
        logger.remove()
        tz = cls._timezone

        def patcher(record: Dict[str, Any]) -> None:
            record["time"] = record["time"].astimezone(tz)

        # 将 stdlib 的 % 占位符转换为 Loguru 支持的 {} 占位符格式
        if fmt:
            log_format = (
                fmt.replace("%(asctime)s", "{time:YYYY-MM-DD THH:mm:ss.SSSZZ}")
                .replace("%(levelname)s", "{level}")
                .replace("%(pathname)s", "{file.path}")
                .replace("%(filename)s", "{file}")
                .replace("%(lineno)d", "{line}")
                .replace("%(funcName)s", "{function}")
                .replace("%(message)s", "{message}")
            ) + " {extra}"
        else:
            log_format = (
                "<green>{time:YYYY-MM-DD THH:mm:ss.SSSZZ}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{file.path}</cyan>:<cyan>{line}</cyan> <cyan>{function}</cyan> - "
                "<level>{message}</level> {extra}"
            )

        logger.configure(patcher=patcher)
        logger.add(sys.stdout, level=level.upper(), format=log_format)