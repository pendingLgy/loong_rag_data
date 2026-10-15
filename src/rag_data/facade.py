# 流程门面（Facade）：集中装配各流程所需的组件。
#
# 调用方只需要引入本模块或包根公共入口，无需再逐个导入 config、
# embedding、parsers、logging 等子模块：
#
#   from rag_data import RagData
#
#   app = RagData(settings)          # 装配配置、日志、句柄与向量化
#
# 需自定义依赖时可直接构造：RagData(settings, parser=...)，其余组件由 settings 派生。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.config import EmbeddingSettings, LoggingSettings, ParsingSettings, Settings
from rag_data.embedding.embedder import Embedder
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import LoggerFactory
from rag_data.parsers import ParserAdapter
from rag_data.parsers.base import load_nlp


# ---------------- 分流程的装配方法 ----------------


def _build_logger(settings: LoggingSettings) -> LoggerAdapter:
    """流程二：装配日志适配层，后端、级别、格式与时区均取自 logging 分区。"""
    logging_settings = settings
    LoggerFactory.setup(
        backend=logging_settings.backend,
        timezone_name=logging_settings.timezone or None,
        level=logging_settings.level,
        fmt=logging_settings.format or None,
    )
    return LoggerFactory.get_logger()


def _build_nlp(settings: ParsingSettings, logger: LoggerAdapter) -> Optional[Any]:
    """流程三：加载 spaCy 句柄；不可用时返回 None，由解析器按自身规则处理。"""
    # 加载与缓存见 rag_data.parsers.base；模型名来自 parsing 分区。
    return load_nlp(settings.spacy_model, logger)


def _build_embedder(settings: EmbeddingSettings, logger: LoggerAdapter, model: Optional[Any] = None) -> Embedder:
    """流程四：装配向量化组件，provider 由 embedding.provider 决定。"""
    # 未注入 model 时按配置装配 provider，注入时直接使用注入对象。
    return Embedder(settings, logger, model=model)


# ---------------- 一站式门面 ----------------


class RagData:
    """一站式门面：一次装配配置、日志、句柄与向量化组件。

    构造时只需传入 settings，logger、nlp 与 embedder 均由其派生为实例变量；
    parser 为解析派发入口，默认 ParserAdapter 实例。
    """

    def __init__(self, settings: Settings, parser: Optional[Any] = None) -> None:
        self.settings = settings
        self.parser = parser if parser is not None else ParserAdapter()
        self.logger = _build_logger(settings.logging)
        self.nlp = _build_nlp(settings.parsing, self.logger)
        self.embedder = _build_embedder(settings.embedding, self.logger)

    # ---------------- 流程方法 ----------------

    @property
    def parser_adapter(self) -> Any:
        """解析派发入口：返回 ParserAdapter 实例。"""
        return self.parser

    def parse_document(self, path: str) -> str:
        """流程：解析单个文件为纯文本。"""
        return self.parser.parse_document(path, logger=self.logger)

    def chunk_document(self, path: str, text: str) -> List[str]:
        """流程：按文件扩展名切块，规则由解析器自持，长度取自 parsing 分区。"""
        parsing = self.settings.parsing
        return self.parser.chunk_document(
            path, text, nlp=self.nlp, logger=self.logger,
            max_chars=parsing.max_chars, safe_max_chars=parsing.safe_max_chars,
        )


__all__ = [
    "RagData",
]
