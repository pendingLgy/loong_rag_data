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

from rag_data.config import DBSettings, EmbeddingSettings, LoggingSettings, ParsingSettings, Settings, VectorSettings
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import LoggerFactory
from rag_data.parsers import ParserHandler
from rag_data.parsers.base import load_nlp
from rag_data.storage.db import BaseDB
from rag_data.storage.db.db_engine import build_database
from rag_data.storage.vector_store.vector_engine import VectorEngine, build_vector_engine


# ---------------- 分流程的装配方法 ----------------


def _build_logger(settings: LoggingSettings) -> Optional[LoggerAdapter]:
    """流程二：装配日志适配层；logging.enable 为 False 时返回 None，不初始化。"""
    if not settings.enable:
        return None
    logging_settings = settings
    LoggerFactory.setup(
        backend=logging_settings.backend,
        timezone_name=logging_settings.timezone or None,
        level=logging_settings.level,
        fmt=logging_settings.format or None,
    )
    return LoggerFactory.get_logger()


def _build_nlp(settings: ParsingSettings, logger: Optional[LoggerAdapter]) -> Optional[Any]:
    """流程三：加载 spaCy 句柄；parsing.enable 为 False 或模型不可用时返回 None。"""
    if not settings.enable:
        return None
    # 加载与缓存见 rag_data.parsers.base；模型名来自 parsing 分区。
    return load_nlp(settings.spacy_model, logger)


def _build_embedder(settings: EmbeddingSettings, logger: Optional[LoggerAdapter] = None, model: Optional[Any] = None) -> Optional[EmbedderHandler]:
    """流程四：装配向量化组件；embedding.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    # 未注入 model 时按配置装配 provider，注入时直接使用注入对象。
    return EmbedderHandler(settings, logger, model=model)



def _build_db(settings: DBSettings, logger: Optional[LoggerAdapter]) -> Optional[BaseDB]:
    """流程五：按 db 分区装配数据库；db.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return build_database(settings, logger)



def _build_vector(settings: VectorSettings, logger: Optional[LoggerAdapter]) -> Optional[VectorEngine]:
    '''流程六：按 vector 分区装配向量引擎；vector.enable 为 False 时返回 None。'''
    if not settings.enable:
        return None
    return build_vector_engine(settings, logger)

# ---------------- 一站式门面 ----------------


class RagData:
    """一站式门面：一次装配配置与各分区组件（日志、句柄、向量化、关系库、向量库）及解析器。

    构造时只需传入 settings；每个分区由自己的 enable 开关控制，为 False 时对应实例为 None，不初始化。解析器固定为 ParserHandler 实例。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # 用 parsing 分区的长度预置解析器实例，使常用路径直接复用缓存实例。
        self.parser = ParserHandler(
            default_max_chars=settings.parsing.max_chars,
            default_safe_max_chars=settings.parsing.safe_max_chars,
        )
        self.logger = _build_logger(settings.logging)
        self.nlp = _build_nlp(settings.parsing, self.logger)
        self.embedder = _build_embedder(settings.embedding, self.logger)
        self.db = _build_db(settings.db, self.logger)
        self.vector = _build_vector(settings.vector, self.logger)

    # ---------------- 流程方法 ----------------

    @property
    def parser_handler(self) -> Any:
        """解析派发入口：返回 ParserHandler 实例。"""
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
