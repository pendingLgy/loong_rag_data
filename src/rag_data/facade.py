# 流程门面：集中装配各流程所需的组件。
#
# 两个公共类：
#   RagData       解析与切块门面：RagSetting（parsing、embedding、logging）
#   MemoryServer  记忆服务：MemorySettings（llm、db、vector）独立装配，并持有 MemoryHandler
#
#   from rag_data import RagData, MemoryServer
#
#   rag = RagData(rag_setting)
#   mem = MemoryServer(memory_setting)
#
# 组件未注入时按各自分区的 enable 开关装配，关闭者保持 None。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.config import (
    DBSettings,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    MemorySettings,
    ParsingSettings,
    RagSetting,
    VectorSettings,
)
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ConfigError
from rag_data.llm.llm_handler import LLMHandler
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import LoggerFactory
from rag_data.memory.handler import MemoryHandler
from rag_data.memory.models import AddResult, DeleteResult, SearchResult, UpdateResult
from rag_data.parsers import ParserHandler
from rag_data.nlp.spacy import load_nlp
from rag_data.storage.db import BaseDB
from rag_data.storage.db.db_engine import build_database
from rag_data.storage.vector_store.vector_engine import VectorEngine, build_vector_engine



# ---------------- 分流程序的装配方法 ----------------


def _build_logger(settings: LoggingSettings) -> Optional[LoggerAdapter]:
    """装配日志适配层；logging.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    LoggerFactory.setup(
        backend=settings.backend,
        timezone_name=settings.timezone or None,
        level=settings.level,
        fmt=settings.format or None,
    )
    return LoggerFactory.get_logger()


def _build_nlp(settings: ParsingSettings, logger: Optional[LoggerAdapter]) -> Optional[Any]:
    """加载 spaCy 句柄；parsing.enable 为 False 或模型不可用时返回 None。"""
    if not settings.enable:
        return None
    return load_nlp(settings.spacy_model, logger)


def _build_embedder(
    settings: EmbeddingSettings,
    logger: Optional[LoggerAdapter] = None,
    model: Optional[Any] = None,
) -> Optional[EmbedderHandler]:
    """装配向量化组件；embedding.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return EmbedderHandler(settings, logger, model=model)


def _build_llm(
    settings: LLMSettings,
    logger: Optional[LoggerAdapter],
    model: Optional[Any] = None,
) -> Optional[LLMHandler]:
    """装配大模型调用入口；llm.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return LLMHandler(settings, logger, model=model)



def _build_db(settings: DBSettings, logger: Optional[LoggerAdapter]) -> Optional[BaseDB]:
    """按 db 分区装配关系库；db.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return build_database(settings, logger)


def _build_vector(
    settings: VectorSettings,
    logger: Optional[LoggerAdapter],
) -> Optional[VectorEngine]:
    """按 vector 分区装配向量引擎；vector.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return build_vector_engine(settings, logger)



def _build_memory(
    settings: MemorySettings,
    logger: Optional[LoggerAdapter],
    embedder: Optional[EmbedderHandler],
    llm: Optional[LLMHandler],
    db: Optional[BaseDB],
    vector: Optional[VectorEngine],
) -> Optional[MemoryHandler]:
    """按 memory 分区装配记忆层；memory.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return MemoryHandler(settings, logger=logger, embedder=embedder, llm=llm, db=db, vector=vector)


# ---------------- 记忆服务 ----------------



class MemoryServer:
    """记忆服务：接收日志、向量化、关系库、向量库与大模型五个组件；

    未注入者按 settings 各分区自行装配，随后构造 MemoryHandler 并代理四条流程。
    """

    def __init__(
        self,
        settings: MemorySettings,
        logger: Optional[LoggerAdapter] = None,
        embedder: Optional[EmbedderHandler] = None,
        db: Optional[BaseDB] = None,
        vector: Optional[VectorEngine] = None,
        llm: Optional[LLMHandler] = None,
    ) -> None:
        self.settings = settings
        # 五个组件都可注入；未注入时按 memory 分区的子配置装配，关闭者保持 None。
        self.logger = logger if logger is not None else LoggerFactory.get_logger()
        self.embedder = embedder
        self.db = db if db is not None else _build_db(settings.db, self.logger)
        self.vector = vector if vector is not None else _build_vector(settings.vector, self.logger)
        self.llm = llm if llm is not None else _build_llm(settings.llm, self.logger)
        self.handler = _build_memory(
            settings, self.logger, self.embedder, self.llm, self.db, self.vector
        )

    # ---------------- 四条流程代理 ----------------

    @property
    def memory(self) -> Optional[MemoryHandler]:
        """记忆处理器实例；memory.enable 为 False 时为 None。"""
        return self.handler



    def add(self, messages: Any, **kwargs: Any) -> AddResult:
        """写入记忆；未启用记忆层时抛 ConfigError。"""
        return self._require().add(messages, **kwargs)

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        """检索记忆。"""
        return self._require().search(query, **kwargs)

    def update(self, memory_id: int, **kwargs: Any) -> UpdateResult:
        """更新记忆。"""
        return self._require().update(memory_id, **kwargs)

    def delete(self, memory_id: int) -> DeleteResult:
        """删除记忆。"""
        return self._require().delete(memory_id)

    def _require(self) -> MemoryHandler:
        """取处理器；未启用时给出明确的配置错误而非空指针。"""
        if self.handler is None:
            raise ConfigError("记忆层未启用：请设置 memory.enable 为 True")
        return self.handler


# ---------------- 解析门面 ----------------


class RagData:
    """解析门面：一次装配配置、日志、句柄与向量化组件，持有解析派发入口。

    构造时只需传入 settings；各分区由自己的 enable 开关控制，为 False 时对应实例为 None。
    解析器固定为 ParserHandler 实例；记忆、关系库、向量库与大模型统一由 MemoryServer 承担。
    """

    def __init__(self, settings: RagSetting) -> None:
        self.settings = settings
        # 用 parsing 分区的长度预置解析器实例，使常用路径直接复用缓存实例。
        self.parser = ParserHandler(
            default_max_chars=settings.parsing.max_chars,
            default_safe_max_chars=settings.parsing.safe_max_chars,
        )
        self.logger = _build_logger(settings.logging)
        self.nlp = _build_nlp(settings.parsing, self.logger)
        self.embedder = _build_embedder(settings.embedding, self.logger)

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
            path,
            text,
            nlp=self.nlp,
            logger=self.logger,
            max_chars=parsing.max_chars,
            safe_max_chars=parsing.safe_max_chars,
        )


__all__ = [
    "RagData",
    "MemoryServer",
]

