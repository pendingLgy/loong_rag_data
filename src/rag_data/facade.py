# 流程门面：集中装配各流程所需的组件。
#
# 两个门面类：
#   RagData       解析与切块门面：RagSetting（parsing、embedding）
#   MemoryServer  记忆服务：MemorySettings（embedding、llm、db、vector）独立装配，并持有 MemoryHandler
#
#   from rag_data import RagData, MemoryServer
#
#   rag = RagData(rag_setting)
#   mem = MemoryServer(memory_setting)
#
# 各组件按各自分区的 enable 开关装配，为 False 时对应实例为 None；日志统一用标准库，由组件按模块名自取。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.config import (
    DBSettings,
    EmbeddingSettings,
    LLMSettings,
    MemorySettings,
    ParsingSettings,
    RagSetting,
    VectorSettings,
)
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ConfigError
from rag_data.llm.llm_handler import LLMHandler
from rag_data.memory.handler import MemoryHandler
from rag_data.memory.models import AddResult, DeleteResult, SearchResult, UpdateResult
from rag_data.parsers import ParserHandler
from rag_data.nlp.spacy import load_nlp
from rag_data.storage.db import BaseDB
from rag_data.storage.db.db_engine import build_database
from rag_data.storage.vector_store.vector_engine import VectorEngine, build_vector_engine


def _build_nlp(settings: ParsingSettings) -> Optional[Any]:
    """加载 spaCy 句柄；parsing.enable 为 False 或模型不可用时返回 None。"""
    if not settings.enable:
        return None
    return load_nlp(settings.spacy_model)


def _build_embedder(
    settings: EmbeddingSettings,
    model: Optional[Any] = None,
) -> Optional[EmbedderHandler]:
    """装配向量化组件；embedding.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return EmbedderHandler(settings, model=model)


def _build_llm(
    settings: LLMSettings,
    model: Optional[Any] = None,
) -> Optional[LLMHandler]:
    """装配大模型调用入口；llm.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return LLMHandler(settings, model=model)


def _build_db(settings: DBSettings) -> Optional[BaseDB]:
    """按 db 分区装配关系库；db.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return build_database(settings)


def _build_vector(settings: VectorSettings) -> Optional[VectorEngine]:
    """按 vector 分区装配向量引擎；vector.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return build_vector_engine(settings)


def _build_memory(
    settings: MemorySettings,
    embedder: Optional[EmbedderHandler],
    llm: Optional[LLMHandler],
    db: Optional[BaseDB],
    vector: Optional[VectorEngine],
) -> Optional[MemoryHandler]:
    """按 memory 分区装配记忆层；memory.enable 为 False 时返回 None。"""
    if not settings.enable:
        return None
    return MemoryHandler(settings, embedder=embedder, llm=llm, db=db, vector=vector)


# ---------------- 记忆服务 ----------------


class MemoryServer:
    """记忆服务门面：接收 MemorySettings，内部按各分区装配嵌入器、大模型、关系库与向量库。

    四条记忆流程（add/search/update/delete）转发给 MemoryHandler；memory.enable 为 False
    时 handler 为 None，四条流程统一抛 ConfigError。
    """

    def __init__(
        self,
        settings: MemorySettings,
    ) -> None:
        self.settings = settings
        self.embedder = _build_embedder(settings.embedding)
        self.llm = _build_llm(settings.llm)
        self.db = _build_db(settings.db)
        self.vector = _build_vector(settings.vector)
        self.handler = _build_memory(
            settings, self.embedder, self.llm, self.db, self.vector
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
    """解析门面：一次装配配置、句柄与向量化组件，持有解析派发入口。

    构造时只需传入 settings；各分区由自己的 enable 开关控制，为 False 时对应实例为 None。
    解析器固定为 ParserHandler 实例；记忆、关系库、向量库与大模型统一由 MemoryServer 承担。
    """

    def __init__(
        self,
        settings: RagSetting,
    ) -> None:
        self.settings = settings
        # 用 parsing 分区的长度预置解析器实例，使常用路径直接复用缓存实例。
        self.parser = ParserHandler(
            default_max_chars=settings.parsing.max_chars,
            default_safe_max_chars=settings.parsing.safe_max_chars,
        )
        self.nlp = _build_nlp(settings.parsing)
        self.embedder = _build_embedder(settings.embedding)

    # ---------------- 流程方法 ----------------

    @property
    def parser_handler(self) -> Any:
        """解析派发入口：返回 ParserHandler 实例。"""
        return self.parser

    def parse_document(self, path: str) -> str:
        """流程：解析单个文件为纯文本。"""
        return self.parser.parse_document(path)

    def chunk_document(self, path: str, text: str) -> List[str]:
        """流程：按文件扩展名切块，规则由解析器自持，长度取自 parsing 分区。"""
        parsing = self.settings.parsing
        return self.parser.chunk_document(
            path,
            text,
            nlp=self.nlp,
            max_chars=parsing.max_chars,
            safe_max_chars=parsing.safe_max_chars,
        )


__all__ = [
    "RagData",
    "MemoryServer",
]
