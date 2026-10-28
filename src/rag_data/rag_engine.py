from __future__ import annotations

from typing import Any, List, Optional

from rag_data.core.config import (
    EmbeddingSettings,
    LLMSettings,
    ParsingSettings,
    RagSetting,
)
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.llm.llm_handler import LLMHandler
from rag_data.nlp.spacy import load_nlp
from rag_data.parsers import ParserHandler


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



class RagEngine:
    """解析门面：一次装配配置、句柄与向量化组件，持有解析派发入口。
    构造时只需传入 settings；各分区由自己的 enable 开关控制，为 False 时对应实例为 None。
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
    "RagEngine",
]
