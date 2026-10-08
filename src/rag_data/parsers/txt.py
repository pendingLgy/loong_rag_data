# 纯文本解析与切块。
#
# 切块复用基类的递归切分与四分之一重叠前缀，规则由本类自持。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.exceptions import ParserDependencyError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DocumentParser

CHUNK_INSTALL_HINT = "pip install -e '.[parsers]'"


class TxtParser(DocumentParser):
    """纯文本：无结构，整篇交给基类递归切分。"""

    SUFFIXES = frozenset({".txt"})

    def __init__(self, max_chars: int = 1000, safe_max_chars: int = 2000, logger: Optional[LoggerAdapter] = None):
        super().__init__(max_chars, safe_max_chars, logger)

    def parse(self, path: str) -> str:
        """读取纯文本原文。"""
        return self._read_text(path)

    def chunk(
        self,
        text: str,
        nlp: Any = None
    ) -> List[str]:
        """切块规则：无结构，整篇作为一个段落递归切分。"""
        if not text or not text.strip():
            return []
        handle = nlp
        if handle is None:
            raise ParserDependencyError(
                f"文本切块需要 spacy 及模型，请执行 {CHUNK_INSTALL_HINT}")
        return self._process_text_recursive(text=text, overlap_prefix="", nlp=handle)

