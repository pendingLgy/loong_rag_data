# Markdown 解析与切块。
#
# 切块按章节标题（行首 #）分层，章节内复用基类的递归切分与四分之一重叠前缀。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.exceptions import ParserDependencyError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DocumentParser

CHUNK_INSTALL_HINT = "pip install -e '.[parsers]'"


class MarkdownParser(DocumentParser):
    """Markdown：按行首 # 分成章节后各自递归切分。"""

    SUFFIXES = frozenset({".md", ".markdown"})

    def __init__(self, max_chars: int = 1000, safe_max_chars: int = 2000, logger: Optional[LoggerAdapter] = None):
        super().__init__(max_chars, safe_max_chars, logger)

    def parse(self, path: str) -> str:
        """读取 Markdown 原文；标题、强调、链接等标记保留。"""
        # TODO: 如需纯语义文本，可在此剥离 Markdown 标记后再返回。
        return self._read_text(path)

    def chunk(
        self,
        text: str,
        nlp: Any = None,
    ) -> List[str]:
        """切块规则：按 # 切出章节，逐章递归切分，块间带上文四分之一重叠。"""
        if not text or not text.strip():
            return []
        handle = nlp
        if handle is None:
            raise ParserDependencyError(
                f"文本切块需要 spacy 及模型，请执行 {CHUNK_INSTALL_HINT}")

        raw_sections = text.split("#")
        all_chunks: List[str] = []

        for section in raw_sections:
            overlap_prefix = all_chunks[-1] if all_chunks else ""
            if not section.strip():
                continue
            current_overlap_prefix = self._extract_overlap_prefix([overlap_prefix])
            section_chunks = self._process_text_recursive(
                text=section, overlap_prefix=current_overlap_prefix, nlp=handle
            )
            all_chunks.extend(section_chunks)

        return all_chunks

