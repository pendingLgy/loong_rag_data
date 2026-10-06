# Word 解析与切块。
#
# 解析依赖 python-docx（pyproject 的 word extra）；纯段落提取，不处理标题样式。
# 旧版 .doc 不受支持，需先转存为 .docx。
# 依赖惰性导入：解析依赖缺失抛 ParserDependencyError；句柄缺失同样抛错。

from __future__ import annotations

import os
from typing import Any, List, Optional

from rag_data.exceptions import ParseError, ParserDependencyError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DocumentParser

PACKAGES = "python-docx"
INSTALL_HINT = "pip install -e '.[word]'"
CHUNK_INSTALL_HINT = "pip install -e '.[parsers]'"


class WordParser(DocumentParser):
    """Word：纯段落抽取正文，按自然文本递归切块。"""

    SUFFIXES = frozenset({".docx"})

    def __init__(self, max_chars: int = 1000, safe_max_chars: int = 2000):
        super().__init__(max_chars, safe_max_chars)

    def parse(self, path: str, logger: Optional[LoggerAdapter] = None) -> str:
        """解析 Word：逐段提取文本，不处理标题样式，返回纯文本。"""
        if os.path.splitext(path)[1].lower() == ".doc":
            raise ParseError("暂不支持旧版 .doc，请先另存为 .docx：" + path)
        document_cls = _import_deps()
        try:
            document = document_cls(path)
            lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        except Exception as exc:  # noqa: BLE001 损坏、加密等统一转为领域异常
            raise ParseError("Word 解析失败：" + path + "：" + str(exc)) from exc
        text = "\n".join(lines)
        if logger is not None:
            logger.info("Word 解析完成", source_path=path, paragraphs=len(lines), chars=len(text))
        return text

    def chunk(
        self,
        text: str,
        nlp: Any = None,
        logger: Optional[LoggerAdapter] = None,
    ) -> List[str]:
        """切块规则：兼容井号分章（若无井号则作为整篇处理），逐段递归切分，带重叠上下文。"""
        if not text or not text.strip():
            return []
        handle = nlp
        if handle is None:
            raise ParserDependencyError(
                f"文本切块需要 spacy 及模型，请执行 {CHUNK_INSTALL_HINT}"
            )
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


def _import_deps() -> Any:
    """导入 python-docx；未安装时给出安装指引。"""
    try:
        import docx
    except ImportError as exc:
        raise ParserDependencyError("Word 解析需要 " + PACKAGES + "，请执行 " + INSTALL_HINT) from exc
    return docx.Document