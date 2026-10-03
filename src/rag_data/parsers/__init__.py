# 文档解析与切块：以 DocumentParser 子类承载各格式的解析与切块规则。
#
# 每种格式一个模块，各自定义一个 DocumentParser 子类：
#   SUFFIXES             该类负责的后缀
#   parse(path, logger)  文件 -> 纯文本
#   chunk(text, ...)     纯文本 -> 切块（规则各格式自持）
# markdown.py  MarkdownParser  .md、.markdown
# txt.py       TxtParser       .txt
# epub.py      EpubParser      .epub
# pdf.py       PdfParser       .pdf（pypdf，按页提取）
# word.py      WordParser      .docx（python-docx，标题映射为井号）
# 新增格式：实现 DocumentParser 子类后加入 PARSERS 即可，
# 后缀唯一性由 _build_index 的断言保证。

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from rag_data.exceptions import ParseError
from rag_data.parsers.base import DocumentParser
from rag_data.parsers.epub import EpubParser
from rag_data.parsers.markdown import MarkdownParser
from rag_data.parsers.pdf import PdfParser
from rag_data.parsers.txt import TxtParser
from rag_data.parsers.word import WordParser
from rag_data.logging.base import LoggerAdapter

__all__ = [
    "DocumentParser",
    "EpubParser",
    "MarkdownParser",
    "PdfParser",
    "TxtParser",
    "WordParser",
    "PARSERS",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "BINARY_SUFFIXES",
    "MARKDOWN_SUFFIXES",
    "TXT_SUFFIXES",
    "EPUB_SUFFIXES",
    "PDF_SUFFIXES",
    "WORD_SUFFIXES",
    "parse_document",
    "chunk_document",
]

# 参与派发的解析器实例，顺序无关；每种格式一个。
PARSERS: Tuple[DocumentParser, ...] = (
    MarkdownParser(),
    TxtParser(),
    EpubParser(),
    PdfParser(),
    WordParser(),
)

MARKDOWN_SUFFIXES = MarkdownParser.SUFFIXES
TXT_SUFFIXES = TxtParser.SUFFIXES
EPUB_SUFFIXES = EpubParser.SUFFIXES
PDF_SUFFIXES = PdfParser.SUFFIXES
WORD_SUFFIXES = WordParser.SUFFIXES

TEXT_SUFFIXES = MARKDOWN_SUFFIXES | TXT_SUFFIXES
BINARY_SUFFIXES = PDF_SUFFIXES | WORD_SUFFIXES
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EPUB_SUFFIXES | BINARY_SUFFIXES


def _build_index() -> Dict[str, DocumentParser]:
    """展开各解析器自述的后缀，得到「扩展名 -> 解析器」的派发表。"""
    index: Dict[str, DocumentParser] = {}
    for parser in PARSERS:
        for suffix in parser.SUFFIXES:
            # 后缀必须唯一归属，重复登记说明两个解析器争抢同一扩展名。
            assert suffix not in index, "后缀重复登记：" + suffix
            index[suffix] = parser
    return index


_BY_SUFFIX = _build_index()


def _resolve(path: str) -> DocumentParser:
    """按扩展名取出负责该格式的解析器。"""
    suffix = os.path.splitext(path)[1].lower()
    parser = _BY_SUFFIX.get(suffix)
    if parser is None:
        raise ParseError("暂不支持的文档格式：" + suffix)
    return parser


def parse_document(path: str, logger: Optional[LoggerAdapter] = None) -> str:
    """解析单个文档并返回纯文本。"""
    if not os.path.exists(path):
        raise ParseError("文件不存在：" + path)
    return _resolve(path).parse(path, logger=logger)


def chunk_document(
    path: str,
    text: str,
    nlp: Any = None,
    logger: Optional[LoggerAdapter] = None,
) -> List[str]:
    """按文件扩展名分派到该格式的 chunk，规则由各子类自持。"""
    return _resolve(path).chunk(text, nlp=nlp, logger=logger)
