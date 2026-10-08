# 文档解析与切块：以 DocumentParser 子类承载各格式的解析与切块规则。
#
# 每种格式一个模块，各自定义一个 DocumentParser 子类：
#   SUFFIXES             该类负责的后缀
#   parse(path, logger)  文件 -> 纯文本
#   chunk(text, ...)     纯文本 -> 切块（规则各格式自持）
# markdown.py  MarkdownParser  .md、.markdown
# txt.py       TxtParser       .txt
# epub.py      EpubParser      .epub
# pdf.py       PdfParser       .pdf（pdfplumber，按物理坐标提取）
# word.py      WordParser      .docx（python-docx，纯段落提取，不处理标题样式）
# 新增格式：实现 DocumentParser 子类后加入 PARSER_CLASSES 即可，
# 后缀唯一性由 _build_index 的断言保证。

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Type

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
    "PARSER_CLASSES",
    "PARSERS",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "BINARY_SUFFIXES",
    "MARKDOWN_SUFFIXES",
    "TXT_SUFFIXES",
    "EPUB_SUFFIXES",
    "PDF_SUFFIXES",
    "WORD_SUFFIXES",
    "resolve_parser",
    "parse_document",
    "chunk_document",
]

# 参与派发的解析器类，顺序无关；每种格式一个。
PARSER_CLASSES: Tuple[Type[DocumentParser], ...] = (
    MarkdownParser,
    TxtParser,
    EpubParser,
    PdfParser,
    WordParser,
)

PARSERS: Tuple[DocumentParser, ...] = tuple(cls() for cls in PARSER_CLASSES)

MARKDOWN_SUFFIXES = MarkdownParser.SUFFIXES
TXT_SUFFIXES = TxtParser.SUFFIXES
EPUB_SUFFIXES = EpubParser.SUFFIXES
PDF_SUFFIXES = PdfParser.SUFFIXES
WORD_SUFFIXES = WordParser.SUFFIXES

TEXT_SUFFIXES = MARKDOWN_SUFFIXES | TXT_SUFFIXES
BINARY_SUFFIXES = PDF_SUFFIXES | WORD_SUFFIXES
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EPUB_SUFFIXES | BINARY_SUFFIXES


def _build_index() -> Dict[str, Type[DocumentParser]]:
    """展开各解析器类自述的后缀，得到「扩展名 -> 解析器类」的派发表。"""
    index: Dict[str, Type[DocumentParser]] = {}
    for parser_cls in PARSER_CLASSES:
        for suffix in parser_cls.SUFFIXES:
            # 后缀必须唯一归属，重复登记说明两个解析器争抢同一扩展名。
            assert suffix not in index, "后缀重复登记：" + suffix
            index[suffix] = parser_cls
    return index


_BY_SUFFIX = _build_index()


def _resolve(path: str) -> Type[DocumentParser]:
    """按扩展名取出负责该格式的解析器类。"""
    suffix = os.path.splitext(path)[1].lower()
    parser_cls = _BY_SUFFIX.get(suffix)
    if parser_cls is None:
        raise ParseError("暂不支持的文档格式：" + suffix)
    return parser_cls


def resolve_parser(
    path: str,
    max_chars: Optional[int] = None,
    safe_max_chars: Optional[int] = None,
    logger = None
) -> DocumentParser:
    """按扩展名构造该格式的解析器实例，可注入切块长度配置。"""
    overrides: Dict[str, Any] = {}
    if max_chars is not None:
        overrides["max_chars"] = max_chars
    if safe_max_chars is not None:
        overrides["safe_max_chars"] = safe_max_chars
    if logger is not None:
        overrides["logger"] = logger
    return _resolve(path)(**overrides)


def parse_document(path: str, logger: Optional[LoggerAdapter] = None) -> str:
    """解析单个文档并返回纯文本。"""
    if not os.path.exists(path):
        raise ParseError("文件不存在：" + path)
    return _resolve(path)(logger=logger).parse(path)


def chunk_document(
    path: str,
    text: str,
    nlp: Any = None,
    logger: Optional[LoggerAdapter] = None,
    max_chars: Optional[int] = None,
    safe_max_chars: Optional[int] = None,
) -> List[str]:
    """按文件扩展名分派到该格式的 chunk，规则由各子类自持。"""
    # 长度覆盖来自 parsing 分区，缺省时用解析器自身默认值。
    parser = resolve_parser(path, max_chars, safe_max_chars, logger=logger)
    return parser.chunk(text, nlp=nlp)
