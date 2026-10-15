# 解析器适配层：按扩展名把解析与切块的请求派发到对应的 DocumentParser 子类。
#
# 每种格式一个模块，各自定义一个 DocumentParser 子类：
#   SUFFIXES             该类负责的后缀
#   parse(path, logger)  文件 -> 纯文本
#   chunk(text, ...)     纯文本 -> 切块（规则各格式自持）
#
# markdown.py  MarkdownParser  .md、.markdown
# txt.py       TxtParser       .txt
# epub.py      EpubParser      .epub
# pdf.py       PdfParser       .pdf（pdfplumber，按物理坐标提取）
# word.py      WordParser      .docx（python-docx，纯段落提取）
#
# 新增格式：实现 DocumentParser 子类后加入 PARSER_CLASSES 即可；
# 后缀唯一性由 ParserAdapter 构造时的断言保证。

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from rag_data.exceptions import ParseError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DocumentParser
from rag_data.parsers.epub import EpubParser
from rag_data.parsers.markdown import MarkdownParser
from rag_data.parsers.pdf import PdfParser
from rag_data.parsers.txt import TxtParser
from rag_data.parsers.word import WordParser

# 默认参与派发的解析器类，顺序无关；每种格式一个。
PARSER_CLASSES: Tuple[Type[DocumentParser], ...] = (
    MarkdownParser,
    TxtParser,
    EpubParser,
    PdfParser,
    WordParser,
)


class ParserAdapter:
    '''解析器适配层：按扩展名派发解析与切块到对应 DocumentParser 子类。

    构造时可替换解析器类集合，默认使用内置 PARSER_CLASSES。
    '''

    def __init__(self, parser_classes: Optional[Sequence[Type[DocumentParser]]] = None) -> None:
        self._parser_classes: Tuple[Type[DocumentParser], ...] = (
            tuple(parser_classes) if parser_classes is not None else PARSER_CLASSES
        )
        self._by_suffix = self._build_index()

    @property
    def parser_classes(self) -> Tuple[Type[DocumentParser], ...]:
        '本次派发使用的解析器类集合。'
        return self._parser_classes

    def _build_index(self) -> Dict[str, Type[DocumentParser]]:
        '展开各解析器类自述的后缀，得到扩展名到解析器类的派发表。'
        index: Dict[str, Type[DocumentParser]] = {}
        for parser_cls in self._parser_classes:
            for suffix in parser_cls.SUFFIXES:
                # 后缀必须唯一归属，重复登记说明两个解析器争抢同一扩展名。
                assert suffix not in index, '后缀重复登记：' + suffix
                index[suffix] = parser_cls
        return index

    def _resolve(self, path: str) -> Type[DocumentParser]:
        '按扩展名取出负责该格式的解析器类。'
        suffix = os.path.splitext(path)[1].lower()
        parser_cls = self._by_suffix.get(suffix)
        if parser_cls is None:
            raise ParseError('暂不支持的文档格式：' + suffix)
        return parser_cls

    def resolve_parser(
        self,
        path: str,
        max_chars: Optional[int] = None,
        safe_max_chars: Optional[int] = None,
        logger: Optional[LoggerAdapter] = None,
    ) -> DocumentParser:
        '按扩展名构造该格式的解析器实例，可注入切块长度配置与日志。'
        overrides: Dict[str, Any] = {}
        if max_chars is not None:
            overrides['max_chars'] = max_chars
        if safe_max_chars is not None:
            overrides['safe_max_chars'] = safe_max_chars
        if logger is not None:
            overrides['logger'] = logger
        return self._resolve(path)(**overrides)

    def parse_document(self, path: str, logger: Optional[LoggerAdapter] = None) -> str:
        '解析单个文档并返回纯文本。'
        if not os.path.exists(path):
            raise ParseError('文件不存在：' + path)
        return self._resolve(path)(logger=logger).parse(path)

    def chunk_document(
        self,
        path: str,
        text: str,
        nlp: Any = None,
        logger: Optional[LoggerAdapter] = None,
        max_chars: Optional[int] = None,
        safe_max_chars: Optional[int] = None,
    ) -> List[str]:
        '按文件扩展名派发到该格式的 chunk，规则由各子类自持。'
        # 长度覆盖来自 parsing 分区，缺省时用解析器自身默认值。
        parser = self.resolve_parser(path, max_chars, safe_max_chars, logger=logger)
        return parser.chunk(text, nlp=nlp)

