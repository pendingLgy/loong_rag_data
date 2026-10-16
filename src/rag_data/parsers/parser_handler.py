# 解析器处理器：按扩展名把解析与切块的请求派发到对应的 DocumentParser 子类。
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
# 后缀唯一性由 ParserHandler 构造时的断言保证。
# 实例化策略：ParserHandler 构造时一次性建好全部解析器实例并复用，减少重复创建。

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from rag_data.exceptions import ParseError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DocumentParser
from rag_data.parsers.registry_parser import parser_classes

# 默认参与派发的解析器类，取自注册表；用户注册的自定义解析器也会被纳入。
PARSER_CLASSES: Tuple[Type[DocumentParser], ...] = tuple(parser_classes())


class ParserHandler:
    """解析器处理器：按扩展名把解析与切块派发到对应 DocumentParser 子类。

    构造时一次性实例化全部解析器并缓存，后续调用直接复用，避免每次使用都新建对象；
    仅当调用方显式传入与缓存实例不同的切块长度时才临时新建实例。
    """

    def __init__(
        self,
        parser_classes: Optional[Sequence[Type[DocumentParser]]] = None,
        default_max_chars: Optional[int] = None,
        default_safe_max_chars: Optional[int] = None,
    ) -> None:
        self._parser_classes: Tuple[Type[DocumentParser], ...] = (
            tuple(parser_classes) if parser_classes is not None else PARSER_CLASSES
        )
        self._by_suffix = self._build_index()
        # 预实例化的共享实例：解析器类 -> 实例。
        self._instances: Dict[Type[DocumentParser], DocumentParser] = self._build_instances(
            default_max_chars, default_safe_max_chars
        )


    # ---------------- 属性 ----------------

    @property
    def parser_classes(self) -> Tuple[Type[DocumentParser], ...]:
        """本次派发使用的解析器类集合。"""
        return self._parser_classes

    @property
    def instances(self) -> Dict[Type[DocumentParser], DocumentParser]:
        """已预实例化的解析器副本，便于外部查看或替换。"""
        return dict(self._instances)

    def get_instance(self, path: str) -> DocumentParser:
        """按扩展名取缓存的解析器实例，不新建对象。"""
        return self._instances[self._resolve(path)]

    # ---------------- 索引与实例 ----------------

    def _build_index(self) -> Dict[str, Type[DocumentParser]]:
        """展开各解析器类自述的后缀，得到扩展名到解析器类的派发表。"""
        index: Dict[str, Type[DocumentParser]] = {}
        for parser_cls in self._parser_classes:
            for suffix in parser_cls.SUFFIXES:
                # 后缀必须唯一归属，重复登记说明两个解析器争抢同一扩展名。
                assert suffix not in index, '后缀重复登记：' + suffix
                index[suffix] = parser_cls
        return index

    def _build_instances(self, max_chars: Optional[int], safe_max_chars: Optional[int]) -> Dict[Type[DocumentParser], DocumentParser]:
        """为每个解析器类预建一个实例；未显式给出的长度沿用各类自身默认值。"""
        overrides: Dict[str, Any] = {}
        if max_chars is not None:
            overrides['max_chars'] = max_chars
        if safe_max_chars is not None:
            overrides['safe_max_chars'] = safe_max_chars
        return {parser_cls: parser_cls(**overrides) for parser_cls in self._parser_classes}


    def _resolve(self, path: str) -> Type[DocumentParser]:
        """按扩展名取出负责该格式的解析器类。"""
        suffix = os.path.splitext(path)[1].lower()
        parser_cls = self._by_suffix.get(suffix)
        if parser_cls is None:
            raise ParseError('暂不支持的文档格式：' + suffix)
        return parser_cls

    @staticmethod
    def _needs_custom(parser: DocumentParser, max_chars: Optional[int], safe_max_chars: Optional[int]) -> bool:
        """判断调用方传入的长度是否与缓存实例不一致；未传入视为一致。"""
        if max_chars is not None and parser.max_chars != max_chars:
            return True
        if safe_max_chars is not None and parser.safe_max_chars != safe_max_chars:
            return True
        return False


    # ---------------- 对外流程 ----------------

    def resolve_parser(
        self,
        path: str,
        max_chars: Optional[int] = None,
        safe_max_chars: Optional[int] = None,
        logger: Optional[LoggerAdapter] = None,
    ) -> DocumentParser:
        """按扩展名取解析器实例；配置与缓存一致时直接复用，否则临时新建。"""
        parser = self.get_instance(path)
        # logger 按调用时绑定，赋值开销极低，不触发实例重建。
        if logger is not None:
            parser.logger = logger
        if not self._needs_custom(parser, max_chars, safe_max_chars):
            return parser
        overrides: Dict[str, Any] = {}
        if max_chars is not None:
            overrides['max_chars'] = max_chars
        if safe_max_chars is not None:
            overrides['safe_max_chars'] = safe_max_chars
        if logger is not None:
            overrides['logger'] = logger
        return type(parser)(**overrides)


    def parse_document(self, path: str, logger: Optional[LoggerAdapter] = None) -> str:
        """解析单个文档并返回纯文本。"""
        if not os.path.exists(path):
            raise ParseError('文件不存在：' + path)
        return self.resolve_parser(path, logger=logger).parse(path)

    def chunk_document(
        self,
        path: str,
        text: str,
        nlp: Any = None,
        logger: Optional[LoggerAdapter] = None,
        max_chars: Optional[int] = None,
        safe_max_chars: Optional[int] = None,
    ) -> List[str]:
        """按文件扩展名派发到该格式的 chunk，规则由各子类自持。"""
        # 长度覆盖来自 parsing 分区，缺省时用解析器自身默认值。
        parser = self.resolve_parser(path, max_chars, safe_max_chars, logger=logger)
        return parser.chunk(text, nlp=nlp)

