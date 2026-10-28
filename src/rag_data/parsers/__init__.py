# 文档解析与切块子包：对外统一出口。
#
# 派发逻辑在 parser_handler 的 ParserHandler；本文件只做导出，不含业务逻辑。
#
#   from rag_data.parsers import parse_document, chunk_document, resolve_parser

from rag_data.parsers.base import DocumentParser
from rag_data.parsers.parser_handler import PARSER_CLASSES, ParserHandler
from rag_data.parsers.registry_parser import (
    BUILTIN_PARSERS,
    available_parsers,
    is_parser_registered,
    register_parser,
    register_parser_provider,
    resolve_parser_class,
)

# 默认处理器实例；模块级函数即其绑定方法，便于直接调用。
_handler = ParserHandler()
resolve_parser = _handler.resolve_parser
parse_document = _handler.parse_document
chunk_document = _handler.chunk_document

__all__ = [
    "DocumentParser",
    "ParserHandler",
    "PARSER_CLASSES",
    "resolve_parser",
    "parse_document",
    "chunk_document",
    "register_parser",
    "register_parser_provider",
    "available_parsers",
    "is_parser_registered",
    "resolve_parser_class",
    "BUILTIN_PARSERS",
]

