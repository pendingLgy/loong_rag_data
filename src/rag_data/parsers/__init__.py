# 文档解析与切块子包：对外统一出口。
#
# 派发逻辑在 parser_adapter 的 ParserAdapter；本文件只做导出，不含业务逻辑。
#
#   from rag_data.parsers import parse_document, chunk_document, resolve_parser

from rag_data.parsers.base import DocumentParser
from rag_data.parsers.parser_adapter import PARSER_CLASSES, ParserAdapter

# 默认适配器实例；模块级函数即其绑定方法，便于直接调用。
_adapter = ParserAdapter()
resolve_parser = _adapter.resolve_parser
parse_document = _adapter.parse_document
chunk_document = _adapter.chunk_document

__all__ = [
    "DocumentParser",
    "ParserAdapter",
    "PARSER_CLASSES",
    "resolve_parser",
    "parse_document",
    "chunk_document",
]

