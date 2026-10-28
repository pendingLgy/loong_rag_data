# 解析器注册表：登记 DocumentParser 实现类，只负责注册与解析，不做实例化。
#
# 注册方式二选一：继承 DocumentParser 并声明 backend 与 SUFFIXES（自动注册），
# 或调用 register_parser 显式登记。注册值可为类对象，也可为点分路径（首次解析时惰性导入）。
# 实例化在 ParserHandler 构造与派发时进行，本模块不持有任何实例。

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

from rag_data.core.registry import Ref, Registry

if TYPE_CHECKING:  # pragma: no cover
    from rag_data.parsers.base import DocumentParser

# 内置解析器名称到类路径的映射，按需惰性导入。
BUILTIN_PARSERS: Dict[str, str] = {
    "markdown": "rag_data.parsers.markdown.MarkdownParser",
    "txt": "rag_data.parsers.txt.TxtParser",
    "epub": "rag_data.parsers.epub.EpubParser",
    "pdf": "rag_data.parsers.pdf.PdfParser",
    "word": "rag_data.parsers.word.WordParser",
}

_REGISTRY = Registry("文档解析器", BUILTIN_PARSERS)



def register_parser(name: str, ref: Ref) -> None:
    """登记文档解析器：ref 可为类对象，也可为点分路径字符串。"""
    _REGISTRY.register(name, ref)


def register_parser_provider(name: str):
    """装饰器形式登记文档解析器。"""
    return _REGISTRY.decorator(name)


def is_parser_registered(name: str) -> bool:
    """判断文档解析器是否已登记。"""
    return name in _REGISTRY


def available_parsers() -> List[str]:
    """返回已登记的解析器名，按字典序排列。"""
    return _REGISTRY.names()


def resolve_parser_class(name: str) -> Type[DocumentParser]:
    """按名取解析器实现类，路径登记的在此惰性导入。"""
    return _REGISTRY.resolve(name)


def parser_classes() -> List[Type[DocumentParser]]:
    """返回全部已登记解析器类，顺序与 BUILTIN_PARSERS 的字典序一致。"""
    return [resolve_parser_class(name) for name in available_parsers()]

