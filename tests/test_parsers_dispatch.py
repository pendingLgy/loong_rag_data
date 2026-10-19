# parsers 派发测试：按后缀构造解析器并注入切块长度配置。

import pytest

from rag_data.exceptions import ParseError
from rag_data.parsers import (
    PARSER_CLASSES,
    chunk_document,
    resolve_parser,
)
from rag_data.parsers.base import DocumentParser
from rag_data.parsers.txt import TxtParser


class _Sent:
    def __init__(self, text):
        self.text = text


class _Doc:
    def __init__(self, sentences):
        self.sents = [_Sent(sentence) for sentence in sentences]


def _slice_handle(size):
    # 按固定长度切片，模拟可递归续切的句柄。
    def _call(text):
        return _Doc([text[i:i + size] for i in range(0, len(text), size)])

    return _call


def _explode(text):
    raise AssertionError('短文本不应调用句柄')


# ---------------- resolve_parser ----------------


def test_resolve_parser_returns_instance():
    assert isinstance(resolve_parser('a.txt'), TxtParser)


def test_resolve_parser_injects_limits():
    parser = resolve_parser('a.txt', max_chars=7, safe_max_chars=11)
    assert parser.max_chars == 7
    assert parser.safe_max_chars == 11


def test_resolve_parser_keeps_defaults():
    parser = resolve_parser('a.txt')
    assert parser.max_chars == 1000
    assert parser.safe_max_chars == 2000


def test_resolve_parser_rejects_unknown_suffix():
    with pytest.raises(ParseError):
        resolve_parser('a.xyz')


# ---------------- chunk_document ----------------


def test_chunk_document_injects_limits():
    result = chunk_document('a.txt', 'abcdefghijkl', nlp=_slice_handle(3), max_chars=6)
    assert result == ['abcdef', 'def\nghijkl']


def test_chunk_document_default_single_chunk():
    assert chunk_document('a.txt', 'short', nlp=_explode) == ['short']


def test_chunk_document_rejects_unknown_suffix():
    with pytest.raises(ParseError):
        chunk_document('a.xyz', 'text')

# ---------------- 预实例化与复用 ----------------


def test_handler_pre_initializes_all_parsers():
    from rag_data.parsers.parser_handler import PARSER_CLASSES, ParserHandler

    handler = ParserHandler()
    instances = handler.instances
    assert set(instances) == set(PARSER_CLASSES)
    assert all(isinstance(instance, parser_cls) for parser_cls, instance in instances.items())
    assert len(instances) == len(PARSER_CLASSES)


def test_handler_reuses_cached_instance():
    from rag_data.parsers.parser_handler import ParserHandler

    handler = ParserHandler()
    first = handler.resolve_parser('a.txt')
    second = handler.resolve_parser('b.txt')
    assert first is second
    assert handler.get_instance('a.txt') is first


def test_handler_returns_custom_instance_when_limits_differ():
    from rag_data.parsers.parser_handler import ParserHandler

    handler = ParserHandler()
    cached = handler.resolve_parser('a.txt')
    custom = handler.resolve_parser('a.txt', max_chars=7, safe_max_chars=11)
    assert custom is not cached
    assert custom.max_chars == 7 and custom.safe_max_chars == 11
    # 缓存实例不被污染。
    assert cached.max_chars == 1000 and cached.safe_max_chars == 2000
    assert handler.resolve_parser('a.txt') is cached


def test_handler_accepts_default_limits():
    from rag_data.parsers.parser_handler import ParserHandler

    handler = ParserHandler(default_max_chars=42, default_safe_max_chars=84)
    parser = handler.resolve_parser('a.txt')
    assert parser.max_chars == 42
    assert parser.safe_max_chars == 84


def test_builtin_parsers_declared():
    from rag_data.parsers import BUILTIN_PARSERS

    assert set(BUILTIN_PARSERS) == {'markdown', 'txt', 'epub', 'pdf', 'word'}


def test_available_parsers_sorted():
    from rag_data.parsers import available_parsers, is_parser_registered

    names = available_parsers()
    assert names == sorted(names)
    for name in ('markdown', 'txt', 'epub', 'pdf', 'word'):
        assert is_parser_registered(name)


def test_resolve_parser_class():
    from rag_data.parsers import resolve_parser_class
    from rag_data.parsers.txt import TxtParser

    assert resolve_parser_class('txt') is TxtParser


def test_parser_classes_covers_builtin_registry():
    from rag_data.parsers import PARSER_CLASSES, resolve_parser_class

    # PARSER_CLASSES 在导入时由注册表派生，至少覆盖全部内置解析器。
    for name in ('markdown', 'txt', 'epub', 'pdf', 'word'):
        assert resolve_parser_class(name) in PARSER_CLASSES


def test_register_parser_by_decorator():
    from rag_data.parsers import available_parsers, is_parser_registered, register_parser_provider, resolve_parser_class

    @register_parser_provider('custom')
    class _Custom(DocumentParser):
        SUFFIXES = frozenset({'.custom'})

        def parse(self, path): return ''
        def chunk(self, text, nlp=None): return []

    assert is_parser_registered('custom')
    assert resolve_parser_class('custom') is _Custom
    assert 'custom' in available_parsers()


def test_register_parser_by_class_object():
    from rag_data.parsers import register_parser, resolve_parser_class

    class _Local(DocumentParser):
        SUFFIXES = frozenset({'.local'})

        def parse(self, path): return ''
        def chunk(self, text, nlp=None): return []

    register_parser('local-parser', _Local)
    assert resolve_parser_class('local-parser') is _Local


def test_backend_declaration_auto_registers():
    from rag_data.parsers import is_parser_registered, resolve_parser_class

    class _Auto(DocumentParser):
        backend = 'auto-parser'
        SUFFIXES = frozenset({'.auto'})

        def parse(self, path): return ''
        def chunk(self, text, nlp=None): return []

    # 声明 backend 即在类定义时自动注册，无需额外调用。
    assert is_parser_registered('auto-parser')
    assert resolve_parser_class('auto-parser') is _Auto


def test_handler_dispatches_user_registered_parser(tmp_path):
    from rag_data.parsers.parser_handler import ParserHandler

    class _User(DocumentParser):
        backend = None  # 显式登记，避免污染注册表命名空间
        SUFFIXES = frozenset({'.user'})

        def parse(self, path): return 'parsed'
        def chunk(self, text, nlp=None): return ['chunked']

    target = tmp_path / 'doc.user'
    target.write_text('body', encoding='utf-8')
    handler = ParserHandler(parser_classes=(_User,))
    assert handler.get_instance(str(target)) is handler.get_instance(str(target))
    assert handler.parse_document(str(target)) == 'parsed'
    assert handler.chunk_document(str(target), 't') == ['chunked']
