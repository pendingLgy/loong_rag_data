# parsers 派发测试：按后缀构造解析器并注入切块长度配置。

import pytest

from rag_data.exceptions import ParseError
from rag_data.parsers import (
    PARSER_CLASSES,
    PARSERS,
    SUPPORTED_SUFFIXES,
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


# ---------------- 后缀索引 ----------------


def test_index_covers_all_suffixes():
    expected = set().union(*(cls.SUFFIXES for cls in PARSER_CLASSES))
    assert SUPPORTED_SUFFIXES == expected


def test_default_instances_match_classes():
    assert len(PARSERS) == len(PARSER_CLASSES)
    assert all(isinstance(parser, DocumentParser) for parser in PARSERS)


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
