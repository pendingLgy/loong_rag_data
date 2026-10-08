# DocumentParser 单元测试：分句、重叠前缀与递归切块等基类能力。
#
# 用「假句柄」返回受控句子序列，隔离真实模型。

import pytest

from rag_data.parsers.base import DocumentParser


class _Sent:
    def __init__(self, text):
        self.text = text


class _Doc:
    def __init__(self, sentences):
        self.sents = [_Sent(sentence) for sentence in sentences]


class _Dummy(DocumentParser):
    "最小具体子类：用于直接调用基类提供的分句与切块方法。"

    def parse(self, path):
        return ""

    def chunk(self, text, nlp=None):
        return []


def _handle(sentences):
    "返回固定句子序列的假句柄。"
    return lambda text: _Doc(sentences)


def _slice_handle(size):
    "按固定长度切片文本的假句柄，模拟可递归续切的句柄。"
    def _call(text):
        return _Doc([text[i:i + size] for i in range(0, len(text), size)])

    return _call


# ---------------- 初始化 ----------------


def test_base_init_defaults():
    parser = _Dummy()
    assert parser.max_chars == 500
    assert parser.safe_max_chars == 2000


def test_base_max_chars_is_configurable():
    assert _Dummy(max_chars=7).max_chars == 7


# ---------------- split_sentences ----------------


def test_split_sentences_returns_stripped_sentences():
    handle = _handle(["  a  ", "b"])
    assert _Dummy().split_sentences("x", handle) == ["a", "b"]


def test_split_sentences_expands_embedded_newlines():
    handle = _handle(["a\nb"])
    assert _Dummy().split_sentences("a\nb", handle) == ["a\n", "b"]


def test_split_sentences_falls_back_to_raw_text():
    handle = _handle([])
    assert _Dummy().split_sentences("raw text", handle) == ["raw text"]


def test_split_sentences_empty_text_returns_empty():
    assert _Dummy().split_sentences("", _handle(["x"])) == []


# ---------------- _extract_overlap_prefix ----------------


def test_overlap_prefix_takes_last_quarter():
    sentences = ["aaa", "bbb", "ccc", "ddd"]
    assert _Dummy()._extract_overlap_prefix(sentences) == "ddd"


def test_overlap_prefix_accumulates_until_reaching_quarter():
    assert _Dummy()._extract_overlap_prefix(["1111", "2"]) == "1111\n2"


def test_overlap_prefix_flattens_embedded_newlines():
    assert _Dummy()._extract_overlap_prefix(["aa\nbb", "cc"]) == "cc"


def test_overlap_prefix_empty_inputs():
    assert _Dummy()._extract_overlap_prefix([]) == ""
    assert _Dummy()._extract_overlap_prefix([""]) == ""
    assert _Dummy()._extract_overlap_prefix(["   "]) == ""


# ---------------- _process_text_recursive ----------------


def test_process_short_text_returns_single_chunk_without_prefix():
    handle = _handle(["x"])
    assert _Dummy(max_chars=100)._process_text_recursive("hello", "", handle) == ["hello"]


def test_process_short_text_prepends_overlap_prefix():
    handle = _handle(["x"])
    result = _Dummy(max_chars=100)._process_text_recursive("hello", "PRE", handle)
    assert result == ["PRE\nhello"]


def test_process_empty_returns_empty():
    handle = _handle(["x"])
    assert _Dummy()._process_text_recursive("", "", handle) == []
    assert _Dummy()._process_text_recursive("   ", "", handle) == []


def test_process_splits_and_overlaps_by_quarter():
    result = _Dummy(max_chars=6)._process_text_recursive("abcdefghijkl", "", _slice_handle(3))
    assert result == ["abcdef", "def\nghijkl"]


def test_process_chains_recursion_with_carry_overlap():
    result = _Dummy(max_chars=6)._process_text_recursive("abcdefghijklmno", "", _slice_handle(3))
    assert result == ["abcdef", "def\nghijkl", "jkl\nmno"]


def test_process_hard_cuts_sentence_longer_than_limit():
    result = _Dummy(max_chars=5)._process_text_recursive("ABCDEFGH", "", _handle(["ABCDEFGH"]))
    assert result == ["ABCDE", "ABCDE\nFGH"]
