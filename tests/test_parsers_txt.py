# TxtParser 单元测试：契约、解析、chunk 切块与本机文件。
#
# 用「假句柄」返回受控句子序列隔离模型；真实分句用 spacy.blank 的 sentencizer。

import os

import pytest

from rag_data.exceptions import ParserDependencyError
from rag_data.parsers.base import DocumentParser, load_nlp, DEFAULT_MODEL
from rag_data.parsers.txt import TxtParser


class _Sent:
    def __init__(self, text):
        self.text = text


class _Doc:
    def __init__(self, sentences):
        self.sents = [_Sent(sentence) for sentence in sentences]


def _handle(sentences):
    "返回固定句子序列的假句柄。"
    return lambda text: _Doc(sentences)


def _slice_handle(size):
    "按固定长度切片文本的假句柄，模拟可递归续切的句柄。"
    def _call(text):
        return _Doc([text[i:i + size] for i in range(0, len(text), size)])

    return _call


def _blank_zh_handle():
    nlp = load_nlp(DEFAULT_MODEL)
    return nlp


# ---------------- 契约与初始化 ----------------


def test_txt_parser_is_concrete_document_parser():
    parser = TxtParser()
    assert isinstance(parser, DocumentParser)
    assert TxtParser.__abstractmethods__ == frozenset()
    assert ".txt" in TxtParser.SUFFIXES


def test_txt_default_max_chars():
    parser = TxtParser()
    assert parser.max_chars == 1000
    assert parser.safe_max_chars == 2000


# ---------------- parse ----------------


def test_parse_reads_text_verbatim(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("第一行\n第二行", encoding="utf-8")
    assert TxtParser().parse(str(path)) == "第一行\n第二行"


def test_parse_accepts_logger(tmp_path, logger):
    path = tmp_path / "a.txt"
    path.write_text("正文", encoding="utf-8")
    assert TxtParser().parse(str(path), logger=logger) == "正文"


# ---------------- chunk ----------------


def test_chunk_empty_returns_empty_without_handle():
    assert TxtParser().chunk("") == []
    assert TxtParser().chunk("   ") == []


def test_chunk_requires_injected_handle():
    with pytest.raises(ParserDependencyError):
        TxtParser().chunk("some text")


def test_chunk_short_text_is_single_chunk_without_calling_handle():
    def _explode(text):
        raise AssertionError("短文本不应调用句柄")

    assert TxtParser(max_chars=100).chunk("short text", nlp=_explode) == ["short text"]


def test_chunk_splits_and_overlaps_by_quarter():
    result = TxtParser(max_chars=6).chunk("abcdefghijkl", nlp=_slice_handle(3))
    assert result == ["abcdef", "def\nghijkl"]


def test_chunk_chains_recursion_with_carry_overlap():
    result = TxtParser(max_chars=6).chunk("abcdefghijklmno", nlp=_slice_handle(3))
    assert result == ["abcdef", "def\nghijkl", "jkl\nmno"]


def test_chunk_hard_cuts_overlong_sentence():
    result = TxtParser(max_chars=5).chunk("ABCDEFGH", nlp=_handle(["ABCDEFGH"]))
    assert result == ["ABCDE", "ABCDE\nFGH"]


def test_chunk_real_sentencizer_complex_chinese():
    nlp = _blank_zh_handle()
    text = (
        "自然语言处理（NLP）是计算机科学的重要方向。它研究人与计算机之间用自然语言通信的方法。"
        "rag-data 项目正在优化切分逻辑，旨在提升 LLM 的 Retrieval 效果！\n\n"
        "例如 Dr. Smith 提出的 Chunking 策略，强调重叠前缀能够保留上下文语义。"
    )
    chunks = TxtParser(max_chars=60).chunk(text, nlp)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


# ---------------- 本机真实文件（可选） ----------------


def _local_txt():
    "返回本机可用于冒烟的 txt 路径；不可用则返回空串。"
    override = os.environ.get("RAG_TEST_TXT","C:\\Users\\loong\\Desktop\\伯特兰·罗素：哲学问题.txt")
    if override:
        return override
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        return ""
    files = sorted(name for name in os.listdir(desktop) if name.lower().endswith(".txt"))
    return os.path.join(desktop, files[0]) if files else ""


_LOCAL_TXT = _local_txt()


@pytest.mark.skipif(not _LOCAL_TXT or not os.path.exists(_LOCAL_TXT), reason="本机示例 txt 不存在")
def test_local_txt_parse_then_chunk():
    parser = TxtParser(max_chars=2000)
    text = parser.parse(_LOCAL_TXT)
    assert text.strip(), "解析结果不应为空"

    # 大文件逐字切句较慢；冒烟取前 2 万字符切块
    sample = text[:3000]
    chunks = parser.chunk(sample, _blank_zh_handle())

    # for chunk in chunks:
    #     print(chunk)
    #     print("--------- end ----------")

    assert chunks, "切块结果不应为空"
    assert all(chunk.strip() for chunk in chunks)
