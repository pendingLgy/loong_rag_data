# EpubParser 单元测试：契约、初始化与 chunk 切块（含复杂内容）。
#
# 用「假句柄」返回受控句子序列隔离模型；
# 真实分句用 spacy.blank 的 sentencizer，无需下载模型。

import os

import pytest

from rag_data.exceptions import ParserDependencyError
from rag_data.parsers.base import DocumentParser
from rag_data.parsers.epub import EpubParser


class _Sent:
    def __init__(self, text):
        self.text = text


class _Doc:
    def __init__(self, sentences):
        self.sents = [_Sent(sentence) for sentence in sentences]


def _handle(sentences):
    "返回固定句子序列的假句柄。"
    return lambda text: _Doc(sentences)


def _blank_zh_handle():
    "构造不依赖下载模型的 spaCy 句柄（仅标点 sentencizer）。"
    spacy = pytest.importorskip("spacy", reason="真实分句需要 spacy")
    nlp = spacy.blank("zh")
    nlp.add_pipe("sentencizer")
    return nlp


# ---------------- 契约与初始化 ----------------


def test_epub_parser_is_concrete_document_parser():
    parser = EpubParser()
    assert isinstance(parser, DocumentParser)
    assert EpubParser.__abstractmethods__ == frozenset()
    assert ".epub" in EpubParser.SUFFIXES


def test_epub_default_max_chars():
    parser = EpubParser()
    assert parser.max_chars == 1000
    assert parser.safe_max_chars == 2000


# ---------------- chunk ----------------


def test_chunk_empty_returns_empty_without_handle():
    assert EpubParser().chunk("", nlp=None) == []
    assert EpubParser().chunk("   ", nlp=None) == []


def test_chunk_requires_injected_handle():
    with pytest.raises(ParserDependencyError):
        EpubParser().chunk("some text", nlp=None)


def test_chunk_short_text_is_single_chunk_without_calling_handle():
    def _explode(text):
        raise AssertionError("短文本不应调用句柄")

    assert EpubParser(max_chars=100).chunk("short text", nlp=_explode) == ["short text"]


def test_chunk_splits_by_hash_and_carries_overlap():
    parser = EpubParser(max_chars=1000)
    handle = _handle(["x"])
    result = parser.chunk("#A\nxxxx\n#B\nyyyy", nlp=handle)
    assert result == ["A\nxxxx\n", "xxxx\nB\nyyyy"]


def test_chunk_skips_blank_hash_segments():
    parser = EpubParser(max_chars=1000)
    handle = _handle(["x"])
    assert parser.chunk("#   #A\nb", nlp=handle) == ["A\nb"]


def test_chunk_real_sentencizer_covers_all_sentences():
    nlp = _blank_zh_handle()
    text = "第一句。第二句。第三句。第四句。第五句。"
    chunks = EpubParser(max_chars=8).chunk(text, nlp=nlp)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    joined = "".join(chunk.replace("\n", "") for chunk in chunks)
    for sentence in ["第一句。", "第二句。", "第三句。", "第四句。", "第五句。"]:
        assert sentence in joined


def test_chunk_splits_complex_chinese_text():
    parser = EpubParser(max_chars=120)
    text = (
        "自然语言处理（NLP）是计算机科学与人工智能领域的重要方向。它研究实现人与计算机之间用自然语言通信的理论与方法。"
        "rag-data 项目目前正在优化文本章节的切分逻辑，旨在提升 LLM 的 Retrieval 效果！\n\n"
        "在实际应用中，比如使用 Python 3.10 和 Pytest 进行自动化测试时，我们需要保证句子完整性。"
        "例如 Dr. Smith 在论文中提出的 Chunking 策略，强调重叠前缀（Overlap Prefix）能够显著保留上下文语义。"
    )
    chunks = parser.chunk(text, _blank_zh_handle())
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


# ---------------- 本机真实文件（可选） ----------------


def _local_epub():
    "返回本机可用于冒烟的 epub 路径；不可用则返回空串。"
    override = os.environ.get("RAG_TEST_EPUB","C:\\Users\\loong\\Desktop\\三国演义 ([明] 罗贯中 著  [清] 毛宗岗 等批) (z-library.sk, 1lib.sk, z-lib.sk).epub")
    if override:
        return override
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        return ""
    files = sorted(name for name in os.listdir(desktop) if name.lower().endswith(".epub"))
    return os.path.join(desktop, files[0]) if files else ""


_LOCAL_EPUB = _local_epub()


@pytest.mark.skipif(not _LOCAL_EPUB or not os.path.exists(_LOCAL_EPUB), reason="本机示例 epub 不存在")
def test_local_epub_parse_then_chunk():
    parser = EpubParser(max_chars=500)
    text = parser.parse(_LOCAL_EPUB)
    assert text.strip(), "解析结果不应为空"

    # 整本数十万字，spaCy 逐字切句在此规模上过慢；冒烟取前 5 万字符切块
    sample = text[:50000]
    chunks = parser.chunk(sample, _blank_zh_handle())

    # for chunk in chunks:
    #     print(chunk)
    #     print("-------------------")

    assert chunks, "切块结果不应为空"
    assert all(chunk.strip() for chunk in chunks)
