# MarkdownParser 单元测试：契约、解析、chunk 切块与本机文件。
#
# 用「假句柄」返回受控句子序列隔离模型；真实分句用 spacy.blank 的 sentencizer。

import os

import pytest

from rag_data.exceptions import ParserDependencyError
from rag_data.parsers.base import DocumentParser, load_nlp, DEFAULT_MODEL
from rag_data.parsers.markdown import MarkdownParser


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
    nlp = load_nlp(DEFAULT_MODEL)
    return nlp


# ---------------- 契约与初始化 ----------------


def test_markdown_parser_is_concrete_document_parser():
    parser = MarkdownParser()
    assert isinstance(parser, DocumentParser)
    assert MarkdownParser.__abstractmethods__ == frozenset()
    assert {".md", ".markdown"} <= MarkdownParser.SUFFIXES


def test_markdown_default_max_chars():
    parser = MarkdownParser()
    assert parser.max_chars == 1000
    assert parser.safe_max_chars == 2000


# ---------------- parse ----------------


def test_parse_reads_markdown_verbatim(tmp_path):
    path = tmp_path / "a.md"
    content = "# 标题\n正文一。\n正文二。"
    path.write_text(content, encoding="utf-8")
    assert MarkdownParser().parse(str(path)) == content


def test_parse_accepts_logger(tmp_path, logger):
    path = tmp_path / "a.md"
    path.write_text("正文", encoding="utf-8")
    assert MarkdownParser().parse(str(path), logger=logger) == "正文"


# ---------------- chunk ----------------


def test_chunk_empty_returns_empty_without_handle():
    assert MarkdownParser().chunk("") == []
    assert MarkdownParser().chunk("   \n") == []


def test_chunk_requires_injected_handle():
    with pytest.raises(ParserDependencyError):
        MarkdownParser().chunk("some text")


def test_chunk_short_single_section_without_calling_handle():
    def _explode(text):
        raise AssertionError("短文本不应调用句柄")

    assert MarkdownParser(max_chars=100).chunk("just text", nlp=_explode) == ["just text"]


def test_chunk_splits_by_heading_and_carries_overlap():
    parser = MarkdownParser(max_chars=1000)
    handle = _handle(["x"])
    result = parser.chunk("#A\nxxxx\n#B\nyyyy", nlp=handle)
    assert result == ["A\nxxxx\n", "xxxx\nB\nyyyy"]


def test_chunk_skips_blank_heading_segments():
    parser = MarkdownParser(max_chars=1000)
    handle = _handle(["x"])
    assert parser.chunk("#   #A\nb", nlp=handle) == ["A\nb"]


def test_chunk_real_sentencizer_covers_all_sentences():
    nlp = _blank_zh_handle()
    text = "# 第一章\n第一句。第二句。第三句。\n# 第二章\n第四句。第五句。"
    chunks = MarkdownParser(max_chars=12).chunk(text, nlp)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    joined = "".join(chunk.replace("\n", "") for chunk in chunks)
    for sentence in ["第一句。", "第二句。", "第三句。", "第四句。", "第五句。"]:
        assert sentence in joined


def test_chunk_splits_complex_chinese_markdown():
    parser = MarkdownParser(max_chars=120)
    text = (
        "# 概述\n"
        "自然语言处理（NLP）是计算机科学与人工智能领域的重要方向。它研究人与计算机之间用自然语言通信的方法。\n"
        "# 实现\n"
        "rag-data 正在优化切分逻辑，例如 Dr. Smith 提出的 Chunking 策略强调重叠前缀。"
    )
    chunks = parser.chunk(text, _blank_zh_handle())
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


# ---------------- 本机真实文件（可选） ----------------


def _local_md():
    "返回本机可用于冒烟的 markdown 路径；不可用则返回空串。"
    override = os.environ.get("RAG_TEST_MD","F:\\workspace\\code\\python\\rag-data\\.vcl\\struct.md")
    if override:
        return override
    workspace = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
    if os.path.exists(workspace):
        return workspace
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        return ""
    files = sorted(name for name in os.listdir(desktop) if name.lower().endswith((".md", ".markdown")))
    return os.path.join(desktop, files[0]) if files else ""


_LOCAL_MD = _local_md()


@pytest.mark.skipif(not _LOCAL_MD or not os.path.exists(_LOCAL_MD), reason="本机示例 markdown 不存在")
def test_local_md_parse_then_chunk():
    parser = MarkdownParser(max_chars=500)
    text = parser.parse(_LOCAL_MD)
    assert text.strip(), "解析结果不应为空"

    # 冒烟取前 2 万字符切块，兼顾真实内容与测试时延
    sample = text[:3000]
    chunks = parser.chunk(sample, _blank_zh_handle())

    # for chunk in chunks:
    #     print(chunk)
    #     print("--------- end ----------")

    assert chunks, "切块结果不应为空"
    assert all(chunk.strip() for chunk in chunks)
