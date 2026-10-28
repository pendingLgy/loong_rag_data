# WordParser 单元测试：契约、解析依赖与异常、chunk 切块。
#
# chunk 用「假句柄」返回受控句子序列隔离模型；真实分句用 spacy.blank 的 sentencizer。
# 解析用例需要 python-docx，未安装时带原因跳过。

import os
import sys

import pytest

from conftest import SAMPLES_CONFIG

from rag_data.core.exceptions import ParseError, ParserDependencyError
from rag_data.nlp import load_nlp, DEFAULT_MODEL
from rag_data.parsers.base import DocumentParser
from rag_data.parsers.word import WordParser


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


def test_word_parser_is_concrete_document_parser():
    parser = WordParser()
    assert isinstance(parser, DocumentParser)
    assert WordParser.__abstractmethods__ == frozenset()
    assert WordParser.SUFFIXES == frozenset({".docx"})


def test_word_default_max_chars():
    parser = WordParser()
    assert parser.max_chars == 1000
    assert parser.safe_max_chars == 2000


# ---------------- chunk ----------------


def test_chunk_empty_returns_empty_without_handle():
    assert WordParser().chunk("") == []
    assert WordParser().chunk("   \n") == []


def test_chunk_requires_injected_handle():
    with pytest.raises(ParserDependencyError):
        WordParser().chunk("some text")


def test_chunk_short_single_section_without_calling_handle():
    def _explode(text):
        raise AssertionError("短文本不应调用句柄")

    assert WordParser(max_chars=100).chunk("just text", nlp=_explode) == ["just text"]


def test_chunk_splits_by_heading_and_carries_overlap():
    parser = WordParser(max_chars=1000)
    handle = _handle(["x"])
    result = parser.chunk("#A\nxxxx\n#B\nyyyy", nlp=handle)
    assert result == ["A\nxxxx\n", "xxxx\nB\nyyyy"]


def test_chunk_skips_blank_heading_segments():
    parser = WordParser(max_chars=1000)
    handle = _handle(["x"])
    assert parser.chunk("#   #A\nb", nlp=handle) == ["A\nb"]


def test_chunk_real_sentencizer_covers_all_sentences():
    nlp = _blank_zh_handle()
    text = "# 第一章\n第一句。第二句。第三句。\n# 第二章\n第四句。第五句。"
    chunks = WordParser(max_chars=12).chunk(text, nlp)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    joined = "".join(chunk.replace("\n", "") for chunk in chunks)
    for sentence in ["第一句。", "第二句。", "第三句。", "第四句。", "第五句。"]:
        assert sentence in joined


# ---------------- parse 依赖与异常 ----------------


def test_parse_without_dependency_raises_parser_dependency_error(tmp_path, monkeypatch):
    path = tmp_path / "a.docx"
    path.write_bytes(b"PK")
    monkeypatch.setitem(sys.modules, "docx", None)
    with pytest.raises(ParserDependencyError):
        WordParser().parse(str(path))


def test_parse_rejects_legacy_doc_without_dependency():
    # .doc 的校验先于依赖导入，故无需 python-docx 也会抛 ParseError
    with pytest.raises(ParseError):
        WordParser().parse("legacy.doc")


# ---------------- 解析行为（需 python-docx） ----------------


def test_parse_extracts_paragraphs_verbatim(tmp_path):
    try:
        import docx
    except ImportError as exc:
        pytest.skip("缺少可选依赖 python-docx：" + str(exc) + "；安装方式：pip install python-docx")
    document = docx.Document()
    document.add_heading("标题一", level=1)
    document.add_paragraph("正文一。")
    document.add_paragraph("正文二。")
    path = tmp_path / "a.docx"
    document.save(str(path))

    text = WordParser().parse(str(path))
    lines = text.split("\n")
    assert "标题一" in lines
    assert "正文一。" in lines
    assert "正文二。" in lines


# ---------------- 本机真实文件（可选） ----------------


def _local_docx():
    # 样例路径来自 conftest 的 SAMPLES_CONFIG
    return SAMPLES_CONFIG["docx"]


_LOCAL_DOCX = _local_docx()


@pytest.mark.skipif(not _LOCAL_DOCX or not os.path.exists(_LOCAL_DOCX), reason="本机示例 docx 不存在：请设置 samples.docx，或把 .docx 放到桌面")
def test_local_docx_parse_then_chunk():
    parser = WordParser(max_chars=500)
    text = parser.parse(_LOCAL_DOCX)
    assert text.strip(), "解析结果不应为空"

    # 大文件逐字切句较慢；冒烟取前 2 万字符切块
    sample = text[:3000]
    chunks = parser.chunk(sample, _blank_zh_handle())

    # for chunk in chunks:
    #     print(chunk)
    #     print("--------- end ----------")

    assert chunks, "切块结果不应为空"
    assert all(chunk.strip() for chunk in chunks)
