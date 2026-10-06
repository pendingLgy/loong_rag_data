# PdfParser 单元测试：契约、解析依赖、chunk 切块与本机文件。
#
# chunk 用「假句柄」返回受控句子序列隔离模型；真实分句用 spacy.blank 的 sentencizer。
# 解析用例需要 pypdf，未安装时逐个 importorskip 跳过。

import os
import sys

import pytest

from rag_data.exceptions import ParseError, ParserDependencyError
from rag_data.parsers.base import DocumentParser, load_nlp, DEFAULT_MODEL
from rag_data.parsers.pdf import PdfParser


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


def test_pdf_parser_is_concrete_document_parser():
    parser = PdfParser()
    assert isinstance(parser, DocumentParser)
    assert PdfParser.__abstractmethods__ == frozenset()
    assert ".pdf" in PdfParser.SUFFIXES


def test_pdf_default_max_chars():
    parser = PdfParser()
    assert parser.max_chars == 1000
    assert parser.safe_max_chars == 2000


# ---------------- chunk ----------------


def test_chunk_empty_returns_empty_without_handle():
    assert PdfParser().chunk("") == []
    assert PdfParser().chunk("   ") == []


def test_chunk_requires_injected_handle():
    with pytest.raises(ParserDependencyError):
        PdfParser().chunk("some text")


def test_chunk_short_text_is_single_chunk_without_calling_handle():
    def _explode(text):
        raise AssertionError("短文本不应调用句柄")

    assert PdfParser(max_chars=100).chunk("short text", nlp=_explode) == ["short text"]


def test_chunk_splits_and_overlaps_by_quarter():
    result = PdfParser(max_chars=6).chunk("abcdefghijkl", nlp=_slice_handle(3))
    assert result == ["abcdef", "def\nghijkl"]


def test_chunk_hard_cuts_overlong_sentence():
    result = PdfParser(max_chars=5).chunk("ABCDEFGH", nlp=_handle(["ABCDEFGH"]))
    assert result == ["ABCDE", "ABCDE\nFGH"]


def test_chunk_real_sentencizer_complex_chinese():
    nlp = _blank_zh_handle()
    text = "PDF 提取的正文往往缺少段落边界。第二句紧随其后！第三句用于验证按句切分是否生效。"
    chunks = PdfParser(max_chars=20).chunk(text, nlp)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


# ---------------- parse 依赖与异常 ----------------


def test_parse_without_dependency_raises_parser_dependency_error(tmp_path, monkeypatch):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4")
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    with pytest.raises(ParserDependencyError):
        PdfParser().parse(str(path))


# ---------------- 本机真实文件（可选） ----------------


def _local_pdf():
    "返回本机可用于冒烟的 pdf 路径；不可用则返回空串。"
    override = os.environ.get("RAG_TEST_PDF","C:\\Users\\loong\\Desktop\\图解HTTP+彩色版.pdf")
    if override:
        return override
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        return ""
    files = sorted(name for name in os.listdir(desktop) if name.lower().endswith(".pdf"))
    return os.path.join(desktop, files[0]) if files else ""


_LOCAL_PDF = _local_pdf()


@pytest.mark.skipif(not _LOCAL_PDF or not os.path.exists(_LOCAL_PDF), reason="本机示例 pdf 不存在")
def test_local_pdf_parse_then_chunk():
    pytest.importorskip("pdfplumber", reason="解析本机 pdf 需要 pdfplumber")
    if os.path.getsize(_LOCAL_PDF) > 15 * 1024 * 1024:
        pytest.skip("本机 pdf 过大，跳过以避免长耗时")
    parser = PdfParser(max_chars=500)
    text = parser.parse(_LOCAL_PDF)
    assert text.strip(), "解析结果不应为空"

    # 冒烟取前 1 万字符切块，兼顾真实内容与测试时延
    sample = text[:10000]
    chunks = parser.chunk(sample, _blank_zh_handle())

    # for chunk in chunks:
    #     print(chunk)
    #     print("--------- end ----------")

    assert chunks, "切块结果不应为空"
    assert all(chunk.strip() for chunk in chunks)
