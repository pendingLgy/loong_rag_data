# 门面集成测试：RagData 用真实 ParserHandler 解析并切块临时文件。
#
# 覆盖门面装配到 handler 派发再到各格式 parse 与 chunk 的真实链路。
# spaCy 句柄缺失时仅跳过切块用例，可用 pytest -rs 查看跳过清单。
#
# 配置来源为 conftest 的 *_JSON 与各分区设置对象。

from __future__ import annotations

import os
import sys

import pytest

from conftest import CASES_CONFIG, EMBEDDING_SETTINGS, PARSING_SETTINGS
from rag_data import RagData, RagSetting
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ParseError, ParserDependencyError
from rag_data.parsers.parser_handler import ParserHandler

# 切块长度取 facade_ragdata 组的覆盖值，其余参数来自设置对象。
SPACY_MODEL = PARSING_SETTINGS.spacy_model
MAX_CHARS = CASES_CONFIG["facade_ragdata"]["max_chars"]
SAFE_MAX_CHARS = CASES_CONFIG["facade_ragdata"]["safe_max_chars"]

# 两段短正文：句子均短于 max_chars，便于在句边界收块。
SECTION_ONE = (
    "第一段用于验证纯文本解析，句子之间用中文句号分隔。"
    "切块应保留全部句子，并按长度上限逐块收拢。"
)
SECTION_TWO = (
    "第二段用于验证尾部不丢失，唯一定位句：紫金山下的旧书店。"
    "最后一句作为收尾，确保递归切块处理干净。"
)

pytestmark = pytest.mark.integration

if "--integration" not in sys.argv:
    pytest.skip(
        f"[SKIP INFO] File {__file__}，integration 集成测试默认跳过",
        allow_module_level=True,
    )


# ---------------- 辅助 ----------------


def _requires_api_key() -> None: 
    if not EMBEDDING_SETTINGS.api_key: 
        pytest.skip("未配置 API Key，请在 conftest 的 embedding.api_key 中填入")


def _requires_nlp(app: RagData) -> None: 
    # 切块依赖 spaCy 句柄：模型缺失时门面 nlp 为 None，跳过而非失败。
    if app.nlp is None: 
        pytest.skip("未安装 spaCy 或模型 " + SPACY_MODEL + "，跳过切块用例")


# ---------------- 夹具 ----------------


@pytest.fixture(scope="module")
def settings() -> RagSetting: 
    return RagSetting.load_mapping(
        parsing={
            "enable": True,
            "spacy_model": SPACY_MODEL,
            "max_chars": MAX_CHARS,
            "safe_max_chars": SAFE_MAX_CHARS,
        },
        embedding={
            "enable": True,
            "provider": EMBEDDING_SETTINGS.provider,
            "model": EMBEDDING_SETTINGS.model,
            "api_key": EMBEDDING_SETTINGS.api_key,
            "dim": EMBEDDING_SETTINGS.dim,
        },
    )


@pytest.fixture(scope="module")
def app(settings: RagSetting) -> RagData: 
    return RagData(settings)


@pytest.fixture(scope="module")
def doc_dir(tmp_path_factory) -> str: 
    return str(tmp_path_factory.mktemp("ragdata_it"))


@pytest.fixture(scope="module")
def txt_file(doc_dir: str) -> str: 
    path = os.path.join(doc_dir, "sample.txt")
    with open(path, "w", encoding="utf-8") as stream: 
        stream.write(SECTION_ONE + chr(10) + SECTION_TWO)
    return path


@pytest.fixture(scope="module")
def md_file(doc_dir: str) -> str: 
    path = os.path.join(doc_dir, "sample.md")
    with open(path, "w", encoding="utf-8") as stream: 
        stream.write("# 第一章" + chr(10) + SECTION_ONE + chr(10) + "# 第二章" + chr(10) + SECTION_TWO)
    return path


def test_wires_real_parser_handler(app: RagData, settings: RagSetting): 
    assert app.settings is settings
    assert isinstance(app.parser_handler, ParserHandler)
    assert app.parser_handler is app.parser


def test_parser_handler_prebuilds_all_parsers(app: RagData): 
    instances = app.parser_handler.instances
    assert instances
    assert all(parser.max_chars == MAX_CHARS for parser in instances.values())


def test_disabled_sections_leave_components_none(): 
    plain = RagData(RagSetting.load_mapping())
    assert plain.nlp is None
    assert plain.embedder is None
    assert isinstance(plain.parser, ParserHandler)


def test_parse_txt_returns_raw_text(app: RagData, txt_file: str): 
    text = app.parse_document(txt_file)
    assert text.startswith(SECTION_ONE[:12])
    assert len(text) == len(SECTION_ONE) + 1 + len(SECTION_TWO)


def test_parse_markdown_keeps_headings(app: RagData, md_file: str): 
    text = app.parse_document(md_file)
    assert "# 第一章" in text
    assert "# 第二章" in text


def test_parse_missing_file_raises_parse_error(app: RagData, doc_dir: str): 
    with pytest.raises(ParseError): 
        app.parse_document(os.path.join(doc_dir, "missing.txt"))


def test_parse_unsupported_suffix_raises_parse_error(app: RagData, doc_dir: str): 
    path = os.path.join(doc_dir, "sample.unknown")
    with open(path, "w", encoding="utf-8") as stream: 
        stream.write("占位")
    with pytest.raises(ParseError): 
        app.parse_document(path)


def test_chunk_txt_covers_whole_text(app: RagData, txt_file: str): 
    _requires_nlp(app)
    text = app.parse_document(txt_file)
    chunks = app.chunk_document(txt_file, text)
    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)
    assert SECTION_TWO[:8] in "".join(chunks)
    assert max(len(chunk) for chunk in chunks) <= MAX_CHARS * 2


def test_chunk_markdown_splits_by_heading(app: RagData, md_file: str): 
    _requires_nlp(app)
    text = app.parse_document(md_file)
    chunks = app.chunk_document(md_file, text)
    assert len(chunks) >= 2
    assert any("第一章" in chunk for chunk in chunks)
    assert any("第二章" in chunk for chunk in chunks)


def test_chunk_length_follows_parsing_limit(txt_file: str): 
    narrow = RagData(RagSetting.load_mapping(parsing={
        "enable": True, "spacy_model": SPACY_MODEL, "max_chars": 40, "safe_max_chars": SAFE_MAX_CHARS,
    }))
    _requires_nlp(narrow)
    wide = RagData(RagSetting.load_mapping(parsing={
        "enable": True, "spacy_model": SPACY_MODEL, "max_chars": 400, "safe_max_chars": SAFE_MAX_CHARS,
    }))
    text = narrow.parse_document(txt_file)
    assert len(narrow.chunk_document(txt_file, text)) > len(wide.chunk_document(txt_file, text))


def test_chunk_without_handle_raises(txt_file: str): 
    plain = RagData(RagSetting.load_mapping())
    assert plain.nlp is None
    with pytest.raises(ParserDependencyError): 
        plain.chunk_document(txt_file, SECTION_ONE)


def test_embedder_encodes_with_real_provider(app: RagData): 
    _requires_api_key()
    assert isinstance(app.embedder, EmbedderHandler)
    vectors = app.embedder.encode([SECTION_ONE[:20], SECTION_TWO[:20]])
    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_SETTINGS.dim for vector in vectors)
