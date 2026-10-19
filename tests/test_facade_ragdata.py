# 门面单元测试：RagData 的组件装配、解析派发与切块委派，以及构造签名契约。
#
# 用假向量模型与假解析器隔离网络与 spaCy；默认装配分支由 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import RagSetting
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ParseError
from rag_data.parsers.parser_handler import ParserHandler

# ---------------- 测试替身 ----------------


class _FakeModel:
    # 受控向量模型：记录每次 encode 的入参，返回固定维度向量。
    def __init__(self, dim, max_batch_size=0):
        self.dim = dim
        self.max_batch_size = max_batch_size
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * self.dim for _ in texts]


class _FakeParser:
    # 受控解析派发替身：记录调用入参，返回预置结果或抛出预置异常。
    def __init__(self, text='', chunks=None, error=None):
        self.text = text
        self.chunks = list(chunks) if chunks is not None else []
        self.error = error
        self.parse_calls = []
        self.chunk_calls = []

    def parse_document(self, path):
        self.parse_calls.append({'path': path})
        if self.error is not None:
            raise self.error
        return self.text

    def chunk_document(self, path, text, nlp=None, max_chars=None, safe_max_chars=None):
        self.chunk_calls.append({
            'path': path, 'text': text, 'nlp': nlp,
            'max_chars': max_chars, 'safe_max_chars': safe_max_chars,
        })
        if self.error is not None:
            raise self.error
        return list(self.chunks)


# ---------------- 夹具与构造 ----------------


@pytest.fixture()
def det_settings():
    # 不读取环境变量，保证配置可复现；各分区默认关闭，这里统一打开以便测试装配分支。
    return RagSetting.load_mapping(
        parsing={'enable': True},
        embedding={'enable': True},
    )


def _install_fakes(monkeypatch, nlp, embedder):
    # 打桩 RagData 用到的两个私有装配方法，避免触碰真实 spaCy 与向量 provider。
    monkeypatch.setattr(facade, '_build_nlp', lambda *args, **kwargs: nlp)
    monkeypatch.setattr(facade, '_build_embedder', lambda *args, **kwargs: embedder)


def _make_app(monkeypatch, settings, nlp=None, parser=None):
    # 构造经 monkeypatch 注入假组件的 RagData；替换解析器时打桩 facade.ParserHandler。
    embedder = EmbedderHandler(settings.embedding, model=_FakeModel(dim=settings.embedding.dim))
    _install_fakes(monkeypatch, nlp if nlp is not None else object(), embedder)
    if parser is not None:
        monkeypatch.setattr(facade, 'ParserHandler', lambda *args, **kwargs: parser)
    return facade.RagData(settings)


# ---------------- RagData 构造 ----------------


def test_ragdata_builds_components_from_settings(monkeypatch, det_settings):
    # 构造只装配解析器、句柄与嵌入器；库与模型交给 MemoryServer。
    nlp = object()
    embedder = EmbedderHandler(det_settings.embedding, model=_FakeModel(dim=det_settings.embedding.dim))
    _install_fakes(monkeypatch, nlp, embedder)

    app = facade.RagData(det_settings)

    assert app.settings is det_settings
    assert app.nlp is nlp
    assert app.embedder is embedder
    assert isinstance(app.parser, ParserHandler)
    assert not hasattr(app, 'db')
    assert not hasattr(app, 'vector')
    assert not hasattr(app, 'llm')


def test_ragdata_passes_section_settings_to_builders(monkeypatch, det_settings):
    # 两个装配方法分别收到 parsing、embedding 分区。
    captured = {}

    def fake_nlp(settings):
        captured['nlp'] = settings
        return None

    def fake_embedder(settings, model=None):
        captured['embedder'] = settings
        return object()

    monkeypatch.setattr(facade, '_build_nlp', fake_nlp)
    monkeypatch.setattr(facade, '_build_embedder', fake_embedder)
    facade.RagData(det_settings)

    assert captured['nlp'] is det_settings.parsing
    assert captured['embedder'] is det_settings.embedding


def test_ragdata_signature_accepts_settings():
    # 构造签名只保留 settings，解析器不对外暴露。
    import inspect

    parameters = inspect.signature(facade.RagData.__init__).parameters
    assert 'parser' not in parameters
    assert list(parameters) == ['self', 'settings']


def test_ragdata_builds_parser_handler_with_parsing_limits(monkeypatch, det_settings):
    # ParserHandler 以 parsing 分区的长度上限预实例化。
    captured = {}

    def fake_handler(**kwargs):
        captured.update(kwargs)
        return _FakeParser()

    _install_fakes(monkeypatch, object(), object())
    monkeypatch.setattr(facade, 'ParserHandler', fake_handler)
    facade.RagData(det_settings)
    assert captured == {
        'default_max_chars': det_settings.parsing.max_chars,
        'default_safe_max_chars': det_settings.parsing.safe_max_chars,
    }


# ---------------- parser_handler ----------------


def test_parser_handler_returns_patched_parser(monkeypatch, det_settings):
    parser = _FakeParser()
    app = _make_app(monkeypatch, det_settings, parser=parser)
    assert app.parser_handler is parser


def test_parser_handler_defaults_to_real_handler(monkeypatch, det_settings):
    app = _make_app(monkeypatch, det_settings)
    assert isinstance(app.parser_handler, ParserHandler)


# ---------------- parse_document ----------------


def test_parse_document_delegates_to_parser(monkeypatch, det_settings):
    parser = _FakeParser(text='纯文本')
    app = _make_app(monkeypatch, det_settings, parser=parser)
    assert app.parse_document('a.txt') == '纯文本'
    assert parser.parse_calls == [{'path': 'a.txt'}]


def test_parse_document_propagates_parse_error(monkeypatch, det_settings):
    parser = _FakeParser(error=ParseError('损坏的文件'))
    app = _make_app(monkeypatch, det_settings, parser=parser)
    with pytest.raises(ParseError):
        app.parse_document('a.txt')


# ---------------- chunk_document ----------------


def test_chunk_document_delegates_with_limits(monkeypatch, det_settings):
    # 应把 nlp、logger 与 parsing 分区的长度配置一并透传给解析派发入口。
    parser = _FakeParser(chunks=['c1', 'c2'])
    nlp = object()
    app = _make_app(monkeypatch, det_settings, nlp=nlp, parser=parser)
    assert app.chunk_document('a.txt', '正文') == ['c1', 'c2']
    assert parser.chunk_calls == [{
        'path': 'a.txt', 'text': '正文', 'nlp': nlp,
        'max_chars': det_settings.parsing.max_chars,
        'safe_max_chars': det_settings.parsing.safe_max_chars,
    }]


def test_chunk_document_forwards_overridden_limits(monkeypatch):
    settings = RagSetting.load_mapping(parsing={'max_chars': 7, 'safe_max_chars': 9})
    parser = _FakeParser(chunks=['ok'])
    app = _make_app(monkeypatch, settings, parser=parser)
    app.chunk_document('a.txt', '正文')
    assert parser.chunk_calls[0]['max_chars'] == 7
    assert parser.chunk_calls[0]['safe_max_chars'] == 9


def test_chunk_document_empty_text_returns_parser_result(monkeypatch, det_settings):
    parser = _FakeParser(chunks=[])
    app = _make_app(monkeypatch, det_settings, parser=parser)
    assert app.chunk_document('a.txt', '') == []


def test_chunk_document_propagates_parse_error(monkeypatch, det_settings):
    parser = _FakeParser(error=ParseError('暂不支持的文档格式'))
    app = _make_app(monkeypatch, det_settings, parser=parser)
    with pytest.raises(ParseError):
        app.chunk_document('a.xyz', '正文')


# ---------------- 契约 ----------------


def test_ragdata_exposes_flow_members():
    for name in ('parser_handler', 'parse_document', 'chunk_document'):
        assert hasattr(facade.RagData, name)


def test_ragdata_skips_disabled_sections():
    # 默认配置下 RagData 只装配解析器，其余组件均为 None。
    app = facade.RagData(RagSetting.load_mapping())
    assert app.nlp is None
    assert app.embedder is None
    assert isinstance(app.parser, ParserHandler)

