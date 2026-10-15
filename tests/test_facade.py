# 流程门面（facade）单元测试：装配方法 _build_*、RagData 构造与流程方法。
#
# 用假向量模型与假解析器隔离网络与 spaCy；默认装配分支由 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.exceptions import ParseError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.parser_adapter import ParserAdapter


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

    def parse_document(self, path, logger=None):
        self.parse_calls.append({'path': path, 'logger': logger})
        if self.error is not None:
            raise self.error
        return self.text

    def chunk_document(self, path, text, nlp=None, logger=None, max_chars=None, safe_max_chars=None):
        self.chunk_calls.append({
            'path': path, 'text': text, 'nlp': nlp, 'logger': logger,
            'max_chars': max_chars, 'safe_max_chars': safe_max_chars,
        })
        if self.error is not None:
            raise self.error
        return list(self.chunks)


# ---------------- 夹具与构造 ----------------


@pytest.fixture()
def det_settings():
    # 不读取环境变量，保证配置可复现。
    return Settings.load_mapping()


def _install_fakes(monkeypatch, logger, nlp, embedder):
    # 打桩三个私有装配方法，使 RagData 构造不触真实日志后端、spaCy 与向量 provider。
    monkeypatch.setattr(facade, '_build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, '_build_nlp', lambda *args, **kwargs: nlp)
    monkeypatch.setattr(facade, '_build_embedder', lambda *args, **kwargs: embedder)


def _make_app(monkeypatch, settings, logger, nlp=None, parser=None):
    # 构造经 monkeypatch 注入假组件的 RagData；parser 缺省时用真实 ParserAdapter。
    embedder = Embedder(settings.embedding, logger, model=_FakeModel(dim=settings.embedding.dim))
    _install_fakes(monkeypatch, logger, nlp if nlp is not None else object(), embedder)
    return facade.RagData(settings, parser=parser)


# ---------------- _build_logger ----------------


def test_build_logger_returns_adapter(det_settings):
    assert isinstance(facade._build_logger(det_settings.logging), LoggerAdapter)


def test_build_logger_forwards_logging_fields(monkeypatch, det_settings):
    # 后端、级别、格式与时区应逐字段取自 logging 分区。
    captured = {}

    class _Factory:
        @staticmethod
        def setup(**kwargs):
            captured.update(kwargs)

        @staticmethod
        def get_logger():
            return 'sentinel'

    monkeypatch.setattr(facade, 'LoggerFactory', _Factory)
    logging_settings = det_settings.logging
    assert facade._build_logger(logging_settings) == 'sentinel'
    assert captured == {
        'backend': logging_settings.backend,
        'timezone_name': logging_settings.timezone or None,
        'level': logging_settings.level,
        'fmt': logging_settings.format or None,
    }


# ---------------- _build_nlp ----------------


def test_build_nlp_delegates_model_and_logger(monkeypatch, det_settings, logger):
    captured = {}
    sentinel = object()

    def fake_load(model_name, log):
        captured['model'] = model_name
        captured['logger'] = log
        return sentinel

    monkeypatch.setattr(facade, 'load_nlp', fake_load)
    assert facade._build_nlp(det_settings.parsing, logger) is sentinel
    assert captured['model'] == det_settings.parsing.spacy_model
    assert captured['logger'] is logger


def test_build_nlp_returns_handle_or_none(det_settings, logger):
    result = facade._build_nlp(det_settings.parsing, logger)
    assert result is None or callable(result)


# ---------------- _build_embedder ----------------


def test_build_embedder_injected_model(det_settings, logger):
    model = _FakeModel(dim=det_settings.embedding.dim)
    result = facade._build_embedder(det_settings.embedding, logger, model=model)
    assert isinstance(result, Embedder)
    assert result.provider is model


def test_build_embedder_without_model_is_lazy(det_settings, logger):
    result = facade._build_embedder(det_settings.embedding, logger)
    assert isinstance(result, Embedder)


# ---------------- RagData 构造 ----------------


def test_ragdata_builds_components_from_settings(monkeypatch, det_settings, logger):
    # 构造只接收 settings，其余组件均由 _build_* 派生。
    nlp = object()
    embedder = Embedder(det_settings.embedding, logger, model=_FakeModel(dim=det_settings.embedding.dim))
    _install_fakes(monkeypatch, logger, nlp, embedder)

    app = facade.RagData(det_settings)

    assert app.settings is det_settings
    assert app.logger is logger
    assert app.nlp is nlp
    assert app.embedder is embedder
    assert isinstance(app.parser, ParserAdapter)


def test_ragdata_passes_section_settings_to_builders(monkeypatch, det_settings, logger):
    # 三个装配方法分别接收 logging、parsing、embedding 分区。
    captured = {}

    def fake_logger(logging_settings):
        captured['logger'] = logging_settings
        return logger

    def fake_nlp(parsing_settings, log):
        captured['nlp'] = (parsing_settings, log)
        return None

    def fake_embedder(embedding_settings, log):
        captured['embedder'] = (embedding_settings, log)
        return object()

    monkeypatch.setattr(facade, '_build_logger', fake_logger)
    monkeypatch.setattr(facade, '_build_nlp', fake_nlp)
    monkeypatch.setattr(facade, '_build_embedder', fake_embedder)

    facade.RagData(det_settings)

    assert captured['logger'] is det_settings.logging
    assert captured['nlp'] == (det_settings.parsing, logger)
    assert captured['embedder'] == (det_settings.embedding, logger)


def test_ragdata_keeps_injected_parser(monkeypatch, det_settings, logger):
    parser = _FakeParser()
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    assert app.parser is parser


# ---------------- parser_adapter ----------------


def test_parser_adapter_returns_injected_parser(monkeypatch, det_settings, logger):
    parser = _FakeParser()
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    assert app.parser_adapter is parser


def test_parser_adapter_default_is_parser_adapter_instance(monkeypatch, det_settings, logger):
    app = _make_app(monkeypatch, det_settings, logger)
    assert isinstance(app.parser_adapter, ParserAdapter)


# ---------------- parse_document ----------------


def test_parse_document_delegates_to_parser(monkeypatch, det_settings, logger):
    parser = _FakeParser(text='纯文本')
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    assert app.parse_document('a.txt') == '纯文本'
    assert parser.parse_calls == [{'path': 'a.txt', 'logger': logger}]


def test_parse_document_propagates_parse_error(monkeypatch, det_settings, logger):
    parser = _FakeParser(error=ParseError('损坏的文件'))
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    with pytest.raises(ParseError):
        app.parse_document('a.txt')


# ---------------- chunk_document ----------------


def test_chunk_document_delegates_with_limits(monkeypatch, det_settings, logger):
    # 应把 nlp、logger 与 parsing 分区的长度配置一并透传给解析派发入口。
    parser = _FakeParser(chunks=['c1', 'c2'])
    nlp = object()
    app = _make_app(monkeypatch, det_settings, logger, nlp=nlp, parser=parser)
    assert app.chunk_document('a.txt', '正文') == ['c1', 'c2']
    assert parser.chunk_calls == [{
        'path': 'a.txt', 'text': '正文', 'nlp': nlp, 'logger': logger,
        'max_chars': det_settings.parsing.max_chars,
        'safe_max_chars': det_settings.parsing.safe_max_chars,
    }]


def test_chunk_document_forwards_overridden_limits(monkeypatch, logger):
    settings = Settings.load_mapping(parsing={'max_chars': 7, 'safe_max_chars': 9})
    parser = _FakeParser(chunks=['ok'])
    app = _make_app(monkeypatch, settings, logger, parser=parser)
    app.chunk_document('a.txt', '正文')
    assert parser.chunk_calls[0]['max_chars'] == 7
    assert parser.chunk_calls[0]['safe_max_chars'] == 9


def test_chunk_document_empty_text_returns_parser_result(monkeypatch, det_settings, logger):
    parser = _FakeParser(chunks=[])
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    assert app.chunk_document('a.txt', '') == []


def test_chunk_document_propagates_parse_error(monkeypatch, det_settings, logger):
    parser = _FakeParser(error=ParseError('暂不支持的文档格式'))
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    with pytest.raises(ParseError):
        app.chunk_document('a.xyz', '正文')


# ---------------- 契约 ----------------


def test_ragdata_exposes_flow_members():
    for name in ('parser_adapter', 'parse_document', 'chunk_document'):
        assert hasattr(facade.RagData, name)


def test_facade_all_symbols():
    assert set(facade.__all__) == {'RagData'}


