# 流程门面（facade）单元测试：装配方法 _build_*、RagData 构造与流程方法。
#
# 用假向量模型与假解析器隔离网络与 spaCy；默认装配分支由 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import Settings
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ConfigError, ParseError
from rag_data.logging.base import LoggerAdapter
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
    # 不读取环境变量，保证配置可复现；各分区默认关闭，这里统一打开以便测试装配分支。
    return Settings.load_mapping(
        parsing={'enable': True},
        embedding={'enable': True},
        logging={'enable': True},
        llm={'enable': True},
        db={'enable': True},
        vector={'enable': True},
    )


def _install_fakes(monkeypatch, logger, nlp, embedder, db=None, vector=None):
    # 打桩五个私有装配方法，使 RagData 构造不触真实日志后端、spaCy、向量 provider、关系库与向量库。
    monkeypatch.setattr(facade, '_build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, '_build_nlp', lambda *args, **kwargs: nlp)
    monkeypatch.setattr(facade, '_build_embedder', lambda *args, **kwargs: embedder)
    monkeypatch.setattr(facade, '_build_db', lambda *args, **kwargs: db if db is not None else object())
    monkeypatch.setattr(facade, '_build_vector', lambda *args, **kwargs: vector if vector is not None else object())


def _make_app(monkeypatch, settings, logger, nlp=None, parser=None):
    # 构造经 monkeypatch 注入假组件的 RagData。
    # RagData 不再接受 parser 参数；需要替身时打桩 facade.ParserHandler。
    embedder = EmbedderHandler(settings.embedding, logger, model=_FakeModel(dim=settings.embedding.dim))
    _install_fakes(monkeypatch, logger, nlp if nlp is not None else object(), embedder)
    if parser is not None:
        monkeypatch.setattr(facade, 'ParserHandler', lambda *args, **kwargs: parser)
    return facade.RagData(settings)


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
    assert isinstance(result, EmbedderHandler)
    assert result.provider is model


def test_build_embedder_without_model_is_lazy(det_settings, logger):
    result = facade._build_embedder(det_settings.embedding, logger)
    assert isinstance(result, EmbedderHandler)


# ---------------- RagData 构造 ----------------


def test_ragdata_builds_components_from_settings(monkeypatch, det_settings, logger):
    # 构造只接收 settings，其余组件均由 _build_* 派生。
    nlp = object()
    db = object()
    vector = object()
    embedder = EmbedderHandler(det_settings.embedding, logger, model=_FakeModel(dim=det_settings.embedding.dim))
    _install_fakes(monkeypatch, logger, nlp, embedder, db=db, vector=vector)

    app = facade.RagData(det_settings)

    assert app.settings is det_settings
    assert app.logger is logger
    assert app.nlp is nlp
    assert app.embedder is embedder
    assert app.db is db
    assert app.vector is vector
    assert isinstance(app.parser, ParserHandler)


def test_ragdata_passes_section_settings_to_builders(monkeypatch, det_settings, logger):
    # 五个装配方法分别接收 logging、parsing、embedding、db、vector 分区。
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

    def fake_db(db_settings, log):
        captured['db'] = (db_settings, log)
        return object()

    def fake_vector(vector_settings, log):
        captured['vector'] = (vector_settings, log)
        return object()

    monkeypatch.setattr(facade, '_build_logger', fake_logger)
    monkeypatch.setattr(facade, '_build_nlp', fake_nlp)
    monkeypatch.setattr(facade, '_build_embedder', fake_embedder)
    monkeypatch.setattr(facade, '_build_db', fake_db)
    monkeypatch.setattr(facade, '_build_vector', fake_vector)

    facade.RagData(det_settings)

    assert captured['logger'] is det_settings.logging
    assert captured['nlp'] == (det_settings.parsing, logger)
    assert captured['embedder'] == (det_settings.embedding, logger)
    assert captured['db'] == (det_settings.db, logger)
    assert captured['vector'] == (det_settings.vector, logger)


def test_ragdata_uses_patched_parser_handler(monkeypatch, det_settings, logger):
    parser = _FakeParser()
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    assert app.parser is parser


# ---------------- parser_handler ----------------


def test_parser_handler_returns_patched_parser(monkeypatch, det_settings, logger):
    parser = _FakeParser()
    app = _make_app(monkeypatch, det_settings, logger, parser=parser)
    assert app.parser_handler is parser


def test_parser_handler_defaults_to_real_handler(monkeypatch, det_settings, logger):
    app = _make_app(monkeypatch, det_settings, logger)
    assert isinstance(app.parser_handler, ParserHandler)


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
    for name in ('parser_handler', 'parse_document', 'chunk_document'):
        assert hasattr(facade.RagData, name)


def test_facade_all_symbols():
    assert set(facade.__all__) == {'RagData'}



# ---------------- _build_db ----------------


class _FakeDB:
    # 受控数据库替身：记录被装配的配置与日志适配器。
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger


def _install_fake_db(monkeypatch):
    # 打桩 facade.build_database，捕获 DBSettings 与 logger。
    captured = {}

    def fake_build_database(settings, logger=None):
        captured['settings'] = settings
        captured['logger'] = logger
        return _FakeDB(settings, logger)

    monkeypatch.setattr(facade, 'build_database', fake_build_database)
    return captured


def test_build_db_delegates_settings(monkeypatch, det_settings):
    captured = _install_fake_db(monkeypatch)
    result = facade._build_db(det_settings.db, object())
    assert isinstance(result, _FakeDB)
    assert captured['settings'] is det_settings.db


def test_build_db_sqlite_uses_configured_path(monkeypatch, det_settings):
    captured = _install_fake_db(monkeypatch)
    db = det_settings.db.model_copy(update={
        'backend': 'sqlite',
        'sqlite': det_settings.db.sqlite.model_copy(update={'path': 'store.db'}),
    })
    facade._build_db(db, object())
    assert captured['settings'].backend == 'sqlite'
    assert captured['settings'].sqlite.path == 'store.db'


def test_build_db_passes_logger(monkeypatch, det_settings):
    captured = _install_fake_db(monkeypatch)
    sentinel = object()
    facade._build_db(det_settings.db, sentinel)
    assert captured['logger'] is sentinel



def test_build_db_sqlite_requires_path(det_settings):
    # 未配置 sqlite.path 时由装配层抛出明确配置错误。
    with pytest.raises(ConfigError):
        facade._build_db(det_settings.db, object())


def test_build_db_reads_sqlite_path_from_env(monkeypatch, tmp_path):
    captured = _install_fake_db(monkeypatch)
    path = str(tmp_path / 'env.db')
    settings = Settings.load_env({'RAG_DB__BACKEND': 'sqlite', 'RAG_DB__ENABLE': True,'RAG_DB__SQLITE__PATH': path})
    facade._build_db(settings.db, object())
    # print("DEBUG captured:", captured)

    assert captured['settings'].sqlite.path == path



# ---------------- 构造签名与解析器装配 ----------------


def test_ragdata_signature_has_no_parser_param():
    import inspect

    parameters = inspect.signature(facade.RagData.__init__).parameters
    assert 'parser' not in parameters
    assert list(parameters) == ['self', 'settings']


def test_ragdata_builds_parser_handler_with_parsing_limits(monkeypatch, det_settings, logger):
    captured = {}

    def fake_handler(**kwargs):
        captured.update(kwargs)
        return _FakeParser()

    _install_fakes(monkeypatch, logger, object(), object())
    monkeypatch.setattr(facade, 'ParserHandler', fake_handler)
    facade.RagData(det_settings)
    assert captured == {
        'default_max_chars': det_settings.parsing.max_chars,
        'default_safe_max_chars': det_settings.parsing.safe_max_chars,
    }

# ---------------- embedding 模块命名 ----------------


def test_embedding_modules_renamed():
    import rag_data.embedding.embedder_handler as handler_module
    import rag_data.embedding.openai_embedder as openai_module
    import rag_data.embedding.qwen_embedder as qwen_module
    import rag_data.embedding.registry_embedder as registry_module

    assert handler_module.__name__.endswith('.embedder_handler')
    assert registry_module.__name__.endswith('.registry_embedder')
    assert openai_module.__name__.endswith('.openai_embedder')
    assert qwen_module.__name__.endswith('.qwen_embedder')


def test_embedding_exports_handler_only():
    import rag_data.embedding as embedding_package

    assert 'EmbedderHandler' in embedding_package.__all__
    assert 'Embedder' not in embedding_package.__all__
    assert not hasattr(embedding_package, 'Embedder')


def test_embedding_old_module_paths_are_gone():
    import importlib

    for path in (
        'rag_data.embedding.embedder',
        'rag_data.embedding.registry',
        'rag_data.embedding.openai_provider',
        'rag_data.embedding.qwen_provider',
        'rag_data.embedding.registry_embedding',
    ):
        with pytest.raises(ImportError):
            importlib.import_module(path)


def test_embedding_registry_has_no_factory_api():
    import rag_data.embedding.registry_embedder as registry_module

    assert not hasattr(registry_module, 'create_embedding_provider')
    assert not hasattr(registry_module, 'create')


# ---------------- _build_vector ----------------


def test_build_vector_delegates_settings(monkeypatch, det_settings):
    captured = {}

    def fake_builder(settings, logger=None, **kwargs):
        captured['settings'] = settings
        captured['logger'] = logger
        return 'ENGINE'

    monkeypatch.setattr(facade, 'build_vector_engine', fake_builder)
    result = facade._build_vector(det_settings.vector, 'LOG')
    assert result == 'ENGINE'
    assert captured['settings'] is det_settings.vector
    assert captured['logger'] == 'LOG'




# ---------------- enable 门控 ----------------


def test_builders_return_none_when_disabled():
    '''各分区默认关闭时不初始化任何组件。'''
    settings = Settings.load_mapping()
    assert facade._build_logger(settings.logging) is not None
    assert facade._build_nlp(settings.parsing, None) is None
    assert facade._build_embedder(settings.embedding, None) is None
    assert facade._build_db(settings.db, None) is None
    assert facade._build_vector(settings.vector, None) is None


def test_ragdata_skips_disabled_sections():
    '''默认配置下 RagData 只装配解析器，其余组件均为 None。'''
    app = facade.RagData(Settings.load_mapping())
    assert app.logger is not None
    assert app.nlp is None
    assert app.embedder is None
    assert app.db is None
    assert app.vector is None
    assert isinstance(app.parser, ParserHandler)


