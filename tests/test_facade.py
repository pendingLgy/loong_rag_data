# 流程门面（facade）单元测试：公开的私有装配方法 _build_*、
# RagData 与 MemoryServer 的构造、logger 透传与流程代理。
#
# 用假向量模型与假解析器隔离网络与 spaCy；默认装配分支由 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import MemorySettings, RagSetting
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ConfigError, ParseError
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


@pytest.fixture()
def memory_settings():
    # 记忆分区独立于 RagSetting：直接构造 MemorySettings，开启四类依赖子分区。
    return MemorySettings(
        enable=True,
        embedding={'enable': True},
        db={'enable': True, 'sqlite': {'path': './test_facade.db'}},
        vector={'enable': True},
        llm={'enable': True},
    )


def _install_fakes(monkeypatch, nlp, embedder):
    # 打桩 RagData 用到的两个私有装配方法，避免触碰真实 spaCy 与向量 provider。
    monkeypatch.setattr(facade, '_build_nlp', lambda *args, **kwargs: nlp)
    monkeypatch.setattr(facade, '_build_embedder', lambda *args, **kwargs: embedder)


def _install_memory_fakes(monkeypatch):
    # 打桩 MemoryServer 的四类组件装配，避免建立真实数据库与向量库连接。
    monkeypatch.setattr(facade, '_build_embedder', lambda settings: 'EMB')
    monkeypatch.setattr(facade, '_build_llm', lambda settings: 'LLM')
    monkeypatch.setattr(facade, '_build_db', lambda settings: 'DB')
    monkeypatch.setattr(facade, '_build_vector', lambda settings: 'VEC')


def _make_app(monkeypatch, settings, nlp=None, parser=None):
    # 构造经 monkeypatch 注入假组件的 RagData；替换解析器时打桩 facade.ParserHandler。
    embedder = EmbedderHandler(settings.embedding, model=_FakeModel(dim=settings.embedding.dim))
    _install_fakes(monkeypatch, nlp if nlp is not None else object(), embedder)
    if parser is not None:
        monkeypatch.setattr(facade, 'ParserHandler', lambda *args, **kwargs: parser)
    return facade.RagData(settings)


# ---------------- _build_nlp ----------------


def test_build_nlp_delegates_model(monkeypatch, det_settings):
    captured = {}
    sentinel = object()

    def fake_load(model_name):
        captured['model'] = model_name
        return sentinel

    monkeypatch.setattr(facade, 'load_nlp', fake_load)
    assert facade._build_nlp(det_settings.parsing) is sentinel
    assert captured['model'] == det_settings.parsing.spacy_model


def test_build_nlp_returns_handle_or_none(det_settings):
    result = facade._build_nlp(det_settings.parsing)
    assert result is None or callable(result)


# ---------------- _build_embedder ----------------


def test_build_embedder_injected_model(det_settings):
    model = _FakeModel(dim=det_settings.embedding.dim)
    result = facade._build_embedder(det_settings.embedding, model=model)
    assert isinstance(result, EmbedderHandler)
    assert result.provider is model


def test_build_embedder_without_model_is_lazy(det_settings):
    result = facade._build_embedder(det_settings.embedding)
    assert isinstance(result, EmbedderHandler)


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


def test_facade_all_symbols():
    assert set(facade.__all__) == {'RagData', 'MemoryServer'}


# ---------------- _build_db ----------------


class _FakeDB:
    # 受控数据库替身：记录被装配的配置与日志适配器。
    def __init__(self, settings):
        self.settings = settings


def _install_fake_db(monkeypatch):
    # 打桩 facade.build_database，捕获 DBSettings。
    captured = {}

    def fake_build_database(settings):
        captured['settings'] = settings
        return _FakeDB(settings)

    monkeypatch.setattr(facade, 'build_database', fake_build_database)
    return captured


def test_build_db_delegates_settings(monkeypatch, memory_settings):
    captured = _install_fake_db(monkeypatch)
    result = facade._build_db(memory_settings.db)
    assert isinstance(result, _FakeDB)
    assert captured['settings'] is memory_settings.db


def test_build_db_sqlite_uses_configured_path(monkeypatch, memory_settings):
    captured = _install_fake_db(monkeypatch)
    db = memory_settings.db.model_copy(update={
        'backend': 'sqlite',
        'sqlite': memory_settings.db.sqlite.model_copy(update={'path': 'store.db'}),
    })
    facade._build_db(db)
    assert captured['settings'].backend == 'sqlite'
    assert captured['settings'].sqlite.path == 'store.db'


def test_build_db_sqlite_requires_path():
    # 未配置 sqlite.path 时由装配层抛出明确配置错误。
    settings = MemorySettings(db={'enable': True}).db
    with pytest.raises(ConfigError):
        facade._build_db(settings)


def test_build_db_reads_sqlite_path_from_env(monkeypatch, tmp_path):
    captured = _install_fake_db(monkeypatch)
    path = str(tmp_path / 'env.db')
    monkeypatch.setenv('RAG_MEMORY__DB__BACKEND', 'sqlite')
    monkeypatch.setenv('RAG_MEMORY__DB__ENABLE', 'true')
    monkeypatch.setenv('RAG_MEMORY__DB__SQLITE__PATH', path)
    facade._build_db(MemorySettings().db)
    assert captured['settings'].sqlite.path == path


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


def test_build_vector_delegates_settings(monkeypatch, memory_settings):
    captured = {}

    def fake_builder(settings, **kwargs):
        captured['settings'] = settings
        return 'ENGINE'

    monkeypatch.setattr(facade, 'build_vector_engine', fake_builder)
    result = facade._build_vector(memory_settings.vector)
    assert result == 'ENGINE'
    assert captured['settings'] is memory_settings.vector


# ---------------- enable 门控 ----------------


def test_builders_return_none_when_disabled():
    # 各分区默认关闭时不初始化任何组件。
    settings = RagSetting.load_mapping()
    assert facade._build_nlp(settings.parsing) is None
    assert facade._build_embedder(settings.embedding) is None
    assert facade._build_db(MemorySettings().db) is None
    assert facade._build_vector(MemorySettings().vector) is None


def test_build_memory_returns_none_when_disabled():
    # memory.enable 为 False 时不构造 MemoryHandler。
    assert facade._build_memory(MemorySettings(), None, None, None, None) is None


def test_ragdata_skips_disabled_sections():
    # 默认配置下 RagData 只装配解析器，其余组件均为 None。
    app = facade.RagData(RagSetting.load_mapping())
    assert app.nlp is None
    assert app.embedder is None
    assert isinstance(app.parser, ParserHandler)


# ---------------- MemoryServer ----------------


class _FakeMemoryHandler:
    # 受控记忆处理器替身：记录四条流程的入参并返回固定结果。
    def __init__(self):
        self.calls = []

    def add(self, messages, **kwargs):
        self.calls.append(('add', messages, kwargs))
        return 'ADD'

    def search(self, query, **kwargs):
        self.calls.append(('search', query, kwargs))
        return 'SEARCH'

    def update(self, memory_id, **kwargs):
        self.calls.append(('update', memory_id, kwargs))
        return 'UPDATE'

    def delete(self, memory_id):
        self.calls.append(('delete', memory_id, {}))
        return 'DELETE'


def test_memory_server_builds_all_components(monkeypatch, memory_settings):
    # 四类依赖由门面内部按分区装配。
    _install_memory_fakes(monkeypatch)
    captured = {}

    def fake_memory(settings, embedder, llm, db, vector):
        captured['args'] = (embedder, llm, db, vector)
        return 'HANDLER'

    monkeypatch.setattr(facade, '_build_memory', fake_memory)
    server = facade.MemoryServer(memory_settings)

    assert server.settings is memory_settings
    assert server.embedder == 'EMB'
    assert server.llm == 'LLM'
    assert server.db == 'DB'
    assert server.vector == 'VEC'
    assert server.memory == 'HANDLER'
    assert captured['args'] == ('EMB', 'LLM', 'DB', 'VEC')


def test_memory_server_passes_section_settings(monkeypatch, memory_settings):
    # 各装配方法收到的应是 memory 分区下对应的子设置对象。
    captured = {}

    def record(name):
        def inner(settings):
            captured[name] = settings
            return name
        return inner

    monkeypatch.setattr(facade, '_build_embedder', lambda settings: 'EMB')
    monkeypatch.setattr(facade, '_build_llm', record('LLM'))
    monkeypatch.setattr(facade, '_build_db', record('DB'))
    monkeypatch.setattr(facade, '_build_vector', record('VEC'))
    monkeypatch.setattr(facade, '_build_memory', lambda *args, **kwargs: None)
    facade.MemoryServer(memory_settings)

    assert captured['LLM'] is memory_settings.llm
    assert captured['DB'] is memory_settings.db
    assert captured['VEC'] is memory_settings.vector


def test_memory_server_delegates_flows(monkeypatch, memory_settings):
    # 四条流程逐条转发给 MemoryHandler，服务层不做业务加工。
    handler = _FakeMemoryHandler()
    _install_memory_fakes(monkeypatch)
    monkeypatch.setattr(facade, '_build_memory', lambda *args, **kwargs: handler)
    server = facade.MemoryServer(memory_settings)

    assert server.add('hi', user_id='u1') == 'ADD'
    assert server.search('q', top_k=3) == 'SEARCH'
    assert server.update(1, text='t') == 'UPDATE'
    assert server.delete(1) == 'DELETE'
    assert [call[0] for call in handler.calls] == ['add', 'search', 'update', 'delete']


def test_memory_server_requires_enabled_memory():
    # 默认配置关闭记忆层：各组件为 None，四条流程统一抛 ConfigError。
    server = facade.MemoryServer(MemorySettings())
    assert server.memory is None
    assert server.embedder is None
    assert server.db is None
    assert server.vector is None
    assert server.llm is None
    for call in (lambda: server.add('hi'), lambda: server.search('q'), lambda: server.update(1), lambda: server.delete(1)):
        with pytest.raises(ConfigError):
            call()


def test_memory_server_signature_accepts_settings():
    # 构造签名只保留 settings，组件装配全部下沉到门面内部。
    import inspect

    parameters = list(inspect.signature(facade.MemoryServer.__init__).parameters)
    assert parameters == ['self', 'settings']

