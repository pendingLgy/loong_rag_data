# 流程门面（facade）单元测试：装配方法、RagData 逐流程调用与一键入口。
#
# 用假向量模型与假句柄隔离网络与 spaCy；默认装配分支用 monkeypatch 隔离。

import pytest

from rag_data import facade
from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.exceptions import ConfigError, EmbeddingError, ParseError
from rag_data.logging.base import LoggerAdapter


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


class _Sent:
    def __init__(self, text):
        self.text = text


class _Doc:
    def __init__(self, sentences):
        self.sents = [_Sent(sentence) for sentence in sentences]


def _handle(sentences):
    # 返回固定句子序列的假句柄。
    return lambda text: _Doc(sentences)


def _explode(text):
    raise AssertionError('短文本不应调用句柄')


# ---------------- 夹具与构造 ----------------


@pytest.fixture()
def det_settings():
    # 关闭环境变量，保证配置可复现。
    return Settings.load(use_env=False)


def _make_app(settings, logger, nlp=None, model=None):
    # 构造注入了假模型与假句柄的 RagData，避免真实网络与模型加载。
    model = model if model is not None else _FakeModel(dim=settings.embedding.dim)
    return facade.RagData(
        settings,
        embedder=Embedder(settings, logger, model=model),
        logger=logger,
        nlp=nlp if nlp is not None else object(),
    )


# ---------------- build_settings ----------------


def test_build_settings_returns_settings():
    assert isinstance(facade.build_settings(), Settings)


def test_build_settings_applies_overrides():
    result = facade.build_settings(parsing={'max_chars': 7, 'safe_max_chars': 9}, embedding={'dim': 8})
    assert result.parsing.max_chars == 7
    assert result.parsing.safe_max_chars == 9
    assert result.embedding.dim == 8


def test_build_settings_applies_source_mapping():
    result = facade.build_settings({'embedding': {'provider': 'qwen'}})
    assert result.embedding.provider == 'qwen'


def test_build_settings_rejects_unknown_section():
    with pytest.raises(ConfigError):
        facade.build_settings(bogus={'x': 1})


def test_build_settings_rejects_non_mapping_source():
    with pytest.raises(ConfigError):
        facade.build_settings('config.yaml')


# ---------------- build_logger ----------------


def test_build_logger_returns_adapter(det_settings):
    assert isinstance(facade.build_logger(det_settings), LoggerAdapter)


# ---------------- build_nlp ----------------


def test_build_nlp_delegates_model_and_logger(monkeypatch, det_settings, logger):
    captured = {}
    sentinel = object()

    def fake_load(model_name, log):
        captured['model'] = model_name
        captured['logger'] = log
        return sentinel

    monkeypatch.setattr(facade, 'load_nlp', fake_load)
    assert facade.build_nlp(det_settings, logger) is sentinel
    assert captured['model'] == det_settings.parsing.spacy_model
    assert captured['logger'] is logger


def test_build_nlp_returns_handle_or_none(det_settings, logger):
    result = facade.build_nlp(det_settings, logger)
    assert result is None or callable(result)


# ---------------- build_embedder ----------------


def test_build_embedder_injected_model(det_settings, logger):
    model = _FakeModel(dim=det_settings.embedding.dim)
    result = facade.build_embedder(det_settings, logger, model=model)
    assert isinstance(result, Embedder)
    assert result.provider is model


def test_build_embedder_without_model_is_lazy(det_settings, logger):
    assert isinstance(facade.build_embedder(det_settings, logger), Embedder)


# ---------------- RagData 构造与 create ----------------


def test_ragdata_keeps_injected_components(det_settings, logger):
    nlp = object()
    model = _FakeModel(dim=det_settings.embedding.dim)
    embedder = Embedder(det_settings, logger, model=model)
    app = facade.RagData(det_settings, embedder=embedder, logger=logger, nlp=nlp)
    assert app.settings is det_settings
    assert app.logger is logger
    assert app.nlp is nlp
    assert app.embedder is embedder


def test_ragdata_defaults_build_components(monkeypatch, det_settings, logger):
    embedder = Embedder(det_settings, logger, model=_FakeModel(dim=det_settings.embedding.dim))
    nlp = object()
    monkeypatch.setattr(facade, 'build_settings', lambda *args, **kwargs: det_settings)
    monkeypatch.setattr(facade, 'build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, 'build_nlp', lambda *args, **kwargs: nlp)
    monkeypatch.setattr(facade, 'build_embedder', lambda *args, **kwargs: embedder)
    app = facade.RagData()
    assert app.settings is det_settings
    assert app.logger is logger
    assert app.nlp is nlp
    assert app.embedder is embedder


def test_create_passes_source_and_overrides(monkeypatch, det_settings, logger):
    captured = {}

    def fake_settings(source=None, **overrides):
        captured['source'] = source
        captured['overrides'] = overrides
        return det_settings

    monkeypatch.setattr(facade, 'build_settings', fake_settings)
    monkeypatch.setattr(facade, 'build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, 'build_nlp', lambda *args, **kwargs: None)
    embedder = Embedder(det_settings, logger, model=_FakeModel(dim=det_settings.embedding.dim))
    app = facade.RagData.create(
        {'logging': {'level': 'DEBUG'}},
        overrides={'parsing': {'max_chars': 7}},
        embedder=embedder,
    )
    assert captured['source'] == {'logging': {'level': 'DEBUG'}}
    assert captured['overrides'] == {'parsing': {'max_chars': 7}}
    assert app.embedder is embedder
    assert app.nlp is None


def test_create_uses_subclass(monkeypatch, det_settings, logger):
    class SubFacade(facade.RagData):
        pass

    monkeypatch.setattr(facade, 'build_settings', lambda *args, **kwargs: det_settings)
    monkeypatch.setattr(facade, 'build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, 'build_nlp', lambda *args, **kwargs: None)
    monkeypatch.setattr(facade, 'build_embedder', lambda *args, **kwargs: Embedder(det_settings, logger))
    assert isinstance(SubFacade.create(), SubFacade)


# ---------------- parse ----------------


def test_parse_reads_file(tmp_path, det_settings, logger):
    path = tmp_path / 'a.txt'
    path.write_text('你好，世界', encoding='utf-8')
    app = _make_app(det_settings, logger)
    assert app.parse(str(path)) == '你好，世界'


def test_parse_missing_file_raises(tmp_path, det_settings, logger):
    app = _make_app(det_settings, logger)
    with pytest.raises(ParseError):
        app.parse(str(tmp_path / 'missing.txt'))


def test_parse_unsupported_suffix_raises(tmp_path, det_settings, logger):
    path = tmp_path / 'a.xyz'
    path.write_text('内容', encoding='utf-8')
    app = _make_app(det_settings, logger)
    with pytest.raises(ParseError):
        app.parse(str(path))


# ---------------- chunk ----------------


def test_chunk_delegates_to_document_chunk(det_settings, logger):
    app = _make_app(det_settings, logger, nlp=_handle(['第一句。', '第二句。']))
    assert app.chunk('a.txt', '第一句。第二句。') == ['第一句。第二句。']


def test_chunk_dispatches_by_suffix(det_settings, logger):
    app = _make_app(det_settings, logger, nlp=_explode)
    assert app.chunk('a.txt', '短文本') == ['短文本']
    assert app.chunk('a.md', '短文本') == ['短文本']


def test_chunk_empty_text_returns_empty(det_settings, logger):
    app = _make_app(det_settings, logger, nlp=_explode)
    assert app.chunk('a.txt', '') == []


def test_chunk_unsupported_suffix_raises(det_settings, logger):
    app = _make_app(det_settings, logger, nlp=_explode)
    with pytest.raises(ParseError):
        app.chunk('a.xyz', '文本')


# ---------------- vectorize ----------------


def test_vectorize_returns_ordered_vectors(det_settings, logger):
    app = _make_app(det_settings, logger)
    result = app.vectorize(['a', 'b'])
    assert len(result) == 2
    assert all(len(vec) == det_settings.embedding.dim for vec in result)
    assert app.embedder.provider.calls == [['a', 'b']]


def test_vectorize_empty_returns_empty(det_settings, logger):
    app = _make_app(det_settings, logger)
    assert app.vectorize([]) == []
    assert app.embedder.provider.calls == []


def test_vectorize_text_returns_single_vector(det_settings, logger):
    app = _make_app(det_settings, logger)
    vector = app.vectorize_text('单独文本')
    assert isinstance(vector, list)
    assert len(vector) == det_settings.embedding.dim


def test_vectorize_splits_by_provider_batch_limit(det_settings, logger):
    model = _FakeModel(dim=det_settings.embedding.dim, max_batch_size=2)
    app = _make_app(det_settings, logger, model=model)
    app.vectorize(['a', 'b', 'c', 'd', 'e'])
    assert [len(batch) for batch in model.calls] == [2, 2, 1]


def test_vectorize_rejects_dim_mismatch(det_settings, logger):
    model = _FakeModel(dim=det_settings.embedding.dim + 1)
    app = _make_app(det_settings, logger, model=model)
    with pytest.raises(EmbeddingError):
        app.vectorize(['x'])


# ---------------- 模块级一键入口 ----------------


def test_module_vectorize_one_shot(monkeypatch, det_settings, logger):
    model = _FakeModel(dim=det_settings.embedding.dim)
    monkeypatch.setattr(facade, 'build_settings', lambda *args, **kwargs: det_settings)
    monkeypatch.setattr(facade, 'build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, 'build_nlp', lambda *args, **kwargs: object())
    embedder = Embedder(det_settings, logger, model=model)
    result = facade.vectorize(['a', 'b'], embedder=embedder)
    assert len(result) == 2
    assert model.calls == [['a', 'b']]


def test_module_vectorize_forwards_source(monkeypatch, det_settings, logger):
    captured = {}

    def fake_settings(source=None, **overrides):
        captured['source'] = source
        return det_settings

    monkeypatch.setattr(facade, 'build_settings', fake_settings)
    monkeypatch.setattr(facade, 'build_logger', lambda *args, **kwargs: logger)
    monkeypatch.setattr(facade, 'build_nlp', lambda *args, **kwargs: None)
    model = _FakeModel(dim=det_settings.embedding.dim)
    monkeypatch.setattr(facade, 'build_embedder', lambda *args, **kwargs: Embedder(det_settings, logger, model=model))
    facade.vectorize(['a'], {'embedding': {'dim': 4}})
    assert captured['source'] == {'embedding': {'dim': 4}}


# ---------------- 契约 ----------------


def test_ragdata_exposes_flow_methods():
    for name in ('parse', 'chunk', 'vectorize', 'vectorize_text'):
        assert callable(getattr(facade.RagData, name))


def test_facade_all_symbols():
    assert set(facade.__all__) == {
        'RagData',
        'build_settings',
        'build_logger',
        'build_nlp',
        'build_embedder',
        'vectorize',
    }


def test_chunk_forwards_parsing_limits(monkeypatch, det_settings, logger):
    # chunk 应把 parsing 分区的长度配置透传给分派函数。
    captured = {}

    def fake_chunk(path, text, nlp=None, logger=None, max_chars=None, safe_max_chars=None):
        captured['max_chars'] = max_chars
        captured['safe_max_chars'] = safe_max_chars
        return ['ok']

    monkeypatch.setattr(facade, 'chunk_document', fake_chunk)
    app = _make_app(det_settings, logger)
    assert app.chunk('a.txt', 'text') == ['ok']
    assert captured['max_chars'] == det_settings.parsing.max_chars
    assert captured['safe_max_chars'] == det_settings.parsing.safe_max_chars
