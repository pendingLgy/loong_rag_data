# 流程门面的单元测试：验证各流程装配方法与一站式调用。

import os

import rag_data
from fake_store import FakeStore
from rag_data.storage.milvus_store import MilvusVectorStore
from rag_data import (
    RagData,
    Settings,
    build_embedder,
    build_logger,
    build_nlp,
    build_pipeline,
    build_settings,
    build_store,
    ingest,
    init_collection,
)


def _settings(**overrides):
    return Settings(**overrides)


class _FakeModel:
    def __init__(self, dim):
        self._dim = dim

    def encode(self, texts):
        return [[0.1] * self._dim for _ in texts]


def _app(settings, logger, model):
    return RagData(
        settings,
        logger=logger,
        nlp=None,
        store=FakeStore(),
        embedder=build_embedder(settings, logger, model=model),
)


def _doc(tmp_path, name, content):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


# ---------- 分流程装配方法 ----------


def test_build_settings_returns_settings():
    assert isinstance(build_settings(), Settings)


def test_build_settings_accepts_hardcoded_overrides():
    assert build_settings(storage={"vector_dim": 8}).storage.vector_dim == 8
    assert build_settings({"chunking": {"max_chars": 180}}).chunking.max_chars == 180


def test_build_logger_returns_adapter(settings):
    assert build_logger(settings) is not None


def test_build_store_defaults_to_milvus(settings, logger):
    store = build_store(settings, logger)
    assert isinstance(store, MilvusVectorStore)


def test_build_nlp_degrades_when_spacy_missing(settings, logger):
    result = build_nlp(settings, logger)
    assert result is None or hasattr(result, "pipe_names")


def test_build_pipeline_returns_pipeline(settings, logger):
    store = build_store(settings, logger)
    embedder = build_embedder(settings, logger, model=_FakeModel(settings.storage.vector_dim))
    pipeline = build_pipeline(settings, store, embedder, logger, nlp=None)
    assert pipeline is not None


# ---------- 一站式门面 ----------


def test_create_assembles_all_flow_components(logger, settings):
    app = RagData(
        settings,
        logger=logger,
        nlp=None,
        store=FakeStore(),
        embedder=build_embedder(settings, logger, model=_FakeModel(settings.storage.vector_dim)),
    )
    assert isinstance(app.settings, Settings)
    assert isinstance(app.store, FakeStore)
    assert app.pipeline is not None


def test_init_collection_is_idempotent(settings, logger):
    app = _app(settings, logger, _FakeModel(settings.storage.vector_dim))
    app.init_collection()
    app.init_collection()


def test_ingest_file_flow(tmp_path, settings, logger):
    app = _app(settings, logger, _FakeModel(settings.storage.vector_dim))
    path = _doc(tmp_path, "a.md", "第一句。第二句。")
    written = app.ingest_file(path)
    assert written > 0
    assert app.store.count() == written


def test_ingest_paths_flow(tmp_path, settings, logger):
    app = _app(settings, logger, _FakeModel(settings.storage.vector_dim))
    first = _doc(tmp_path, "a.md", "甲甲。乙乙。")
    second = _doc(tmp_path, "b.md", "丙丙。丁丁。")
    written = app.ingest([first, second])
    assert written > 0
    assert app.store.count() == written


def test_context_manager_closes_store(settings, logger):
    with _app(settings, logger, _FakeModel(settings.storage.vector_dim)) as app:
        assert isinstance(app.store, FakeStore)


def test_one_shot_ingest(tmp_path, monkeypatch):
    path = _doc(tmp_path, "one.md", "一键导入内容。")
    settings = Settings()
    logger = build_logger(settings)
    app_kwargs = {
        "logger": logger,
        "nlp": None,
        "store": FakeStore(),
        "embedder": build_embedder(settings, logger, model=_FakeModel(settings.storage.vector_dim)),
    }
    written = ingest([path], None, **app_kwargs)
    assert written > 0


def test_one_shot_init_collection():
    settings = Settings()
    logger = build_logger(settings)
    init_collection(None, logger=logger, nlp=None, store=FakeStore())


def test_facade_is_exported_from_package_root():
    assert rag_data.RagData is RagData
    for name in rag_data.facade.__all__:
        assert hasattr(rag_data, name), name
