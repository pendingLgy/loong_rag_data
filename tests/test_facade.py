# 流程门面的单元测试：验证各流程装配方法与一站式调用。

import os

import rag_data
from rag_data import (
    RagData,
    Settings,
    build_embedder,
    build_logger,
    build_nlp,
    build_pipeline,
    build_settings,
    ingest,
    vectorize,
)
from rag_data.models import DocumentChunk


class _FakeModel:
    def __init__(self, dim):
        self._dim = dim

    def encode(self, texts):
        return [[0.1] * self._dim for _ in texts]


def _app(settings, logger):
    return RagData(
        settings,
        logger=logger,
        nlp=None,
        embedder=build_embedder(settings, logger, model=_FakeModel(settings.embedding.dim)),
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
    assert build_settings(embedding={"dim": 8}).embedding.dim == 8
    assert build_settings({"chunking": {"max_chars": 180}}).chunking.max_chars == 180


def test_build_logger_returns_adapter(settings):
    assert build_logger(settings) is not None


def test_build_nlp_degrades_when_spacy_missing(settings, logger):
    result = build_nlp(settings, logger)
    assert result is None or hasattr(result, "pipe_names")


def test_build_embedder_uses_injected_model(logger):
    settings = Settings(embedding={"dim": 8})
    embedder = build_embedder(settings, logger, model=_FakeModel(8))
    assert embedder.encode(["a"])[0] == [0.1] * 8


def test_build_pipeline_returns_pipeline(settings, logger):
    embedder = build_embedder(settings, logger, model=_FakeModel(settings.embedding.dim))
    pipeline = build_pipeline(settings, embedder, logger, nlp=None)
    assert pipeline is not None


# ---------- 一站式门面 ----------


def test_create_assembles_all_flow_components(logger, settings):
    app = _app(settings, logger)
    assert isinstance(app.settings, Settings)
    assert app.pipeline is not None
    assert app.embedder is not None


def test_ingest_file_flow(tmp_path, settings, logger):
    app = _app(settings, logger)
    path = _doc(tmp_path, "a.md", "第一句。第二句。")
    embedded = app.ingest_file(path)
    assert len(embedded) > 0
    assert all(isinstance(item, DocumentChunk) for item in embedded)


def test_ingest_paths_flow(tmp_path, settings, logger):
    app = _app(settings, logger)
    first = _doc(tmp_path, "a.md", "甲甲。乙乙。")
    second = _doc(tmp_path, "b.md", "丙丙。丁丁。")
    embedded = app.ingest([first, second])
    assert len(embedded) > 0
    assert {item.source_path for item in embedded} == {first, second}


def test_one_shot_ingest(tmp_path):
    path = _doc(tmp_path, "one.md", "一键导入内容。")
    settings = Settings()
    logger = build_logger(settings)
    embedded = ingest(
        [path],
        None,
        logger=logger,
        nlp=None,
        embedder=build_embedder(settings, logger, model=_FakeModel(settings.embedding.dim)),
    )
    assert len(embedded) > 0


def test_vectorize_texts_returns_vectors_in_order(settings, logger):
    app = _app(settings, logger)
    vectors = app.vectorize(["甲", "乙", "丙"])
    assert len(vectors) == 3
    assert all(len(vector) == settings.embedding.dim for vector in vectors)


def test_vectorize_text_returns_single_vector(settings, logger):
    app = _app(settings, logger)
    vector = app.vectorize_text("一段文本")
    assert len(vector) == settings.embedding.dim


def test_vectorize_empty_returns_empty(settings, logger):
    assert _app(settings, logger).vectorize([]) == []


def test_one_shot_vectorize():
    settings = Settings()
    logger = build_logger(settings)
    vectors = vectorize(
        ["甲", "乙"],
        None,
        logger=logger,
        nlp=None,
        embedder=build_embedder(settings, logger, model=_FakeModel(settings.embedding.dim)),
    )
    assert len(vectors) == 2


def test_facade_is_exported_from_package_root():
    assert rag_data.RagData is RagData
    for name in rag_data.facade.__all__:
        assert hasattr(rag_data, name), name
