# 公共入口的守护测试：确保外部调用只需引入 rag_data。

import os

import rag_data
from rag_data import (
    Embedder,
    IngestionPipeline,
    Settings,
    configure_logging,
    get_logger,
)


def test_version_is_exported():
    assert isinstance(rag_data.__version__, str)
    assert rag_data.__version__.count(".") == 2


def test_all_names_are_resolvable():
    missing = [name for name in rag_data.__all__ if not hasattr(rag_data, name)]
    assert missing == []


def test_all_has_no_duplicates():
    assert len(rag_data.__all__) == len(set(rag_data.__all__))


def test_core_symbols_available_from_package_root():
    assert Settings is rag_data.Settings
    assert IngestionPipeline is rag_data.IngestionPipeline
    assert callable(configure_logging) and callable(get_logger)


def test_public_api_end_to_end(tmp_path, logger):
    path = os.path.join(str(tmp_path), "doc.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("第一句。第二句。第三句。")

    settings = Settings(chunking={"overlap_sents": 0})

    class _FakeModel:
        def encode(self, texts):
            return [[0.1] * settings.embedding.dim for _ in texts]

    embedder = Embedder(settings, logger, model=_FakeModel())
    pipeline = IngestionPipeline(embedder, settings, logger)
    embedded = pipeline.run([path])

    assert len(embedded) > 0
    assert all(len(item.vector) == settings.embedding.dim for item in embedded)


def test_storage_layer_is_removed():
    # 存储层已整体移除，公共入口不应再暴露相关符号。
    for name in ("build_store", "init_collection", "MilvusRecord", "StorageSettings", "register_store"):
        assert not hasattr(rag_data, name)


def test_exceptions_are_exported():
    for name in ("RagDataError", "ConfigError", "DataError", "EmbeddingError"):
        assert issubclass(getattr(rag_data, name), Exception)
