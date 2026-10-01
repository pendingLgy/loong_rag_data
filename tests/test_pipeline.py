# 导入管道端到端测试，使用假存储与假向量化。

import os

from rag_data.config import Settings
from rag_data.ingestion.pipeline import IngestionPipeline
from fake_store import FakeStore


class _FakeEmbedder:
    def __init__(self, dim):
        self._dim = dim

    def encode(self, texts):
        return [[float(len(text))] + [0.0] * (self._dim - 1) for text in texts]


def _pipeline(store, settings, logger):
    embedder = _FakeEmbedder(settings.storage.vector_dim)
    return IngestionPipeline(store, embedder, settings, logger)


def _write(tmp_path, name, content):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def test_ingest_single_file(tmp_path, logger):
    path = _write(tmp_path, "sample.md", "第一句。第二句。第三句。")
    settings = Settings(chunking={"overlap_sents": 0}, embedding={"batch_size": 2})
    store = FakeStore()
    written = _pipeline(store, settings, logger).run([path])
    assert written >= 1
    assert store.count() == written


def test_ingest_produces_multiple_chunks(tmp_path, logger):
    long_text = "这是一个句子。" * 60
    path = _write(tmp_path, "long.md", long_text)
    settings = Settings(chunking={"overlap_sents": 0}, embedding={"batch_size": 2})
    store = FakeStore()
    written = _pipeline(store, settings, logger).run([path])
    assert written > 1


def test_ingest_is_idempotent(tmp_path, logger):
    path = _write(tmp_path, "idem.md", "唯一内容。")
    settings = Settings(embedding={"batch_size": 2})
    store = FakeStore()
    pipeline = _pipeline(store, settings, logger)
    first = pipeline.run([path])
    second = pipeline.run([path])
    assert first == second
    assert store.count() == first


def test_missing_file_is_skipped(tmp_path, logger):
    missing = os.path.join(str(tmp_path), "missing.md")
    settings = Settings()
    store = FakeStore()
    assert _pipeline(store, settings, logger).run([missing]) == 0
    assert store.count() == 0
