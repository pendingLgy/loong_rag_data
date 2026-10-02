# 导入管道测试：解析、切块、实体抽取与向量化，使用假向量化模型。

import os

from rag_data.config import Settings
from rag_data.ingestion.pipeline import IngestionPipeline
from rag_data.models import DocumentChunk


class _FakeEmbedder:
    def __init__(self, dim):
        self._dim = dim

    def encode(self, texts):
        return [[float(len(text))] + [0.0] * (self._dim - 1) for text in texts]


class _RecordingEmbedder:
    """记录每次编码的批次，用于验证分批行为。"""

    def __init__(self, dim):
        self._dim = dim
        self.batches = []

    def encode(self, texts):
        self.batches.append(list(texts))
        return [[0.1] * self._dim for _ in texts]


def _pipeline(settings, logger):
    return IngestionPipeline(_FakeEmbedder(settings.embedding.dim), settings, logger)


def _write(tmp_path, name, content):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def test_ingest_single_file(tmp_path, logger):
    path = _write(tmp_path, "sample.md", "第一句。第二句。第三句。")
    settings = Settings(chunking={"overlap_sents": 0}, embedding={"batch_size": 2})
    embedded = _pipeline(settings, logger).run([path])
    assert len(embedded) >= 1
    assert all(isinstance(item, DocumentChunk) for item in embedded)
    assert all(len(item.vector) == settings.embedding.dim for item in embedded)
    assert embedded[0].source_path == path
    assert embedded[0].user_id == "default"


def test_ingest_produces_multiple_chunks(tmp_path, logger):
    long_text = "这是一个句子。" * 60
    path = _write(tmp_path, "long.md", long_text)
    settings = Settings(chunking={"overlap_sents": 0}, embedding={"batch_size": 2})
    embedded = _pipeline(settings, logger).run([path])
    assert len(embedded) > 1


def test_ingest_is_idempotent(tmp_path, logger):
    path = _write(tmp_path, "idem.md", "唯一内容。")
    settings = Settings(embedding={"batch_size": 2})
    pipeline = _pipeline(settings, logger)
    first = pipeline.run([path])
    second = pipeline.run([path])
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_batching_splits_embedding_calls(tmp_path, logger):
    path = _write(tmp_path, "batch.md", "这是一个句子。" * 200)
    settings = Settings(chunking={"overlap_sents": 0}, embedding={"batch_size": 2})
    embedder = _RecordingEmbedder(settings.embedding.dim)
    embedded = IngestionPipeline(embedder, settings, logger).run([path])
    assert len(embedded) > 2
    assert all(len(batch) <= 2 for batch in embedder.batches)


def test_missing_file_is_skipped(tmp_path, logger):
    missing = os.path.join(str(tmp_path), "missing.md")
    settings = Settings()
    assert _pipeline(settings, logger).run([missing]) == []
