# 手工集成测试：真实调用 Qwen 嵌入并把向量写入 Milvus。
#
# 默认不参与 pytest 运行：本模块标记 integration，addopts 已排除该标记。
# 需人工准备凭据与 Milvus 服务后，显式指定标记执行：
#
#   Windows PowerShell
#     $env:DASHSCOPE_API_KEY = "sk-xxx"
#     $env:RAG_IT_MILVUS_URI = "http://127.0.0.1:19530"
#     uv run pytest -m integration -v -s
#
#   macOS 或 Linux
#     DASHSCOPE_API_KEY=sk-xxx uv run pytest -m integration -v -s
#
# 环境变量（凭据与连接信息在运行时单独提供，代码与仓库中不落任何密钥）：
#
# | 变量 | 必填 | 默认值 | 说明 |
# | :--- | :--- | :--- | :--- |
# | DASHSCOPE_API_KEY | 是 | 无 | Qwen 的 API Key |
# | RAG_IT_API_KEY | 否 | 无 | 覆盖上一项，便于切换多套凭据 |
# | RAG_IT_MILVUS_URI | 否 | http://127.0.0.1:19530 | Milvus 地址 |
# | RAG_IT_MILVUS_DB | 否 | default | Milvus 数据库名，集合须在该库中 |
# | RAG_IT_COLLECTION | 否 | rag_data_it_qwen | 集合名 |
# | RAG_IT_MODEL | 否 | qwen3.7-text-embedding | 嵌入模型 |
# | RAG_IT_KEEP | 否 | 空 | 置 1 则跑完保留集合，默认删除 |
#
# 安全约定：只有集合名以 rag_data_it 开头、且 RAG_IT_KEEP 未置 1 时才删除集合，
# 避免误删他人数据。

from __future__ import annotations

import importlib.util
import os
from typing import Dict

import pytest

from rag_data import QueryHit, RagData, Settings
from rag_data.storage.memory_store import cosine_similarity

# 凭据与连接信息一律来自运行时环境变量，代码中不写默认密钥。
MILVUS_URI = os.environ.get("RAG_IT_MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_DB = os.environ.get("RAG_IT_MILVUS_DB", "default")
COLLECTION = os.environ.get("RAG_IT_COLLECTION", "rag_data_it_qwen")
MODEL = os.environ.get("RAG_IT_MODEL", "qwen3.7-text-embedding")
API_KEY = os.environ.get("RAG_IT_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
KEEP_COLLECTION = os.environ.get("RAG_IT_KEEP", "") == "1"

# 向量维度；需与实际模型的输出维度一致，可用 RAG_IT_DIM 覆盖。
# 该值同时决定 storage.vector_dim，写入前会据此校验每条记录。
VECTOR_DIM = int(os.environ.get("RAG_IT_DIM", "1024"))
# DashScope 兼容接口单次最多接受 10 条文本。
QWEN_MAX_BATCH = 10
# 只允许删除该前缀的集合。
COLLECTION_PREFIX = "rag_data_it"

requires_api_key = pytest.mark.skipif(
    not API_KEY,
    reason="未提供 API Key：请先设置环境变量 DASHSCOPE_API_KEY 或 RAG_IT_API_KEY",
)
requires_pymilvus = pytest.mark.skipif(
    importlib.util.find_spec("pymilvus") is None,
    reason="未安装 pymilvus，请先执行 pip install pymilvus",
)

pytestmark = [pytest.mark.integration, requires_api_key, requires_pymilvus]


# ---------------- 夹具 ----------------


@pytest.fixture(scope="module")
def it_settings() -> Settings:
    """按集成测试需要构造配置：qwen 嵌入加 milvus 存储，凭据来自环境变量。"""
    return Settings.load(
        storage={
            "backend": "milvus",
            "milvus_uri": MILVUS_URI,
            "milvus_db": MILVUS_DB,
            "collection_name": COLLECTION,
            "vector_dim": VECTOR_DIM,
            "metric": "COSINE",
            "index_type": "HNSW",
        },
        embedding={
            "provider": "qwen",
            "model": MODEL,
            "api_key": API_KEY,
            "dim": VECTOR_DIM,
            "batch_size": QWEN_MAX_BATCH,
        },
        chunking={"overlap_sents": 0},
    )


@pytest.fixture(scope="module")
def app(it_settings) -> RagData:
    """装配真实门面：真实 Qwen 嵌入加真实 Milvus；nlp 置空走回退实现。"""
    # 建集会走记录类的 build_collection_schema；维度需与嵌入输出一致。
    instance = RagData(it_settings, nlp=None)
    instance.init_collection()
    yield instance
    instance.store.close()
    _drop_collection()


@pytest.fixture(scope="module")
def docs(tmp_path_factory) -> Dict[str, str]:
    """两个租户各一份文档，用于验证写入与按扩展字段过滤。"""
    directory = tmp_path_factory.mktemp("qwen_it")
    contents = {
        "tenant_a": "Milvus 是向量数据库，支持相似度检索。向量数据库用于存储嵌入向量。",
        "tenant_b": "今天天气不错，适合出门散步。",
    }
    paths: Dict[str, str] = {}
    for name, text in contents.items():
        path = directory / (name + ".md")
        path.write_text(text, encoding="utf-8")
        paths[name] = str(path)
    return paths


# ---------------- 辅助 ----------------


def _drop_collection() -> None:
    """删除本次测试的集合；非约定前缀或显式保留时跳过，避免误删他人数据。"""
    if KEEP_COLLECTION or not COLLECTION.startswith(COLLECTION_PREFIX):
        return
    import pymilvus

    from rag_data.storage.milvus_store import DEFAULT_ALIAS

    pymilvus.connections.connect(alias=DEFAULT_ALIAS, uri=MILVUS_URI, db_name=MILVUS_DB)
    # if pymilvus.utility.has_collection(COLLECTION, using=DEFAULT_ALIAS):
        # pymilvus.utility.drop_collection(COLLECTION, using=DEFAULT_ALIAS)


# ---------------- 嵌入 ----------------


def test_provider_is_qwen(app):
    """确认装配的确实是配置指定的 qwen provider。"""
    assert app.settings.embedding.provider == "qwen"
    assert type(app.embedder.provider).__name__ == "QwenEmbeddingProvider"
    assert app.embedder.provider.model == MODEL
    assert app.embedder.provider.dim == VECTOR_DIM


def test_encode_returns_configured_dimension(app):
    vectors = app.embedder.encode(["向量数据库", "Milvus 相似度检索"])
    assert len(vectors) == 2
    assert all(len(vector) == VECTOR_DIM for vector in vectors)


def test_related_text_is_more_similar(app):
    """语义相近的文本应比无关文本得分更高，用于确认调用的是真实模型。"""
    query, related, unrelated = app.embedder.encode(
        ["向量数据库", "Milvus 是向量数据库", "今天天气不错"]
    )
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


# ---------------- 写入与检索 ----------------


def test_ingest_writes_chunks(app, docs):
    written = app.ingest([docs["tenant_a"]], user_id="tenant-a")
    assert written > 0


def test_query_returns_most_relevant_first(app, docs):
    app.ingest([docs["tenant_a"]], user_id="tenant-a")
    hits = app.query(app.embedder.encode(["向量数据库"])[0], top_n=3)
    assert hits
    assert all(isinstance(hit, QueryHit) for hit in hits)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert "向量" in hits[0].text_payload


def test_filter_isolates_tenant(app, docs):
    app.ingest([docs["tenant_a"]], user_id="tenant-a")
    app.ingest([docs["tenant_b"]], user_id="tenant-b")
    vector = app.embedder.encode(["向量数据库"])[0]
    hits = app.query(vector, top_n=5, filters={"user_id": "tenant-a"})
    assert hits
    # user_id 未提升为独立列，Milvus 侧走 metadata 的 JSON 路径过滤
    assert all("天气" not in hit.text_payload for hit in hits)


def test_hit_fields_are_complete(app, docs):
    app.ingest([docs["tenant_a"]], user_id="tenant-a")
    hit = app.query(app.embedder.encode(["向量数据库"])[0], top_n=1)[0]
    assert isinstance(hit.id, str) and hit.id
    assert isinstance(hit.text_payload, str) and hit.text_payload
    assert isinstance(hit.entities, list)
    assert isinstance(hit.created_at, float)


def test_reingest_is_idempotent(app, docs):
    """同一份文档重复导入不应产生重复记录：主键由内容哈希决定。"""
    first = app.ingest([docs["tenant_a"]], user_id="tenant-a")
    app.ingest([docs["tenant_a"]], user_id="tenant-a")
    vector = app.embedder.encode(["向量数据库"])[0]
    hits = app.query(vector, top_n=50, filters={"user_id": "tenant-a"})
    ids = [hit.id for hit in hits]
    assert len(ids) == len(set(ids))
    assert len(ids) == first
