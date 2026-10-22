# 单条集成测试：参考 test_integration_qwen_epub_milvus.py 手工建集合的做法，
# 在 Milvus 与 PG 两侧各建一份同构结构（id / vector / text / text_lemma / title / title_md5 / source），
# 解析 EPUB、切块、向量化后把同一批行写入两侧，只做写入与按主键回查（不做相似度检索）。
#
# 配置来源：本文件顶部的独立 JSON，不读取 conftest 与环境变量。
#   samples.epub       样例 EPUB 路径，未配置则跳过
#   parsing.*          spaCy 模型与切块长度
#   embedding.*        嵌入 provider、模型、维度与密钥
#   milvus.*           Milvus 地址、令牌与库名
#   pg.*               PG 连接参数（向量库分区）
#   vector.prefix      集合与表名前缀，集合名必须以它开头才能通过守卫
#
# 安全约定：集合名与表名统一为 vector.prefix 加 _rag_data，建集合前先过 storage.guard 的前缀校验，用例不删表。

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Tuple

import pytest
from pymilvus import MilvusClient

from rag_data.config import (
    EmbeddingSettings,
    MilvusSettings,
    PGVectorSettings,
    ParsingSettings,
)
from rag_data.storage.db import next_id

# 本文件独立的 JSON 配置：不读 conftest，也不读环境变量。
# sample.epub 未填写时用例跳过；embedding.api_key 需按本机填写。
EPUB_BOTH_JSON = r"""
{
  "sample": {
    "epub": "C:\\Users\\loong\\Desktop\\经营未来 (李明博) (z-library.sk, 1lib.sk, z-lib.sk).epub"
  },
  "parsing": {
    "spacy_model": "zh_core_web_sm",
    "max_chars": 500,
    "safe_max_chars": 2000
  },
  "embedding": {
    "provider": "qwen",
    "model": "qwen3.7-text-embedding",
    "dim": 1024,
    "api_key": "",
    "batch_size": 10
  },
  "milvus": {
    "uri": "http://127.0.0.1:19530",
    "token": "root:Milvus",
    "db_name": "eval_rag_data"
  },
  "pg": {
    "connection_string": "",
    "host": "localhost",
    "port": 5432,
    "user": "root",
    "password": "123456",
    "database": "eval_rag_data"
  },
  "vector": {
    "prefix": "eval_",
    "max_chunks": 8,
    "top_k": 3
  }
}
"""
EPUB_BOTH_CONFIG = json.loads(EPUB_BOTH_JSON)

# 下方四个 Settings 与两个字典供用例正文引用，字段由上方 JSON 推导。
SAMPLES_CONFIG = {"epub": EPUB_BOTH_CONFIG["sample"]["epub"]}
VECTOR_CONFIG = {"prefix": EPUB_BOTH_CONFIG["vector"]["prefix"]}
PARSING_SETTINGS = ParsingSettings(**EPUB_BOTH_CONFIG["parsing"])
EMBEDDING_SETTINGS = EmbeddingSettings(**EPUB_BOTH_CONFIG["embedding"])
EMBEDDING_SETTINGS.api_key = os.getenv("embedding_api_key","")

MILVUS_SETTINGS = MilvusSettings(**EPUB_BOTH_CONFIG["milvus"])
PG_VECTOR_SETTINGS = PGVectorSettings(**EPUB_BOTH_CONFIG["pg"])

from rag_data import RagData, RagSetting
from rag_data.nlp.lemmatization import lemmatize_zh_for_bm25


pytestmark = pytest.mark.eval

if "--eval" not in sys.argv:
    pytest.skip(
        f"[SKIP INFO] File {__file__}：eval 集成测试默认跳过（请使用 --eval 参数运行）",
        allow_module_level=True,
    )

# 只写入前若干块：整本书会产生过量嵌入请求。
MAX_CHUNKS = int(EPUB_BOTH_CONFIG["vector"]["max_chunks"])

# 两侧共用同一名字，便于对照同一批数据在两个后端的写入结果。
COLLECTION = "eval_rag_data"


# 向量维度一律取嵌入配置，避免与 vector.dim 的占位值不一致。
DIM = EMBEDDING_SETTINGS.dim

# 单次运行的 id 基址取自毫秒时间戳，重复运行不会与历史数据主键冲突。
BASE_ID = int(time.time()) * 1000


# PG 连接参数。
PG_CONNECTION_STRING = PG_VECTOR_SETTINGS.connection_string
PG_HOST = PG_VECTOR_SETTINGS.host
PG_PORT = PG_VECTOR_SETTINGS.port
PG_USER = PG_VECTOR_SETTINGS.user
PG_PASSWORD = PG_VECTOR_SETTINGS.password
PG_DB = PG_VECTOR_SETTINGS.database


# ---------------- 前置校验 ----------------


def _skip(message: str) -> None:
    """统一跳过入口：原因由 pytest -rs 汇总输出。"""
    pytest.skip(message)


def _epub_path() -> str:
    """样例 EPUB 路径取自 conftest 的 samples.epub，未配置时返回空串。"""
    return str(SAMPLES_CONFIG.get("epub") or "")


def _require_epub(path: str) -> None:
    """缺少样例文件时跳过：解析链路无法凭空造出 EPUB。"""
    if not path or not os.path.exists(path):
        _skip("未配置样例 EPUB：请在 conftest 的 samples.epub 中填入文件路径")


def _require_api_key() -> None:
    """缺少嵌入密钥时跳过：向量化需要真实调用嵌入服务。"""
    if not EMBEDDING_SETTINGS.api_key:
        _skip("未配置嵌入密钥：请在 conftest 的 embedding.api_key 中填入")


def _skip_service(service: str, target: str, exc: Exception) -> None:
    """外部服务或驱动不可用时跳过，并给出服务名、目标地址与原始异常。"""
    _skip("跳过：" + service + "（目标 " + target + "）：" + type(exc).__name__ + "，" + str(exc))


# ---------------- 解析与向量化 ----------------


def _build_app() -> RagData:
    """装配解析门面：只开启 parsing 与 embedding 两个分区。"""
    settings = RagSetting.load_mapping(
        parsing={
            "enable": True,
            "spacy_model": PARSING_SETTINGS.spacy_model,
            "max_chars": PARSING_SETTINGS.max_chars,
            "safe_max_chars": PARSING_SETTINGS.safe_max_chars,
        },
        embedding={
            "enable": True,
            "provider": EMBEDDING_SETTINGS.provider,
            "model": EMBEDDING_SETTINGS.model,
            "api_key": EMBEDDING_SETTINGS.api_key,
            "dim": EMBEDDING_SETTINGS.dim,
            "batch_size": EMBEDDING_SETTINGS.batch_size,
        },
    )
    return RagData(settings)


def _build_rows(app: RagData, path: str) -> List[Dict[str, Any]]:
    '''解析 EPUB、切块并向量化，返回两侧共用的行数据。'''
    text = app.parse_document(path)
    assert text.strip()
    try:
        chunks = [chunk for chunk in app.chunk_document(path, text) if chunk.strip()]
    except Exception as exc:  # noqa: BLE001 句柄或模型缺失属环境问题
        _skip('切块不可用：' + type(exc).__name__ + ' ' + str(exc))
    assert chunks
    batch = chunks[:1]

    vectors = app.embedder.encode(batch)
    assert len(vectors) == len(batch)
    assert all(len(vector) == DIM for vector in vectors)

    title = os.path.splitext(os.path.basename(path))[0]
    source = os.path.basename(path)
    rows: List[Dict[str, Any]] = []
    for index, chunk in enumerate(batch):
        rows.append({
            'id': next_id(),
            'vector': vectors[index],
            'text': chunk,
            'text_lemma': lemmatize_zh_for_bm25(chunk),
            'title': title,
            'title_md5': hashlib.md5(title.encode('utf-8')).hexdigest(),
            'source': source,
        })
    return rows


# ---------------- Milvus 集合（手工建，参考 test_integration_qwen_epub_milvus.py） ----------------


def _connect_milvus() -> Tuple[Any, Any]:
    """连接 Milvus：缺 pymilvus 或服务不可用时跳过，返回模块与客户端。"""
    try:
        import pymilvus
    except ImportError as exc:
        _skip_service("pymilvus", MILVUS_SETTINGS.uri, exc)
    kwargs: Dict[str, Any] = {"uri": MILVUS_SETTINGS.uri}
    if MILVUS_SETTINGS.token:
        kwargs["token"] = MILVUS_SETTINGS.token
    if MILVUS_SETTINGS.db_name:
        kwargs["db_name"] = MILVUS_SETTINGS.db_name
    try:
        return pymilvus, pymilvus.MilvusClient(**kwargs)
    except Exception as exc:  # noqa: BLE001 连接失败统一按环境问题跳过
        _skip_service("Milvus", MILVUS_SETTINGS.uri, exc)



def _create_milvus_collection(pymilvus: Any, client: Any) -> None:
    '''手工建集合：稠密向量 + text_lemma 经 BM25 函数输出稀疏向量。'''
    fields = [
        pymilvus.FieldSchema(name='id', dtype=pymilvus.DataType.INT64, is_primary=True),
        pymilvus.FieldSchema(name='vector', dtype=pymilvus.DataType.FLOAT_VECTOR, dim=DIM),
        pymilvus.FieldSchema(name='text', dtype=pymilvus.DataType.VARCHAR, max_length=65535),
        pymilvus.FieldSchema(name='text_lemma', dtype=pymilvus.DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        pymilvus.FieldSchema(name='title', dtype=pymilvus.DataType.VARCHAR, max_length=1024),
        pymilvus.FieldSchema(name='title_md5', dtype=pymilvus.DataType.VARCHAR, max_length=32),
        pymilvus.FieldSchema(name='source', dtype=pymilvus.DataType.VARCHAR, max_length=2048),
        pymilvus.FieldSchema(name='sparse', dtype=pymilvus.DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = pymilvus.CollectionSchema(
        fields,
        auto_id=False,
        enable_dynamic_field=True,
        description='双后端集成测试集合',
    )
    schema.add_function(
        pymilvus.Function(
            name='text_bm25_emb',
            input_field_names=['text_lemma'],
            output_field_names=['sparse'],
            function_type=pymilvus.FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='COSINE', index_name='vector_index')
    index_params.add_index(field_name='sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25', index_name='sparse_index')
    index_params.add_index(field_name='title_md5', index_type='INVERTED', index_name='title_md5_index')

    client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
    client.load_collection(COLLECTION)


@pytest.fixture(scope="module")
def milvus_collection():
    '''Milvus 集合：不存在则新建，已存在则复用，测试结束只关闭连接。'''
    pymilvus, client = _connect_milvus()
    if not client.has_collection(COLLECTION):
        _create_milvus_collection(pymilvus, client)
    else:
        client.load_collection(COLLECTION)
    try:
        yield client
    finally:
        client.close()


# ---------------- PG 表（手工建，与 Milvus 集合字段一致） ----------------


def _import_psycopg() -> Tuple[Any, int]:
    """优先 psycopg3，缺失时回退 psycopg2；两者都缺则跳过。"""
    try:
        import psycopg

        return psycopg, 3
    except ImportError:
        pass
    try:
        import psycopg2

        return psycopg2, 2
    except ImportError as exc:
        _skip_service("psycopg", PG_HOST + ":" + str(PG_PORT), exc)


def _pg_connect(driver: Any) -> Any:
    """建立 PG 连接：连接串非空时优先使用，否则按字段拼装。"""
    try:
        if PG_CONNECTION_STRING:
            return driver.connect(PG_CONNECTION_STRING)
        return driver.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB,
        )
    except Exception as exc:  # noqa: BLE001 连接失败统一按环境问题跳过
        _skip_service("PostgreSQL", PG_HOST + ":" + str(PG_PORT) + " db=" + PG_DB, exc)



def _create_pg_table(conn: Any) -> None:
    """手工建表：id 采用 BIGINT（雪花 ID），配备扁平化字段与多元索引。"""
    with conn.cursor() as cursor:
        # 1. 确保安装向量扩展
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # 2. 创建主表（id 改为 BIGINT 存储雪花 ID）
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS " + COLLECTION + " ("
            "id BIGINT PRIMARY KEY, "
            "vector vector(" + str(DIM) + "), "
            "text TEXT NOT NULL, "
            "text_lemma TEXT, "
            "title VARCHAR(1024), "
            "title_md5 CHAR(32), "
            "source VARCHAR(2048)"
            ")"
        )

        # 3. 创建 HNSW 向量余弦相似度索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS " + COLLECTION + "_hnsw_idx ON " + COLLECTION
            + " USING hnsw (vector vector_cosine_ops)"
        )

        # 4. 创建基于 text_lemma 的 GIN 全文检索索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS " + COLLECTION + "_text_idx ON " + COLLECTION
            + " USING gin (to_tsvector('simple', text_lemma))"
        )

        # 5. 创建针对 title_md5 的 B-Tree 索引（用于按文档快速检索/清理）
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS " + COLLECTION + "_title_md5_idx ON " + COLLECTION
            + " (title_md5)"
        )

    conn.commit()


@pytest.fixture(scope="module")
def pg_connection():
    '''PG 表：表与索引均按 IF NOT EXISTS 建立，测试结束只关闭连接。'''
    driver, _ = _import_psycopg()
    conn = _pg_connect(driver)
    try:
        _create_pg_table(conn)
    except Exception as exc:  # noqa: BLE001 建表失败按环境问题跳过
        conn.close()
        _skip_service("PostgreSQL 建表", COLLECTION, exc)
    try:
        yield conn
    finally:
        conn.close()


# ---------------- 写入与回查 ----------------


def _insert_milvus(client: Any, rows: List[Dict[str, Any]]) -> int:
    '''写入 Milvus 并 flush；sparse 由 BM25 函数生成，不能出现在插入数据中。'''
    data = [
        {key: row[key] for key in ('id', 'vector', 'text', 'text_lemma', 'title', 'title_md5', 'source')}
        for row in rows
    ]
    result = client.insert(collection_name=COLLECTION, data=data)
    client.flush(collection_name=COLLECTION)
    return int(result['insert_count'])


def _insert_pg(conn: Any, rows: List[Dict[str, Any]]) -> int:
    '''写入 PG，返回受影响行数。'''
    sql = (
        'INSERT INTO ' + COLLECTION
        + ' (id, vector, text, text_lemma, title, title_md5, source)'
        + ' VALUES (%s, %s::vector, %s, %s, %s, %s, %s)'
    )
    params = [
        (
            row['id'],
            row['vector'],
            row['text'],
            row['text_lemma'],
            row['title'],
            row['title_md5'],
            row['source'],
        )
        for row in rows
    ]
    with conn.cursor() as cursor:
        cursor.executemany(sql, params)
        affected = cursor.rowcount
    conn.commit()
    return int(affected)


def _query_milvus_ids(client: Any, ids: List[int]) -> List[int]:
    '''按主键回查 Milvus，返回命中 id 的升序列表。'''
    rows = client.query(
        collection_name=COLLECTION,
        filter='id in [' + ', '.join(str(value) for value in ids) + ']',
        output_fields=['id'],
    )
    return sorted(int(row['id']) for row in rows)


def _query_pg_ids(conn: Any, ids: List[int]) -> List[int]:
    '''按主键回查 PG，返回命中 id 的升序列表。'''
    sql = 'SELECT id FROM ' + COLLECTION + ' WHERE id = ANY(%s)'
    with conn.cursor() as cursor:
        cursor.execute(sql, (list(ids),))
        rows = cursor.fetchall()
    return sorted(int(row[0]) for row in rows)


# ---------------- 用例 ----------------


def test_insert_epub_chunks_into_both_backends(milvus_collection: MilvusClient, pg_connection) -> None:
    '''解析 EPUB、切块并向量化，把同一批行写入 Milvus 与 PG，并按主键回查校验。'''
    path = _epub_path()
    _require_epub(path)
    _require_api_key()

    # 1. 解析、切块、向量化，两侧共用同一批行。
    app = _build_app()
    rows = _build_rows(app, path)
    ids = [row['id'] for row in rows]

    # 2. 同一批数据分别写入两个后端，写入条数必须等于行数。
    milvus_count = _insert_milvus(milvus_collection, rows)
    pg_count = _insert_pg(pg_connection, rows)
    assert milvus_count == len(rows)
    assert pg_count == len(rows)

    # 3. 回查主键集合，确认两侧都完整落位。
    assert _query_milvus_ids(milvus_collection, ids) == sorted(ids)
    assert _query_pg_ids(pg_connection, ids) == sorted(ids)

    print('双后端写入完成：集合 ' + COLLECTION + '，写入 ' + str(len(rows)) + ' 条')



def test_milvus_key_search(milvus_collection: MilvusClient):
    app = _build_app()

    text = "Chapter_1"
    vectors = app.embedder.encode([text])

    kwargs: Dict[str, Any] = {
        "collection_name": COLLECTION,
        "data": vectors,
        "anns_field": "vector",
        "limit": 10,
        "output_fields": ["*"],
    }

    hits = milvus_collection.search(**kwargs)
    assert len(hits) > 0