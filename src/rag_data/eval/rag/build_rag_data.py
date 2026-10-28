import hashlib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from rag_data.nlp.lemmatization import lemmatize_zh_for_bm25
from rag_data.rag_engine import RagEngine
from rag_data.util.snowflake import next_id

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    from pymilvus import MilvusClient
except Exception as e:
    logger.warning(f"{__file__} 缺少依赖:str{e}")

from dotenv import load_dotenv

from rag_data.core.config import (
    EmbeddingSettings,
    LLMSettings, RagSetting, ParsingSettings,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
env_file = BASE_DIR / ".env_rag_eval"

# 显式加载自定义文件名的环境变量
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    # 如果在某些特殊 CI 环境，也可以尝试不带路径加载默认的
    load_dotenv()

# 本文件独立的 JSON 配置：不读 conftest，也不读环境变量。
# sample.epub 未填写时用例跳过；embedding.api_key 需按本机填写。
RAG_EVAL_JSON = r"""
{
  "parsing": {
    "spacy_model": "zh_core_web_sm",
    "max_chars": 500,
    "safe_max_chars": 2000
  },
  "embedding": {
    "enable": true,
    "provider": "qwen",
    "model": "qwen3.7-text-embedding",
    "dim": 1024,
    "api_key": "",
    "batch_size": 10
  },
  "vector": {
    "max_chunks": 8,
    "top_k": 3
  },
  "llm":{
    "enable": true,
    "provider": "qwen",
    "model": "qwen3.7-flash",
    "api_key": "",
    "temperature": 0.0
  }
}
"""

RAG_EVAL_CONFIG = json.loads(RAG_EVAL_JSON)

PARSING_SETTINGS = ParsingSettings(**RAG_EVAL_CONFIG["parsing"])

EMBEDDING_SETTINGS = EmbeddingSettings(**RAG_EVAL_CONFIG["embedding"])
EMBEDDING_SETTINGS.api_key = os.getenv("embedding_api_key", "")

LLM_SETTINGS = LLMSettings(**RAG_EVAL_CONFIG["llm"])
LLM_SETTINGS.api_key = os.getenv("llm_api_key", "")

COLLECTION = "eval_rag_data"

# 向量维度一律取嵌入配置，避免与 vector.dim 的占位值不一致。
DIM = EMBEDDING_SETTINGS.dim

MILVUS_URI = os.getenv("milvus_url", "http://127.0.0.1:19530")
MILVUS_DB = os.getenv("db_name", "eval_rag_data")
MILVUS_TOKEN = os.getenv("db_token", "")
MILVUS_USERNAME = os.getenv("db_username", "")
MILVUS_PASSWORD = os.getenv("db_password", "")


def _create_milvus_collection(pymilvus: Any, client: Any) -> None:
    '''手工建集合：稠密向量 + text_lemma 经 BM25 函数输出稀疏向量。'''
    fields = [
        pymilvus.FieldSchema(name='id', dtype=pymilvus.DataType.INT64, is_primary=True),
        pymilvus.FieldSchema(name='vector', dtype=pymilvus.DataType.FLOAT_VECTOR, dim=DIM),
        pymilvus.FieldSchema(name='text', dtype=pymilvus.DataType.VARCHAR, max_length=65535),
        pymilvus.FieldSchema(name='text_lemma', dtype=pymilvus.DataType.VARCHAR, max_length=65535,
                             enable_analyzer=True),
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
    index_params.add_index(field_name='sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25',
                           index_name='sparse_index')
    index_params.add_index(field_name='title_md5', index_type='INVERTED', index_name='title_md5_index')

    client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
    client.load_collection(COLLECTION)


def _connect_milvus() -> MilvusClient:
    """连接 Milvus：缺 pymilvus 或服务不可用时跳过，返回模块与客户端。"""
    try:
        import pymilvus
    except ImportError as exc:
        raise exc

    kwargs: Dict[str, Any] = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    if MILVUS_DB:
        kwargs["db_name"] = MILVUS_DB
    if MILVUS_USERNAME:
        kwargs["user"] = MILVUS_USERNAME
    if MILVUS_PASSWORD:
        kwargs["password"] = MILVUS_PASSWORD
    try:
        client = pymilvus.MilvusClient(**kwargs)

    except Exception as exc:  # noqa: BLE001 连接失败统一按环境问题跳过
        raise exc

    if not client.has_collection(COLLECTION):
        _create_milvus_collection(pymilvus, client)
    else:
        client.load_collection(COLLECTION)

    return client


def _build_rag_engine() -> RagEngine:
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
        }
    )

    return RagEngine(settings=settings)


def _build_rows(app: RagEngine, path: str) -> List[Dict[str, Any]]:
    '''解析 EPUB、切块并向量化，返回两侧共用的行数据。'''
    text = app.parse_document(path)
    assert text.strip()
    try:
        chunks = [chunk for chunk in app.chunk_document(path, text) if chunk.strip()]
    except Exception as exc:  # noqa: BLE001 句柄或模型缺失属环境问题
        print('❌ 切块不可用：' + type(exc).__name__ + ' ' + str(exc))

    assert chunks
    batch = chunks[:1]
    # batch = chunks

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


def _insert_milvus(client: Any, rows: List[Dict[str, Any]]) -> int:
    '''写入 Milvus 并 flush；sparse 由 BM25 函数生成，不能出现在插入数据中。'''
    data = [
        {key: row[key] for key in ('id', 'vector', 'text', 'text_lemma', 'title', 'title_md5', 'source')}
        for row in rows
    ]
    result = client.insert(collection_name=COLLECTION, data=data)
    client.flush(collection_name=COLLECTION)
    return int(result['insert_count'])

def main():
    # path_str = "C:\\Users\\loong\\Desktop\\伯特兰·罗素：哲学问题.txt"
    path_str = "C:\\Users\\loong\\Desktop\\test_chunk.txt"
    file_path = Path(path_str).resolve()
    if file_path.exists():
        print(f"✅ 文件存在：{file_path}")
    else:
        print(f"❌ 文件不存在，请检查路径或文件名：{file_path}")
        return

    app = _build_rag_engine()
    rows = _build_rows(app, str(file_path))

    # 同一批数据分别写入两个后端，写入条数必须等于行数。
    client = _connect_milvus()
    milvus_count = _insert_milvus(client, rows)

    print('写入完成：集合 ' + COLLECTION + '，写入 ' + str(len(rows)) + ' 条')

if __name__ == "__main__":
    print("start...")
    main()


