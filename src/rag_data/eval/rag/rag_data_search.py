import json
import logging
import os
from pathlib import Path
from typing import Union, List, Any, Dict

from dotenv import load_dotenv

from rag_data import EmbeddingSettings, LLMSettings, RagSetting
from rag_data.rag_engine import RagEngine, _build_llm

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    from pymilvus import MilvusClient
    from langchain_openai import OpenAI
except Exception as e:
    logger.warning(f"{__file__} 缺少依赖:str{e}")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
env_file = BASE_DIR / ".env_rag_eval"

# 显式加载自定义文件名的环境变量
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    # 如果在某些特殊 CI 环境，也可以尝试不带路径加载默认的
    load_dotenv()
    

RAG_SEARCH_JSON = r"""
{
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


RAG_SEARCH_CONFIG = json.loads(RAG_SEARCH_JSON)


EMBEDDING_SETTINGS = EmbeddingSettings(**RAG_SEARCH_CONFIG["embedding"])
EMBEDDING_SETTINGS.api_key = os.getenv("embedding_api_key", "")

LLM_SETTINGS = LLMSettings(**RAG_SEARCH_CONFIG["llm"])
LLM_SETTINGS.api_key = os.getenv("llm_api_key", "")

COLLECTION = "eval_rag_data"

# 向量维度一律取嵌入配置，避免与 vector.dim 的占位值不一致。
DIM = EMBEDDING_SETTINGS.dim

MILVUS_URI = os.getenv("milvus_url", "http://127.0.0.1:19530")
MILVUS_DB = os.getenv("db_name", "eval_rag_data")
MILVUS_TOKEN = os.getenv("db_token", "")
MILVUS_USERNAME = os.getenv("db_username", "")
MILVUS_PASSWORD = os.getenv("db_password", "")

def fetch_neighbor_contexts(
    client: MilvusClient,
    search_results: Union[List[Any], Any],
    window_size: int = 1,
) -> str:
    """适配现有 Schema [id, source, text, vector] 的邻居窗口扩展函数。
    支持 search_results 数组（单 Query 或多 Query 检索结果）。
    要求 id 格式为 '{doc_prefix}_{chunk_index}'，例如 'a1b2c3_0', 'a1b2c3_1'。
    """

    # 1. 扁平化展开 search_results 数组 (支持多 Query/Hits 嵌套)
    all_hits = []
    if isinstance(search_results, (list, tuple)):
        for item in search_results:
            if hasattr(item, "__iter__") and not isinstance(item, (dict, str)):
                all_hits.extend(item)
            else:
                all_hits.append(item)
    else:
        all_hits = list(search_results)

    # 2. 从 id 字段解析 doc_prefix 与 chunk_index
    doc_ranges = {}  # 格式: {doc_prefix: set(chunk_indices)}

    for hit in all_hits:
        entity = getattr(hit, "entity", hit)

        # 兼容 PyMilvus Hit 对象和 dict 对象
        if isinstance(entity, dict):
            hit_id = entity.get("id")
        else:
            hit_id = getattr(entity, "id", None) or entity.get("id")

        if not hit_id or "-" not in str(hit_id):
            continue

        # 从右侧切割一次：左边是完整的文档唯一标识(doc_prefix)，右边是切片序号(chunk_index)
        parts = str(hit_id).rsplit("-", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue  # 抛弃后缀不是纯数字的非标准 id

        doc_prefix, idx_str = parts[0], parts[1]
        curr_idx = int(idx_str)

        start_idx = max(0, curr_idx - window_size)
        end_idx = curr_idx + window_size

        # 核心：按完整的 doc_prefix 进行分组（自动按文档聚合）
        if hit_id not in doc_ranges:
            doc_ranges[hit_id] = set()

        doc_ranges[hit_id].update(range(start_idx, end_idx + 1))

    # 若无法根据 id 解析出任何邻居范围，退回直接提取当前命中的原始文本
    if not doc_ranges:
        raw_texts = []
        for hit in all_hits:
            entity = getattr(hit, "entity", hit)
            t = entity.get("text") if isinstance(entity, dict) else entity.get("text")
            if t:
                raw_texts.append(t)
        return "\n\n---\n\n".join(raw_texts)

    merged_chunks: List[str] = []

    # 3. 按文档前缀构造 id 列表批量查询相邻块
    for hit_id, indices_set in doc_ranges.items():
        sorted_indices = sorted(list(indices_set))
        target_ids = [f"{str(hit_id).rsplit("-",1)[0]}-{i}" for i in sorted_indices]

        # 从 Milvus 根据主键 id 批量查询
        expr = f'id in {target_ids}'
        records = client.query(
            collection_name=COLLECTION,
            expr=expr,
            output_fields=["id", "text"],
        )

        # 根据 id 中的物理序号升序排列，保证还原上下文连续性
        def parse_idx(item: dict) -> int:
            return int(item["id"].rsplit("-", 1)[1])

        sorted_records = sorted(records, key=parse_idx)
        doc_text = "\n\n".join([item["text"] for item in sorted_records])
        merged_chunks.append(doc_text)

    return "\n\n---\n\n".join(merged_chunks)

def rag_call(app: RagEngine, client:MilvusClient):


    question = "这是在经历了艰难的成长、求学、事业开拓之后"

    query_vector = app.embedder.encode([question])

    # 使用之前实现好的带窗口扩展/检索逻辑获取上下文
    # 示例中使用 search 拉取 top-K，或者使用你的 pipeline 检索
    search_results = client.search(
        collection_name=COLLECTION,
        data=query_vector,
        anns_field="vector",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=3,
        output_fields=["text"],
    )

    # 3. 核心修改：向前后扩展 1 个邻居窗口 (拉取 [idx-1, idx, idx+1])
    if search_results and search_results[0]:
        context = fetch_neighbor_contexts(
            client=client,
            search_results=search_results,
            window_size=1,  # 向前向后各扩 1 个块
        )
    else:
        context = "暂无数据"

    prompt =  f"上下文：\n{context}\n\n问题：{question}"

    llm_client = _build_llm(LLM_SETTINGS)
    response = llm_client.complete(prompt=prompt,system="你是一个专业的 RAG 系统问答助手。请根据提供的上下文回答问题。")

    answer = response
    print("OpenAI 输出结果：\n", answer)


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

if __name__ == "__main__":

    settings = RagSetting.load_mapping(
        embedding={
            "enable": True,
            "provider": EMBEDDING_SETTINGS.provider,
            "model": EMBEDDING_SETTINGS.model,
            "api_key": EMBEDDING_SETTINGS.api_key,
            "dim": EMBEDDING_SETTINGS.dim,
            "batch_size": EMBEDDING_SETTINGS.batch_size,
        }
    )

    app = RagEngine(settings=settings)
    client = _connect_milvus()
    rag_call(app,client)