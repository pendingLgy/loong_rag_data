import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import BadRequestError
from pymilvus import AsyncMilvusClient

from rag_data.config import (
    EmbeddingSettings,
    MilvusSettings,
    PGVectorSettings,
    LLMSettings,
)
from rag_data.embedding import BaseEmbedder
from rag_data.facade import _build_embedder, _build_llm
from rag_data.llm import BaseLLMProvider

BASE_DIR = Path(__file__).resolve().parent.parent
env_file = BASE_DIR / ".env_ai_key"

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
  "llm":{
    "enable": true,
    "provider": "qwen",
    "model": "qwen3.7-plus",
    "api_key":"",
    "temperature": 0.0
  },
  "embedding": {
    "enable": true,
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
    "max_chunks": 8,
    "top_k": 3
  }
}
"""

RAG_EVAL_CONFIG = json.loads(RAG_EVAL_JSON)

# 下方四个 Settings 与两个字典供用例正文引用，字段由上方 JSON 推导。
EMBEDDING_SETTINGS = EmbeddingSettings(**RAG_EVAL_CONFIG["embedding"])
EMBEDDING_SETTINGS.api_key = os.getenv("embedding_api_key", "")

LLM_SETTINGS = LLMSettings(**RAG_EVAL_CONFIG["llm"])
LLM_SETTINGS.api_key = os.getenv("llm_api_key", "")

MILVUS_SETTINGS = MilvusSettings(**RAG_EVAL_CONFIG["milvus"])
PG_VECTOR_SETTINGS = PGVectorSettings(**RAG_EVAL_CONFIG["pg"])

COLLECTION = "eval_rag_data"

# 向量维度一律取嵌入配置，避免与 vector.dim 的占位值不一致。
DIM = EMBEDDING_SETTINGS.dim

# PG 连接参数。
PG_CONNECTION_STRING = PG_VECTOR_SETTINGS.connection_string
PG_HOST = PG_VECTOR_SETTINGS.host
PG_PORT = PG_VECTOR_SETTINGS.port
PG_USER = PG_VECTOR_SETTINGS.user
PG_PASSWORD = PG_VECTOR_SETTINGS.password
PG_DB = PG_VECTOR_SETTINGS.database


def _connect_milvus() -> AsyncMilvusClient:
    """连接 Milvus：缺 pymilvus 或服务不可用时跳过，返回模块与客户端。"""
    try:
        import pymilvus
    except ImportError as exc:
        raise exc

    kwargs: Dict[str, Any] = {"uri": MILVUS_SETTINGS.uri}
    if MILVUS_SETTINGS.token:
        kwargs["token"] = MILVUS_SETTINGS.token
    if MILVUS_SETTINGS.db_name:
        kwargs["db_name"] = MILVUS_SETTINGS.db_name
    try:
        return pymilvus.AsyncMilvusClient(**kwargs)
    except Exception as exc:  # noqa: BLE001 连接失败统一按环境问题跳过
        raise exc


def build_milvus_client():
    '''Milvus 集合：不存在则新建，已存在则复用，测试结束只关闭连接。'''
    client = _connect_milvus()

    try:
        yield client
    finally:
        client.close()


async def safe_generate_testset(generator, chunks, batch_size=5, testset_size_per_batch=10, run_config=None):
    successful_dfs = []

    # 将长切片列表切分为每组 batch_size 个的小批次
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            print(f"正在处理第 {batch_num} 批次切片（包含 {len(batch_chunks)} 个切片）...")
            partial_testset = generator.generate_with_chunks(
                chunks=batch_chunks,
                testset_size=testset_size_per_batch,
                run_config=run_config
            )
            successful_dfs.append(partial_testset.to_pandas())
            print(f"-> 第 {batch_num} 批次生成成功！")

        except BadRequestError as e:
            print(f"[安全审核拦截] 第 {batch_num} 批次触发百炼内容安全审查 (DataInspectionFailed)，已跳过该批次: {e}")
            continue
        except Exception as e:
            print(f"[异常跳过] 第 {batch_num} 批次发生其他错误: {e}")
            continue

    if successful_dfs:
        final_df = pd.concat(successful_dfs, ignore_index=True)
        print(f"\n[成功] 批量测试集生成完毕！总计获取 {len(final_df)} 条有效数据。")
        return final_df
    else:
        print("\n[错误] 没有成功生成任何测试集数据。")
        return None

async def build_chunk_rag_test_docs():

    from langchain_core.documents import Document
    from ragas.testset import TestsetGenerator
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.run_config import RunConfig


    llm_handler = _build_llm(LLM_SETTINGS)
    embedder_handler = _build_embedder(EMBEDDING_SETTINGS)

    llm_model: BaseLLMProvider = llm_handler._target()
    embedder: BaseEmbedder = embedder_handler._target()

    milvus_client = _connect_milvus()

    try:
        res = await milvus_client.query(
            collection_name=COLLECTION,
            filter="",
            limit=100,
            output_fields=["id", "text", "title", "source"]
        )
    finally:
        # 确保异步客户端正常关闭，释放底层连接和事件循环引用
        await milvus_client.close()

    chunks = [
        Document(
            page_content=item["text"],
            metadata={"chunk_id": item["id"], "title": item["title"], "source": item["source"]}
        )
        for item in res
    ]

    # chunks = [
    #     Document(
    #         page_content="""The Eiffel Tower (Tour Eiffel) is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Locally nicknamed "La Dame de Fer" (French for "The Iron Lady"), it was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. Although initially criticized by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognizable structures in the world.""",
    #         metadata={"source": "doc1", "chunk_id": 1}
    #     ),
    #     Document(
    #         page_content="""The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).""",
    #         metadata={"source": "doc1", "chunk_id": 2}
    #     )
    # ]

    # 初始化合成器
    generator = TestsetGenerator.from_langchain(
        llm=ChatOpenAI(model=llm_model.model, api_key=llm_model.api_key, base_url=llm_model.base_url, max_retries=0),
        embedding_model=OpenAIEmbeddings(
            model=embedder.model,
            api_key=embedder.api_key,
            base_url=embedder.base_url,
            check_embedding_ctx_length=False,
            dimensions=1024,
            max_retries=0
        ),
    )

    run_config = RunConfig(
        max_workers=8,  # 建议调小到 4 防止高并发限流
        timeout=300,
        max_retries=3,
        exception_types=(Exception,)
    )

    # 注意：你的 chunks 只有 2 条，但 testset_size=50，建议调试时改成 2
    # testset = generator.generate_with_chunks(chunks=chunks, testset_size=50, run_config=run_config)

    # 替换为具备安全跳过能力的函数
    testset_df = await safe_generate_testset(
        generator=generator,
        chunks=chunks,
        batch_size=10,
        testset_size_per_batch=5,  # 调试时建议设小
        run_config=run_config
    )

    if testset_df is not None:
        output_file = "testset.csv"
        testset_df.to_csv(output_file, index=False)
        print(f"Testset saved to {output_file}")
        print(testset_df.head())


async def main():
    print("=== 开始在独立线程中执行 Ragas 测试集生成 ===")
    await build_chunk_rag_test_docs()

if __name__ == "__main__":
    """r
        dataset from  test_epub_to_milvus_and_pg::test_insert_epub_chunks_into_both_backends
    """
    # 1. 彻底规避 Windows 下的 Proactor 循环残留关闭问题
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
