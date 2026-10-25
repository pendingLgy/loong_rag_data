import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI, AsyncOpenAI
from pymilvus import AsyncMilvusClient
from ragas import aevaluate, RunConfig
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.metrics._context_precision import ContextPrecision, context_precision
from ragas.metrics._context_recall import ContextRecall, context_recall
from ragas.metrics._faithfulness import Faithfulness, faithfulness
from ragas.metrics._nv_metrics import AnswerAccuracy

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
env_file = BASE_DIR / ".env_rag_eval"

# 显式加载自定义文件名的环境变量
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    # 如果在某些特殊 CI 环境，也可以尝试不带路径加载默认的
    load_dotenv()

MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
DB_NAME = os.getenv("DB_NAME", '')
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")

DATASET_PATH = os.getenv("DATASET_PATH", "")
RAG_EVAL_OUTPUT_PATH = os.getenv("RAG_EVAL_OUTPUT_PATH", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_URL = os.getenv("LLM_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")


def build_openai_client(base_url: str, api_key: str, timeout: int = 60):
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


_lock = threading.Lock()


class MyRAGSystem:

    def __init__(self):
        # 声明一个客户端实例，初始为 None
        self._client: AsyncMilvusClient = None

    async def _get_client(self) -> AsyncMilvusClient:
        """懒加载获取或创建唯一的 AsyncMilvusClient 实例"""

        with _lock:
            if self._client is None:
                self._client = AsyncMilvusClient(uri=MILVUS_URL, db_name=DB_NAME)
                await self._client.load_collection(COLLECTION_NAME)

        return self._client

    def embed_query(self, query: str):
        """将用户问题转换为向量（此处可替换为你实际使用的 Embedding 客户端调用）"""
        client = build_openai_client(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_URL,
            timeout=60,
        )

        kwargs: Dict[str, Any] = {
            "model": EMBEDDING_MODEL,
            "input": [query],
            "dimensions": 1024
        }

        response = client.embeddings.create(**kwargs)

        data = response.model_dump()
        items = data.get("data") or []
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [[float(value) for value in item["embedding"]] for item in ordered]

    async def retrieve_from_milvus(self, query: str, top_k: int = 3):
        """利用配置连接 Milvus 并基于你的表结构进行向量检索"""

        client = await self._get_client()
        query_vector = self.embed_query(query)

        # 在 Milvus 中执行向量相似度检索
        search_results = await client.search(
            collection_name=COLLECTION_NAME,
            data=query_vector,
            anns_field="vector",
            limit=top_k,
            output_fields=["text", "source", "title"]  # 对应你的表结构字段
        )

        # 提取 Milvus 检索出来的原始文本内容作为 contexts
        retrieved_contexts = []
        for hits in search_results:
            for hit in hits:
                text_content = hit.get("entity", {}).get("text", "")
                retrieved_contexts.append(text_content)

        return retrieved_contexts

    def generate_answer(self, query: str, contexts: list):
        """使用配置的 LLM 模型根据检索到的 contexts 生成回答（OpenAI 标准消息格式）"""
        context_str = "\n".join(contexts)

        # 将提示词封装为标准的 OpenAI 聊天消息格式（System + User）
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的 AI 助手。请严格根据提供的已知信息来回答问题，不要凭空捏造。如果已知信息中不包含答案，请直接回答“根据已知信息无法回答该问题”。"
            },
            {
                "role": "user",
                "content": f"已知信息：\n{context_str}\n\n请根据上述已知信息回答以下问题：\n{query}"
            }
        ]

        # 实例化大模型客户端（此处已修正为使用 LLM 的 API Key 和 URL）
        client = build_openai_client(
            api_key=LLM_API_KEY,
            base_url=LLM_URL,
            timeout=60,
        )

        kwargs: Dict[str, Any] = {
            "model": LLM_MODEL,
            "messages": messages,  # 传入封装好的 messages 列表
            "temperature": 0.7,
        }

        response = client.chat.completions.create(**kwargs)
        data = response.model_dump()

        choices = data.get("choices") or []
        message = choices[0].get("message") or {}

        return message.get("content") or ""


async def main():
    testset_csv = DATASET_PATH
    if not os.path.exists(testset_csv):
        print(f"[错误] 未找到测试集文件 {testset_csv}，请先运行测试集生成脚本。", file=sys.stderr)
        return

    print("=== 1. 正在加载 Ragas 测试集 ===")
    df = pd.read_csv(testset_csv, encoding='gbk')

    # 调试时可仅取前几条测试：df = df.head(5)
    questions = df['user_input'].tolist()[:2]
    ground_truths = df['reference'].tolist()[:2]  # 标准答案

    print(f"成功加载测试集，共计 {len(questions)} 条测试用例。")

    # 初始化 RAG 系统
    rag_system = MyRAGSystem()

    print("=== 2. 正在运行 RAG 系统获取检索结果与大模型回答 ===")
    eval_questions = []
    eval_answers = []
    eval_contexts = []

    for i, q in enumerate(questions):
        print(f"正在评测第 {i + 1}/{len(questions)} 条: {q[:50]}...")

        try:
            retrieved_contexts = await rag_system.retrieve_from_milvus(q)
        except Exception as e:
            print(f"[警告] Milvus 检索失败: {e}，使用空上下文代替。", file=sys.stderr)
            retrieved_contexts = []

        # B. 大模型生成回答
        answer = rag_system.generate_answer(q, retrieved_contexts)

        eval_questions.append(q)
        eval_answers.append(answer)
        eval_contexts.append(retrieved_contexts)

    print("=== 3. 正在组装 Ragas 评测数据集 ===")
    dataset_dict = {
        "question": eval_questions,
        "answer": eval_answers,
        "contexts": eval_contexts,
        "ground_truth": ground_truths
    }
    eval_dataset = Dataset.from_dict(dataset_dict)

    print("=== 4. 开始执行 Ragas 自动化指标打分 ===")
    # 注意：Ragas 内部调用大模型时，会自动去环境变量中读取 OPENAI_API_KEY 等。
    # 如果你使用的是阿里云百炼等兼容 OpenAI 的接口，建议在 .env 中也将 API Key 同步赋值给 OPENAI_API_KEY/OPENAI_API_BASE，
    # 或者通过 Ragas 的 llm 配置项显式传入。


    client = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_URL,
        timeout=60)

    run_config = RunConfig(
        max_workers=2,
        timeout=300,
        max_retries=1,
        exception_types=(Exception,)
    )

    context_precision.llm = LangchainLLMWrapper(langchain_llm=client,run_config=run_config)
    context_recall.llm = LangchainLLMWrapper(langchain_llm=client,run_config=run_config)
    faithfulness.llm = LangchainLLMWrapper(langchain_llm=client,run_config=run_config)

    answer_accuracy = AnswerAccuracy()
    answer_accuracy.llm = LangchainLLMWrapper(langchain_llm=client,run_config=run_config)

    result = await aevaluate(
        dataset=eval_dataset,
        metrics=[
            context_precision,  # 评估 Milvus 召回准确度
            context_recall,  # 评估 Milvus 召回召回率
            faithfulness,  # 评估大模型防幻觉能力
            answer_accuracy  # 评估回答相关性
        ]
    )

    print("\n=== 评测完成！结果汇总 ===")
    print(result)

    # 导出详细的单条得分报表以便分析
    result_df = result.to_pandas()

    output_report = os.path.join(RAG_EVAL_OUTPUT_PATH, "rag_evaluation_report.csv")
    result_df.to_csv(output_report, index=False)
    print(f"详细评测报告已保存至: {output_report}")


def setup_windows_event_loop():
    """Windows平台下处理 ProactorEventLoop 兼容性"""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


if __name__ == "__main__":
    setup_windows_event_loop()

    asyncio.run(main())
