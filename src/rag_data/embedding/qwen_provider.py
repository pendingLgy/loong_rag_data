# 通义千问（DashScope 兼容模式）嵌入实现，沿用 OpenAI 请求格式。
#
# 与 OpenAI 仅默认模型、接口地址、API Key 环境变量与批量上限不同，故直接继承。

from __future__ import annotations

from rag_data.embedding.openai_provider import OpenAIEmbeddingProvider


class QwenEmbeddingProvider(OpenAIEmbeddingProvider):
    """调用 DashScope 的 OpenAI 兼容接口，默认使用 text-embedding-v3。"""

    backend = "qwen"
    default_model = "text-embedding-v3"
    default_dim = 1024
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"
    # DashScope 单次请求最多接受 10 条文本，超限会被拒绝。
    max_batch_size = 10
