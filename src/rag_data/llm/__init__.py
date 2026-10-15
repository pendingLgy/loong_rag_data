# 大模型调用：抽象接口、注册表与内置 provider。

from rag_data.llm.base import BaseLLMProvider
from rag_data.llm.client import LLM
from rag_data.llm.deepseek_llm import DeepSeekLLMProvider
from rag_data.llm.openai_llm import OpenAILLMProvider
from rag_data.llm.qwen_llm import QwenLLMProvider
from rag_data.llm.registry import (
    BUILTIN_PROVIDERS,
    available_llm_providers,
    create_llm_provider,
    is_llm_registered,
    register_llm,
    register_llm_provider,
    resolve_llm_provider,
)

__all__ = [
    "BaseLLMProvider",
    "LLM",
    "OpenAILLMProvider",
    "QwenLLMProvider",
    "DeepSeekLLMProvider",
    "register_llm",
    "register_llm_provider",
    "available_llm_providers",
    "is_llm_registered",
    "resolve_llm_provider",
    "create_llm_provider",
    "BUILTIN_PROVIDERS",
]

