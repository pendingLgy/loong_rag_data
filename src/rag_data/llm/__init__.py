# 大模型：抽象接口、纯注册表（供用户注册自定义实现）与内置 provider；实例化在处理器调用时进行。

from rag_data.llm.base import BaseLLMProvider
from rag_data.llm.deepseek_llm import DeepSeekLLMProvider
from rag_data.llm.llm_handler import LLMHandler
from rag_data.llm.openai_llm import OpenAILLMProvider
from rag_data.llm.qwen_llm import QwenLLMProvider
from rag_data.llm.registry_llm import (
    BUILTIN_PROVIDERS,
    available_llm_providers,
    is_llm_registered,
    register_llm,
    register_llm_provider,
    resolve_llm_provider,
)

__all__ = [
    "BaseLLMProvider",
    "LLMHandler",
    "OpenAILLMProvider",
    "QwenLLMProvider",
    "DeepSeekLLMProvider",
    "register_llm",
    "register_llm_provider",
    "available_llm_providers",
    "is_llm_registered",
    "resolve_llm_provider",
    "BUILTIN_PROVIDERS",
]

