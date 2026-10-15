# 大模型注册表：继承 BaseLLMProvider 即自动注册，配置改 provider 名即可切换。
#
# 注册值可以是点分路径（惰性导入），也可以是类对象（函数内定义的本地类无法按路径导入）。

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

from rag_data.registry import Ref, Registry

if TYPE_CHECKING:  # pragma: no cover
    from rag_data.config import LLMSettings
    from rag_data.llm.base import BaseLLMProvider
    from rag_data.logging.base import LoggerAdapter

# 内置 provider 名称到类路径的映射，按需惰性导入。
BUILTIN_PROVIDERS: Dict[str, str] = {
    "openai": "rag_data.llm.openai_llm.OpenAILLMProvider",
    "qwen": "rag_data.llm.qwen_llm.QwenLLMProvider",
    "deepseek": "rag_data.llm.deepseek_llm.DeepSeekLLMProvider",
}

_REGISTRY = Registry("大模型", BUILTIN_PROVIDERS)


def register_llm(name: str, ref: Ref) -> None:
    "登记大模型：ref 可为类对象，也可为点分路径字符串。"
    _REGISTRY.register(name, ref)


def register_llm_provider(name: str):
    "装饰器形式登记大模型。"
    return _REGISTRY.decorator(name)


def is_llm_registered(name: str) -> bool:
    "判断大模型是否已登记。"
    return name in _REGISTRY


def available_llm_providers() -> List[str]:
    "返回已登记的 provider 名，按字典序排列。"
    return _REGISTRY.names()


def resolve_llm_provider(name: str) -> Type[BaseLLMProvider]:
    "按 provider 名取得实现类，路径登记的在此惰性导入。"
    return _REGISTRY.resolve(name)


def create_llm_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    "按 provider 名创建实例，统一以关键字传入 settings 与 logger。"
    return _REGISTRY.create(name, **kwargs)

