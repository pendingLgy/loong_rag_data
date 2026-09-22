# 嵌入模型注册表：继承 BaseEmbeddingProvider 即自动注册，配置只写 provider 名即可切换。
#
# 注册值可以是点分路径（惰性导入），也可以是类对象（函数内定义的本地类无法按路径导入）。

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

from rag_data.registry import Ref, Registry

if TYPE_CHECKING:  # pragma: no cover
    from rag_data.config import Settings
    from rag_data.embedding.base import BaseEmbeddingProvider
    from rag_data.logging.base import LoggerAdapter

# 内置 provider 名称到类路径的映射，按需惰性导入。
BUILTIN_PROVIDERS: Dict[str, str] = {
    "openai": "rag_data.embedding.openai_provider.OpenAIEmbeddingProvider",
    "qwen": "rag_data.embedding.qwen_provider.QwenEmbeddingProvider",
}

_REGISTRY = Registry("嵌入模型", BUILTIN_PROVIDERS)


def register_embedding(name: str, ref: Ref) -> None:
    """登记嵌入模型：ref 可为类对象，也可为点分路径字符串。"""
    _REGISTRY.register(name, ref)


def register_embedding_provider(name: str):
    """装饰器形式登记嵌入模型。"""
    return _REGISTRY.decorator(name)


def is_embedding_registered(name: str) -> bool:
    """判断嵌入模型是否已登记。"""
    return name in _REGISTRY


def available_embedding_providers() -> List[str]:
    """返回已登记的 provider 名，按字典序排列。"""
    return _REGISTRY.names()


def resolve_embedding_provider(name: str) -> Type[BaseEmbeddingProvider]:
    """按 provider 名取得实现类，路径登记的在此时惰性导入。"""
    return _REGISTRY.resolve(name)


def create_embedding_provider(
    name: str,
    settings: Settings,
    logger: LoggerAdapter,
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """按 provider 名创建实例，统一以关键字传入 settings 与 logger。"""
    return _REGISTRY.create(name, settings=settings, logger=logger, **kwargs)
