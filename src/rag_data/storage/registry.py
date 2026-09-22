# 存储后端注册表：继承 BaseVectorStore 即自动注册，配置只写后端名即可切换。
#
# 注册值可以是点分路径（惰性导入），也可以是类对象（函数内定义的本地类无法按路径导入）。

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

from rag_data.registry import Ref, Registry

if TYPE_CHECKING:  # pragma: no cover
    from rag_data.config import Settings
    from rag_data.logging.base import LoggerAdapter
    from rag_data.storage.base import BaseVectorStore

# 内置后端名称到类路径的映射，按需惰性导入。
BUILTIN_BACKENDS: Dict[str, str] = {
    "memory": "rag_data.storage.memory_store.InMemoryVectorStore",
    "milvus": "rag_data.storage.milvus_store.MilvusVectorStore",
}

_REGISTRY = Registry("存储后端", BUILTIN_BACKENDS)


def register_store(name: str, ref: Ref) -> None:
    """登记后端：ref 可为类对象，也可为点分路径字符串。"""
    _REGISTRY.register(name, ref)


def register_backend(name: str):
    """装饰器形式登记后端。"""
    return _REGISTRY.decorator(name)


def is_registered(name: str) -> bool:
    """判断后端是否已登记。"""
    return name in _REGISTRY


def available_backends() -> List[str]:
    """返回已登记的后端名，按字典序排列。"""
    return _REGISTRY.names()


def resolve_store(name: str) -> Type[BaseVectorStore]:
    """按后端名取得存储类，路径登记的在此时惰性导入。"""
    return _REGISTRY.resolve(name)


def create_store(
    name: str,
    settings: Settings,
    logger: LoggerAdapter,
    **kwargs: Any,
) -> BaseVectorStore:
    """按后端名创建实例，统一以关键字传入 settings 与 logger。"""
    return _REGISTRY.create(name, settings=settings, logger=logger, **kwargs)
