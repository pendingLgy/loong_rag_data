# 向量库抽象接口，隔离具体实现。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, List, Optional, TypeVar

from rag_data.models import MemoryRecord, QueryHit
from rag_data.storage.registry import register_store

# 存储实现按自己处理的记录类型参数化：内存后端用 MemoryRecord，
# Milvus 后端用 MilvusRecord，两者共享同一套接口。
RecordT = TypeVar("RecordT", bound=MemoryRecord)


class BaseVectorStore(ABC, Generic[RecordT]):
    """向量库的统一接口，按记录类型参数化。"""

    # 配置中 storage.backend 的取值；子类置为该名字即自动注册。
    backend: ClassVar[Optional[str]] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """子类声明 backend 后自动注册，无需额外注册调用。"""
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("backend")
        if isinstance(name, str) and name:
            # 直接注册类对象，避免本地类因 qualname 无法按路径导入。
            register_store(name, cls)

    @abstractmethod
    def ensure_collection(self) -> None:
        """确保集合与索引存在，重复调用需幂等。"""

    @abstractmethod
    def upsert(self, records: List[RecordT]) -> int:
        """按 id 幂等写入，返回写入条数；扩展字段一并落库。"""

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_n: int,
        filters: Optional[Dict[str, Any]] = None,
) -> List[QueryHit]:
        """按向量相似度检索，返回降序候选。"""
        # filters 为字段精确匹配，键可指向基类字段、提升列或扩展字段。

    def close(self) -> None:
        """释放底层资源，默认无操作。"""
        return None
