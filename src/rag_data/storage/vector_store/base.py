# 向量库抽象接口：屏蔽具体实现（Milvus、PG 等）。
#
# 约定：search 返回的 score 越大越相似（建议范围 [0, 1]）。
# 采用距离度量的实现需先转相似度：
#   COSINE 距离：score = max(0.0, 1.0 - distance)
#   L2 距离：score = 1.0 / (1.0 + distance)
#   IP 内积：score 即内积值（越大越相似）

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from rag_data.storage.vector_store.registry_vector import register_vector_store

# 支持的向量度量类型。
MetricType = Literal["COSINE", "L2", "IP"]


class VectorStoreResult(BaseModel):
    """一次检索或读取得到的结果：id、相似度与元数据。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    # 相似度分数，越大越相似；list、get 等无排序语境下为 None。
    score: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None


class VectorStoreBase(ABC):
    """向量库统一接口：声明 backend 后自动登记，配置改后端名即可切换。"""

    # 配置中的向量库后端名；子类置为该名字即自动登记。
    backend: ClassVar[Optional[str]] = None
    # 距离度量，子类可覆盖默认值。
    metric_type: ClassVar[MetricType] = "COSINE"

    def __init_subclass__(cls, **kwargs: object) -> None:
        """子类声明 backend 后自动登记，无需额外注册调用。"""
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("backend")
        if isinstance(name, str) and name:
            # 直接登记类对象，避免本地类因 qualname 无法按路径导入。
            register_vector_store(name, cls)

    @abstractmethod
    def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        """新建集合（PG 中即建表）。"""

    @abstractmethod
    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """批量写入向量及其元数据；ids 缺省时由实现自行生成。"""

    @abstractmethod
    def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[VectorStoreResult]:
        """按相似度检索，返回按 score 降序的结果。"""

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        """按 id 删除单条。"""

    @abstractmethod
    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        """按 id 更新向量与元数据（可只更新其一）。"""

    @abstractmethod
    def get(self, vector_id: str) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""

    @abstractmethod
    def list_cols(self) -> List[str]:
        """列出全部集合名。"""

    @abstractmethod
    def delete_col(self) -> None:
        """删除当前集合。"""

    @abstractmethod
    def col_info(self) -> Dict[str, Any]:
        """返回当前集合的统计信息。"""

    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[List[VectorStoreResult]]:
        """列出集合内容（外层列表按既有约定包裹一层）。"""

    @abstractmethod
    def reset(self) -> None:
        """删除并重建当前集合。"""

    def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        """关键字（BM25 或全文）检索；不支持的实现返回 None。"""
        return None

