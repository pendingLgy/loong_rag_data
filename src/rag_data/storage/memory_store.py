# 内存向量库实现，用于单元测试与本地验证。
#
# 与 Milvus 实现走完全相同的存储行路径：内存中保存的是 record 的存储行，
# 而非 pydantic 对象，因此扩展字段的序列化与还原逻辑在这里同样被覆盖。

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

from rag_data.models import PRIMARY_FIELD, VECTOR_FIELD, MemoryRecord, QueryHit
from rag_data.storage.base import BaseVectorStore


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """计算余弦相似度并归一化到 0 到 1。"""
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(size))
    norm_left = math.sqrt(sum(x * x for x in left[:size]))
    norm_right = math.sqrt(sum(x * x for x in right[:size]))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    raw = dot / (norm_left * norm_right)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


class InMemoryVectorStore(BaseVectorStore[MemoryRecord]):
    """以字典保存存储行的内存实现。"""

    backend = "memory"

    def __init__(
        self,
        settings: Optional[Any] = None,
        logger: Optional[Any] = None,
        record_class: Type[MemoryRecord] = MemoryRecord,
        extra_columns: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
        # 统一构造函数签名，便于按注册表创建；内存实现不需要 settings 与 logger。
        self._settings = settings
        self._logger = logger
        # record_class 由配置 models.record_class 解析后注入，用于还原自定义子类。
        self._record_class = record_class
        # 与 Milvus 实现保持一致：提升列作为平铺列存储，其余扩展字段进 metadata。
        self._extra_columns = [dict(item) for item in (extra_columns or [])]
        self._promoted = [str(item["name"]) for item in self._extra_columns if item.get("name")]
        self._rows: Dict[str, Dict[str, Any]] = {}

    def ensure_collection(self) -> None:
        return None

    def upsert(self, records: List[MemoryRecord]) -> int:
        """记录转为存储行后写入，扩展字段落入 metadata 列。"""
        for record in records:
            self._rows[record.id] = record.to_storage_row(self._promoted)
        return len(records)

    def query(
        self,
        vector: List[float],
        top_n: int,
        filters: Optional[Dict[str, Any]] = None,
) -> List[QueryHit]:
        hits: List[QueryHit] = []
        for row in self._rows.values():
            if not self._record_class.row_matches(row, filters or {}, self._promoted):
                continue
            hits.append(
                QueryHit(
                    id=row[PRIMARY_FIELD],
                    text_payload=row["text_payload"],
                    score=cosine_similarity(vector, row[VECTOR_FIELD]),
                    entities=list(row.get("entities", [])),
                    created_at=row["created_at"],
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_n]

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        """按主键取回记录，验证扩展字段可完整往返。"""
        row = self._rows.get(record_id)
        return self._record_class.from_storage_row(row, self._promoted) if row is not None else None

    def get_row(self, record_id: str) -> Optional[Dict[str, Any]]:
        """按主键取回原始存储行，便于断言落库形态。"""
        row = self._rows.get(record_id)
        return dict(row) if row is not None else None

    def count(self) -> int:
        """返回当前记录数，便于测试断言。"""
        return len(self._rows)
