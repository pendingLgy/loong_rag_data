# 测试用的存储替身：生产代码已不含内存实现，这里提供一个最小后端，
# 使管道、门面与注册表相关测试无需真实 Milvus 即可运行。

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_data.storage.milvus_store import MilvusRecord


class FakeStore:
    """以字典保存存储行的最小后端，接口与 Milvus 实现保持一致。"""

    backend = "fake"

    def __init__(
        self,
        settings: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._rows: Dict[str, Dict[str, Any]] = {}
        self.ensure_collection_calls = 0
        self.closed = False

    def ensure_collection(self) -> None:
        self.ensure_collection_calls += 1

    def upsert(self, records: List[MilvusRecord]) -> int:
        """记录转为存储行后写入，按主键幂等。"""
        for record in records:
            self._rows[record.id] = record.to_storage_row()
        return len(records)

    def get_record(self, record_id: str) -> Optional[MilvusRecord]:
        row = self._rows.get(record_id)
        return MilvusRecord.from_storage_row(row) if row is not None else None

    def get_row(self, record_id: str) -> Optional[Dict[str, Any]]:
        row = self._rows.get(record_id)
        return dict(row) if row is not None else None

    def count(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        self.closed = True
