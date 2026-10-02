# 数据模型：切块中间态与向量化产物合一。

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """切块后的中间态；向量化后填入 vector，同一模型贯穿全链路。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)
    # 向量化后填入；默认空表示尚未向量化，维度由 Embedder 按 embedding.dim 校验。
    vector: List[float] = Field(default_factory=list)
