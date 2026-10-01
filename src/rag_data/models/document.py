# 切块后的中间态模型。

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """切块后的中间态。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)
