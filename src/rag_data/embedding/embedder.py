# 文本向量化封装：provider 由配置选择，模型句柄可注入以便测试。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.config import Settings
from rag_data.embedding.registry import create_embedding_provider
from rag_data.exceptions import EmbeddingError
from rag_data.logging.base import LoggerAdapter


class Embedder:
    """文本向量化封装：按配置装配 provider，注入的 model 优先且绕过配置。"""

    def __init__(self, settings: Settings, logger: LoggerAdapter, model: Optional[Any] = None) -> None:
        self._settings = settings
        self._logger = logger
        # 注入的 model 只需实现 encode(texts)，可选提供 max_batch_size，便于测试与自定义。
        self._model = model
        self._provider: Optional[Any] = None

    @property
    def provider(self) -> Any:
        """实际使用的编码对象：注入的 model 优先，否则按配置装配 provider。"""
        return self._target()

    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为稠密向量，返回顺序与输入一致。"""
        if not texts:
            return []
        target = self._target()
        step = self._batch_size(target)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), step):
            vectors.extend(self._encode_batch(target, texts[start:start + step]))
        self._validate_dim(vectors)
        return vectors

    # ---------------- 内部实现 ----------------

    def _target(self) -> Any:
        if self._model is not None:
            return self._model
        if self._provider is None:
            name = self._settings.embedding.provider
            self._provider = create_embedding_provider(name, self._settings, self._logger)
            self._logger.info(
                "Embedding provider 已装配",
                provider=name,
                model=getattr(self._provider, "model", ""),
                dim=getattr(self._provider, "dim", 0),
            )
        return self._provider

    def _batch_size(self, target: Any) -> int:
        """批次大小：取配置值与 provider 单次上限的较小者，上限为 0 表示不限制。"""
        size = self._settings.embedding.batch_size
        limit = int(getattr(target, "max_batch_size", 0) or 0)
        return min(size, limit) if limit > 0 else size

    def _encode_batch(self, target: Any, batch: List[str]) -> List[List[float]]:
        """编码单批文本；是否归一化由 provider 决定，此处仅透传结果。"""
        result = target.encode(batch)
        return [list(map(float, row)) for row in result]

    def _validate_dim(self, vectors: List[List[float]]) -> None:
        """校验向量维度与 storage.vector_dim 一致，避免写入集合时不匹配。"""
        expected = self._settings.storage.vector_dim
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    "向量维度不一致：期望 " + str(expected) + "，实际 " + str(len(vector))
                )
