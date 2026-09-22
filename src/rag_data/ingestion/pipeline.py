# 离线导入编排：解析、切块、实体抽取、向量化、批量写入。

from __future__ import annotations

import hashlib
import time
from typing import Any, List, Optional, Type

from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.ingestion import chunking
from rag_data.ingestion import entities as entities_module
from rag_data.ingestion import parsers
from rag_data.logging.base import LoggerAdapter
from rag_data.models import DocumentChunk, MemoryRecord
from rag_data.storage.base import BaseVectorStore

DEFAULT_USER_ID = "default"


class IngestionPipeline:
    """离线导入管道：组合各领域服务完成端到端导入。"""

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: Embedder,
        settings: Settings,
        logger: LoggerAdapter,
        nlp: Optional[Any] = None,
        record_class: Type[MemoryRecord] = MemoryRecord,
) -> None:
        # record_class 由配置 models.record_class 解析后注入，写入即产出自定义子类。
        self._record_class = record_class
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._logger = logger
        self._nlp = nlp

    def run(self, paths: List[str], user_id: Optional[str] = None) -> int:
        """批量导入多个文件，返回写入总条数；单文件失败不中断整批。"""
        total = 0
        for path in paths:
            try:
                total += self.ingest_file(path, user_id=user_id)
            except Exception as exc:  # noqa: BLE001 单文件失败应跳过而非中断
                self._logger.error("文件导入失败，已跳过", source_path=path, error=str(exc))
        return total

    def ingest_file(self, path: str, user_id: Optional[str] = None) -> int:
        """导入单个文件，返回写入条数。"""
        resolved_user_id = user_id or DEFAULT_USER_ID
        self._store.ensure_collection()
        text = parsers.parse_document(path, logger=self._logger)
        entity_list = entities_module.extract_entities(text, nlp=self._nlp)
        chunk_texts = chunking.build_semantic_chunks(
            text,
            max_chars=self._settings.chunking.max_chars,
            overlap_sents=self._settings.chunking.overlap_sents,
            nlp=self._nlp,
            logger=self._logger,
        )
        written = 0
        batch: List[MemoryRecord] = []
        for index, chunk_text in enumerate(chunk_texts):
            chunk = self._make_chunk(path, index, chunk_text, entity_list, resolved_user_id)
            batch.append(self._make_record(chunk))
            if len(batch) >= self._settings.embedding.batch_size:
                written += self._store.upsert(batch)
                batch = []
        if batch:
            written += self._store.upsert(batch)
        self._logger.info(
            "文件导入完成",
            source_path=path,
            user_id=resolved_user_id,
            chunks=len(chunk_texts),
            written=written,
        )
        return written

    def _make_chunk(
        self,
        path: str,
        index: int,
        text: str,
        entity_list: List[str],
        user_id: str,
    ) -> DocumentChunk:
        digest = self._content_hash(user_id, path, text)
        return DocumentChunk(
            chunk_id=digest + "-" + str(index),
            user_id=user_id,
            source_path=path,
            text=text,
            entities=list(entity_list),
            created_at=time.time(),
        )

    def _make_record(self, chunk: DocumentChunk) -> MemoryRecord:
        vector = self._embedder.encode([chunk.text])[0]
        # 使用配置的记录类构造记录，使自定义字段获得校验并随记录落库。
        return self._record_class(
            id=chunk.chunk_id,
            text_payload=chunk.text,
            vector=vector,
            entities=list(chunk.entities),
            created_at=chunk.created_at,
            user_id=chunk.user_id,
        )

    @staticmethod
    def _content_hash(user_id: str, source_path: str, text: str) -> str:
        """生成内容哈希，保证重复导入幂等。"""
        digest = hashlib.sha256()
        for part in (user_id, source_path, text):
            digest.update(part.encode("utf-8"))
            digest.update(b"|")
        return digest.hexdigest()
