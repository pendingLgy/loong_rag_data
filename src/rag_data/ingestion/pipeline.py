# 离线导入编排：解析、切块、实体抽取与向量化。

from __future__ import annotations

import hashlib
import time
from typing import Any, List, Optional

from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.ingestion import entities as entities_module
from rag_data import parsers
from rag_data.logging.base import LoggerAdapter
from rag_data.models import DocumentChunk

DEFAULT_USER_ID = "default"


class IngestionPipeline:
    """离线导入管道：解析、切块、实体抽取并向量化，产出携带向量的切块。"""

    def __init__(
        self,
        embedder: Embedder,
        settings: Settings,
        logger: LoggerAdapter,
        nlp: Optional[Any] = None,
    ) -> None:
        self._embedder = embedder
        self._settings = settings
        self._logger = logger
        self._nlp = nlp

    def run(self, paths: List[str], user_id: Optional[str] = None) -> List[DocumentChunk]:
        """批量向量化多个文件，返回携带向量的切块；单文件失败不中断整批。"""
        collected: List[DocumentChunk] = []
        for path in paths:
            try:
                collected.extend(self.ingest_file(path, user_id=user_id))
            except Exception as exc:  # noqa: BLE001 单文件失败应跳过而非中断
                self._logger.error("文件导入失败，已跳过", source_path=path, error=str(exc))
        return collected

    def ingest_file(self, path: str, user_id: Optional[str] = None) -> List[DocumentChunk]:
        """向量化单个文件，返回携带向量的切块。"""
        resolved_user_id = user_id or DEFAULT_USER_ID
        text = parsers.parse_document(path, logger=self._logger)
        entity_list = entities_module.extract_entities(text, nlp=self._nlp)
        chunk_texts = parsers.chunk_document(
            path,
            text,
            max_chars=self._settings.chunking.max_chars,
            overlap_sents=self._settings.chunking.overlap_sents,
            nlp=self._nlp,
            logger=self._logger,
        )
        chunks = [
            self._make_chunk(path, index, chunk_text, entity_list, resolved_user_id)
            for index, chunk_text in enumerate(chunk_texts)
        ]
        embedded = self._embed_chunks(chunks)
        self._logger.info(
            "文件向量化完成",
            source_path=path,
            user_id=resolved_user_id,
            chunks=len(embedded),
        )
        return embedded

    def _embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """按 embedding.batch_size 分批编码，把向量回填到切块上。"""
        embedded: List[DocumentChunk] = []
        step = self._settings.embedding.batch_size
        for start in range(0, len(chunks), step):
            window = chunks[start:start + step]
            vectors = self._embedder.encode([chunk.text for chunk in window])
            embedded.extend(
                chunk.model_copy(update={"vector": vector})
                for chunk, vector in zip(window, vectors)
            )
        return embedded

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

    @staticmethod
    def _content_hash(user_id: str, source_path: str, text: str) -> str:
        """生成内容哈希，保证重复导入幂等。"""
        digest = hashlib.sha256()
        for part in (user_id, source_path, text):
            digest.update(part.encode("utf-8"))
            digest.update(b"|")
        return digest.hexdigest()
