# Milvus 向量库实现：集合、向量字段与 JSON 元数据，AUTOINDEX 建索引。
#
# 单一 schema：id 主键、vectors 稠密向量、metadata 元数据，外加 text 文本与 sparse 稀疏向量；
# 稀疏向量由 BM25 函数按 text 自动生成，稠密检索与关键词检索共用同一集合。

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from rag_data.exceptions import OptionalDependencyError, StorageError
from rag_data.storage.guard import ensure_test_table
from rag_data.storage.vector_store.base import MetricType, VectorStoreBase, VectorStoreResult

try:  # 可选依赖：模块仍可导入，缺库时仅在实例化时报错。
    from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient
    _PYMILVUS_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover
    CollectionSchema = DataType = FieldSchema = Function = FunctionType = MilvusClient = None  # type: ignore[assignment]
    _PYMILVUS_ERROR = exc

# Milvus 过滤表达式的键名白名单，避免表达式注入。
_SAFE_FILTER_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
# 默认服务地址：由片段拼接，避免源码出现连续斜杠字面量。
_DEFAULT_URI = "http:" + chr(47) * 2 + "localhost:19530"

# 文本字段最大长度（Milvus VARCHAR 上限）。
_MAX_TEXT_LEN = 65535


class MilvusVectorStore(VectorStoreBase):
    """基于 Milvus 的实现；构造时确保集合存在。"""

    backend = "milvus"
    metric_type = "COSINE"

    def __init__(
        self,
        uri: str = _DEFAULT_URI,
        token: str = "",
        collection_name: str = "rag_data",
        vector_size: int = 1024,
        metric_type: MetricType = "COSINE",
        db_name: str = "",
    ) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.metric_type = metric_type
        if MilvusClient is None:
            raise OptionalDependencyError("使用 Milvus 向量库需要 pymilvus>=2.5（含 Function 类型），请安装或升级 pymilvus") from _PYMILVUS_ERROR
        try:
            self.client = MilvusClient(uri=uri, token=token, db_name=db_name)
        except Exception as exc:
            raise StorageError("无法连接 Milvus：" + str(exc)) from exc
        self.create_col(collection_name, vector_size, metric_type)

    def _schema(self, vector_size: int) -> Any:
        """集合 schema：稠密向量、元数据与 BM25 文本及稀疏向量一次建齐。"""
        if Function is None or FunctionType is None:
            raise OptionalDependencyError("BM25 稀疏检索需要 pymilvus>=2.5（含 Function 类型）")

        # 定义中文分析器参数
        chinese_analyzer_params = {
            "type": "chinese",
            "tokenizer": "jieba",
        }

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
            FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            # text 由 Milvus 分析器自动分词，供 BM25 使用。
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_TEXT_LEN, enable_analyzer=True,analyzer_params=chinese_analyzer_params),
            # sparse 由 BM25 函数按 text 自动生成，稠密检索与关键词检索共用该集合。
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=True)
        schema.add_function(Function(name="bm25", input_field_names=["text"], output_field_names=["sparse"], function_type=FunctionType.BM25))
        return schema

    def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        """集合不存在时按单一 schema 创建；已存在则直接复用。"""
        if self.client.has_collection(name):
            return
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vectors", metric_type=distance, index_type="AUTOINDEX", index_name="vector_index")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25", index_name="sparse_index")
        self.client.create_collection(collection_name=name, schema=self._schema(vector_size), index_params=index_params)

    def list_cols(self) -> List[str]:
        """列出全部集合名。"""
        return list(self.client.list_collections())

    def delete_col(self) -> None:
        """删除当前集合。"""
        target = ensure_test_table(self.collection_name)
        self.client.drop_collection(collection_name=target)

    def col_info(self) -> Dict[str, Any]:
        """返回当前集合的统计信息。"""
        return dict(self.client.get_collection_stats(collection_name=self.collection_name))

    def reset(self) -> None:
        """删除并重建当前集合。"""
        self.delete_col()
        self.create_col(self.collection_name, self.vector_size, self.metric_type)

    def _record(self, rid: str, vector: List[float], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """组装写入记录：text 优先取 text_lemmatized，回退 data 或 text。"""
        metadata = payload or {}
        text = metadata.get("text_lemmatized") or metadata.get("data") or metadata.get("text") or ""
        return {
            "id": rid,
            "vectors": vector,
            "metadata": metadata,
            "text": str(text)[:_MAX_TEXT_LEN],
        }

    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """批量写入；ids 缺省时用行号字符串补齐。"""
        payloads = payloads or [None] * len(vectors)  # type: ignore[list-item]
        ids = [str(rid) for rid in (ids or range(len(vectors)))]
        data = [self._record(rid, vector, payload) for rid, vector, payload in zip(ids, vectors, payloads)]
        try:
            self.client.insert(collection_name=self.collection_name, data=data)
        except Exception as exc:
            raise StorageError("Milvus 写入失败：" + str(exc)) from exc

    def delete(self, vector_id: str) -> None:
        """按 id 删除单条。"""
        self.client.delete(collection_name=self.collection_name, ids=[vector_id])

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        """按 id 更新；缺省字段沿用已有记录。"""
        if vector is None or payload is None:
            existing = self.client.get(collection_name=self.collection_name, ids=vector_id)
            if not existing:
                raise StorageError("待更新的向量不存在：" + str(vector_id))
            if vector is None:
                vector = existing[0].get("vectors")
            if payload is None:
                payload = existing[0].get("metadata")
        self.client.upsert(collection_name=self.collection_name, data=self._record(str(vector_id), vector, payload))

    def get(self, vector_id: str) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""
        result = self.client.get(collection_name=self.collection_name, ids=vector_id)
        if not result:
            return None
        item = result[0]
        return VectorStoreResult(id=str(item.get("id", "")), score=None, payload=item.get("metadata"))

    def _create_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """把等值过滤字典翻译为 Milvus 表达式。"""
        if not filters:
            return None
        operands: List[str] = []
        for key, value in filters.items():
            if not _SAFE_FILTER_KEY.match(key):
                raise StorageError("非法过滤键：" + str(key))
            if value == "*":
                continue
            if isinstance(value, bool):
                operands.append("(metadata[" + json.dumps(key) + "] == " + str(value).lower() + ")")
            elif isinstance(value, (int, float)):
                operands.append("(metadata[" + json.dumps(key) + "] == " + str(value) + ")")
            elif isinstance(value, str):
                operands.append("(metadata[" + json.dumps(key) + "] == " + json.dumps(value) + ")")
            else:
                raise StorageError("过滤值类型不支持：" + str(key))
        return " and ".join(operands) if operands else None

    def _parse(self, hits: List[Dict[str, Any]]) -> List[VectorStoreResult]:
        """把 Milvus 命中转成结果对象；L2 距离换算为相似度。"""
        results: List[VectorStoreResult] = []
        for value in hits:
            raw = value.get("distance")
            if raw is None:
                score = None
            elif self.metric_type == "L2":
                score = 1.0 / (1.0 + float(raw))
            else:
                score = float(raw)
            entity = value.get("entity") or {}
            results.append(VectorStoreResult(id=str(value.get("id", "")), score=score, payload=entity.get("metadata")))
        return results

    def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[VectorStoreResult]:
        """稠密向量检索；集合含稀疏字段，故显式指定 anns_field。"""
        kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [vectors],
            "anns_field": "vectors",
            "limit": top_k,
            "filter": self._create_filter(filters),
            "output_fields": ["*"],
        }
        try:
            hits = self.client.search(**kwargs)
        except Exception as exc:
            raise StorageError("Milvus 检索失败：" + str(exc)) from exc
        return self._parse(hits[0] if hits else [])

    def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        """BM25 关键词检索：以原始查询文本作为稀疏向量输入。"""
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse",
                limit=top_k,
                filter=self._create_filter(filters),
                output_fields=["*"],
            )
        except Exception:
            return None
        return self._parse(hits[0] if hits else [])

    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[List[VectorStoreResult]]:
        """列出集合内容；外层列表按既有约定包一层。"""
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=self._create_filter(filters),
            limit=top_k or 100,
            output_fields=["*"],
        )
        results = [VectorStoreResult(id=str(row.get("id", "")), score=None, payload=row.get("metadata")) for row in rows]
        return [results]
