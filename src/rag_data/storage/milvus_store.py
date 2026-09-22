# Milvus 向量库实现：建表、写入与检索。
#
# 表结构由记录类掌握：MemoryRecord.build_collection_schema 硬编码基础字段，
# 子类可覆盖该方法自定义表结构；配置中的 extra_columns 用于追加提升列。

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

from rag_data.config import Settings
from rag_data.exceptions import OptionalDependencyError, SchemaMismatchError, StoreError
from rag_data.logging.base import LoggerAdapter
from rag_data.models import MemoryRecord, QueryHit, resolve_record_class
from rag_data.storage import schema
from rag_data.storage.base import BaseVectorStore

# metadata JSON 列的字段访问路径模板，供布尔表达式使用。
METADATA_PATH = "metadata[\"{}\"]"

DEFAULT_ALIAS = "rag_data"

# 检索时需回传的字段；主键由 Milvus 默示返回。
OUTPUT_FIELDS: List[str] = ["text_payload", "entities", "created_at"]

# 检索参数：HNSW 使用 ef，IVFLAT 使用 nprobe。
HNSW_SEARCH_PARAMS: Dict[str, int] = {"ef": 64}
IVFLAT_SEARCH_PARAMS: Dict[str, int] = {"nprobe": 16}


def import_pymilvus() -> Any:
    """导入 pymilvus，未安装时抛出可读异常。"""
    try:
        import pymilvus
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("pymilvus 未安装，请执行 pip install pymilvus") from exc
    return pymilvus


def format_literal(value: Any) -> str:
    """将 Python 值转为过滤表达式中的字面量。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # TODO: 生产实现应补充更完整的转义与注入防护。
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return "\"" + escaped + "\""
    if isinstance(value, (list, tuple)):
        inner = ", ".join(format_literal(item) for item in value)
        return "[" + inner + "]"
    raise TypeError("不支持的过滤值类型：" + type(value).__name__)


def build_filter_expr(
    filters: Optional[Dict[str, Any]],
    column_names: Optional[Sequence[str]] = None,
) -> str:
    """构造 Milvus 布尔过滤表达式，支持基类字段、提升列与扩展字段。"""
    if not filters:
        return ""
    promoted = set(column_names) if column_names else set()
    clauses: List[str] = []
    for key, value in filters.items():
        if schema.is_base_field(key) or key in promoted:
            clauses.append(key + " == " + format_literal(value))
        else:
            clauses.append(METADATA_PATH.format(key) + " == " + format_literal(value))
    return " and ".join(clauses)


class MilvusVectorStore(BaseVectorStore):
    """基于 Milvus 的实现：惰性建连，建表委托给记录类。"""

    backend = "milvus"

    def __init__(
        self,
        settings: Settings,
        logger: LoggerAdapter,
        record_class: Optional[Type[MemoryRecord]] = None,
        extra_columns: Optional[Sequence[Dict[str, Any]]] = None,
        alias: str = DEFAULT_ALIAS,
) -> None:
        self._settings = settings
        self._logger = logger
        self._record_class = record_class or resolve_record_class(settings.models.record_class)
        self._extra_columns: List[Dict[str, Any]] = [dict(item) for item in (extra_columns or [])]
        self._alias = alias
        self._collection: Optional[Any] = None
        self._connected = False

    # ---------------- 表结构（声明式，可单测） ----------------

    def column_names(self) -> List[str]:
        """返回集合中除 metadata 之外的平铺列名。"""

        # 已建表时以集合实际字段为准，可识别记录类覆盖 build_collection_schema
        # 动态新增的列；未建表时按基类字段与配置提升列推算。
        if self._collection is not None:
            fields = getattr(getattr(self._collection, "schema", None), "fields", None) or []
            names = [getattr(field, "name", None) for field in fields]
            return [name for name in names if name and name != schema.METADATA_FIELD]
        return list(schema.BASE_COLUMNS) + self.promoted_names()

    def promoted_columns(self) -> List[str]:
        """返回除基类字段外的平铺列名：含记录类覆盖新增与配置提升的列。"""
        return [name for name in self.column_names() if not schema.is_base_field(name)]

    def vector_index_params(self) -> Dict[str, Any]:
        """返回向量索引参数，索引类型与度量取自配置。"""
        return schema.build_vector_index_params(self._settings.storage.index_type, self._settings.storage.metric)

    def scalar_index_specs(self) -> List[Dict[str, Any]]:
        """返回标量索引配置。"""
        names = self.column_names() + [schema.METADATA_FIELD]
        return schema.build_scalar_index_specs(names)

    def search_params(self) -> Dict[str, Any]:
        """返回检索参数，随索引类型变化。"""
        index_type = self._settings.storage.index_type
        params = dict(HNSW_SEARCH_PARAMS) if index_type == "HNSW" else dict(IVFLAT_SEARCH_PARAMS)
        return {"metric_type": self._settings.storage.metric, "params": params}

    # ---------------- 映射逻辑（已实现，可单测） ----------------

    def promoted_names(self) -> List[str]:
        """返回被提升为独立列的扩展字段名。"""
        return schema.promoted_names(self._extra_columns)

    def record_to_row(self, record: MemoryRecord) -> Dict[str, Any]:
        """记录转存储行：基类字段与提升列平铺，其余扩展字段序列化进 metadata。"""
        return schema.to_storage_row(record, self.promoted_columns())

    @staticmethod
    def row_to_hit(row: Dict[str, Any], score: float) -> QueryHit:
        """存储行转检索结果。"""
        return QueryHit(
            id=row[schema.PRIMARY_FIELD],
            text_payload=row["text_payload"],
            score=score,
            entities=list(row.get("entities", [])),
            created_at=row["created_at"],
        )

    def row_to_record(self, row: Dict[str, Any]) -> MemoryRecord:
        """存储行还原为记录，扩展字段完整回填至配置的记录类。"""
        return schema.from_storage_row(row, self._record_class, self.promoted_columns())

    @staticmethod
    def normalize_score(distance: float, metric: str) -> float:
        """将 Milvus 距离归一化到 0 到 1，与内存实现保持一致。"""
        if metric == "COSINE":
            raw = (distance + 1.0) / 2.0
        elif metric == "IP":
            raw = distance
        else:
            raw = 1.0 / (1.0 + max(0.0, distance))
        return max(0.0, min(1.0, raw))

    # ---------------- 连接与建表 ----------------

    def _connect(self) -> Any:
        """惰性建立连接，重复调用幂等。"""
        pymilvus = import_pymilvus()
        if self._connected:
            return pymilvus
        try:
            pymilvus.connections.connect(alias=self._alias, uri=self._settings.storage.milvus_uri)
        except Exception as exc:  # noqa: BLE001 连接失败统一转领域异常
            raise StoreError("Milvus 连接失败：" + str(exc)) from exc
        self._connected = True
        self._logger.info("Milvus 连接建立", uri=self._settings.storage.milvus_uri, alias=self._alias)
        return pymilvus

    def _has_collection(self, pymilvus: Any) -> bool:
        """判断集合是否已存在。"""
        return bool(pymilvus.utility.has_collection(self._settings.storage.collection_name, using=self._alias))

    def _build_collection_schema(self, pymilvus: Any) -> Any:
        """建表委托给记录类；默认使用 MemoryRecord 声明的字段。"""

        # 继承 MemoryRecord 的类可覆盖 build_collection_schema 自定义表结构；
        # 配置中声明的提升列在此追加为独立物理列。
        collection_schema = self._record_class.build_collection_schema(
            pymilvus, self._settings.storage.vector_dim
        )
        if self._extra_columns:
            fields = list(getattr(collection_schema, "fields", []) or [])
            for column in self._extra_columns:
                fields.append(self._promote_field(pymilvus, column))
            collection_schema.fields = fields
        return collection_schema

    @staticmethod
    def _promote_field(pymilvus: Any, column: Dict[str, Any]) -> Any:
        """将配置声明的提升列转为 FieldSchema，仅支持标量类型。"""
        data_type = pymilvus.DataType
        field_type = column.get("type")
        kwargs: Dict[str, Any] = {}
        if field_type == "VARCHAR":
            dtype = data_type.VARCHAR
            kwargs["max_length"] = int(column.get("max_length", 65535))
        elif field_type == "INT64":
            dtype = data_type.INT64
        elif field_type == "DOUBLE":
            dtype = data_type.DOUBLE
        elif field_type == "BOOL":
            dtype = data_type.BOOL
        elif field_type == "JSON":
            dtype = data_type.JSON
        else:
            raise SchemaMismatchError("不支持的提升列类型：" + str(field_type))
        return pymilvus.FieldSchema(name=column["name"], dtype=dtype, **kwargs)

    def _verify_vector_dim(self, collection: Any) -> None:
        """校验既有集合的向量维度与配置一致。"""
        expected = self._settings.storage.vector_dim
        fields = getattr(getattr(collection, "schema", None), "fields", None) or []
        for field in fields:
            if getattr(field, "name", None) != schema.VECTOR_FIELD:
                continue
            params = getattr(field, "params", {}) or {}
            dim = params.get("dim") if hasattr(params, "get") else None
            if dim is not None and int(dim) != expected:
                raise SchemaMismatchError("向量维度不一致：集合为 " + str(dim) + "，配置为 " + str(expected))

    def _create_collection(self, pymilvus: Any, name: str) -> Any:
        """建表、建索引，返回集合实例。"""
        collection = pymilvus.Collection(name=name, schema=self._build_collection_schema(pymilvus), using=self._alias)
        collection.create_index(field_name=schema.VECTOR_FIELD, index_params=self.vector_index_params())
        for spec in self.scalar_index_specs():
            collection.create_index(field_name=spec["field_name"], index_params=spec["index_params"])
        self._logger.info(
            "Milvus 集合创建完成",
            collection=name,
            record_class=self._record_class.__name__,
            columns=self.column_names(),
            promoted=[item["name"] for item in self._extra_columns],
        )
        return collection

    def ensure_collection(self) -> None:
        """确保集合存在：不存在则建表建索引，存在则校验维度，重复调用幂等。"""
        pymilvus = self._connect()
        name = self._settings.storage.collection_name
        if self._has_collection(pymilvus):
            collection = pymilvus.Collection(name=name, using=self._alias)
            self._verify_vector_dim(collection)
            self._collection = collection
            collection.load()
            self._logger.debug("集合已存在，跳过建表", collection=name)
            return
        collection = self._create_collection(pymilvus, name)
        self._collection = collection
        collection.load()

    def _require_collection(self) -> Any:
        """返回可用集合；未初始化时先建表。"""
        if self._collection is None:
            self.ensure_collection()
        return self._collection

    # ---------------- 写入与检索 ----------------

    def _to_milvus_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """将存储行的 metadata 由 JSON 文本转为字典，符合 Milvus JSON 列的写入要求。"""
        converted = dict(row)
        converted[schema.METADATA_FIELD] = schema.decode_metadata(row.get(schema.METADATA_FIELD))
        return converted

    def _rows_to_columns(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """将行列表转为按列组织的写入结构。"""
        columns: Dict[str, List[Any]] = {}
        for row in rows:
            for key, value in row.items():
                columns.setdefault(key, []).append(value)
        return columns

    def _validate_rows(self, records: Sequence[MemoryRecord]) -> None:
        """校验每条记录的向量维度与配置一致。"""
        expected = self._settings.storage.vector_dim
        for record in records:
            if len(record.vector) != expected:
                raise SchemaMismatchError("向量维度不一致：记录为 " + str(len(record.vector)) + "，配置为 " + str(expected))

    def upsert(self, records: List[MemoryRecord]) -> int:
        """按主键幂等写入，分批提交，返回写入条数。"""
        if not records:
            return 0
        collection = self._require_collection()
        self._validate_rows(records)
        batch_size = self._settings.embedding.batch_size
        written = 0
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            rows = [self._to_milvus_row(self.record_to_row(record)) for record in chunk]
            try:
                collection.upsert(self._rows_to_columns(rows))
            except Exception as exc:  # noqa: BLE001 写入失败统一转领域异常
                raise StoreError("Milvus 写入失败：" + str(exc)) from exc
            written += len(chunk)
        collection.flush()
        self._logger.info("Milvus 写入完成", written=written, collection=self._settings.storage.collection_name)
        return written

    def query(
        self,
        vector: List[float],
        top_n: int,
        filters: Optional[Dict[str, Any]] = None,
) -> List[QueryHit]:
        """按向量检索，filters 转为布尔表达式，结果按相似度降序返回。"""
        collection = self._require_collection()
        columns = self.column_names()
        promoted = self.promoted_columns()
        expr = build_filter_expr(filters, column_names=columns)
        try:
            results = collection.search(
                data=[list(vector)],
                anns_field=schema.VECTOR_FIELD,
                param=self.search_params(),
                limit=top_n,
                expr=expr or None,
                output_fields=OUTPUT_FIELDS + promoted,
            )
        except Exception as exc:  # noqa: BLE001 检索失败统一转领域异常
            raise StoreError("Milvus 检索失败：" + str(exc)) from exc
        return self._to_hits(results)

    def _to_hits(self, results: Any) -> List[QueryHit]:
        """将 Milvus 检索结果转为 QueryHit 列表。"""
        metric = self._settings.storage.metric
        hits: List[QueryHit] = []
        for result in results or []:
            for item in result:
                entity = dict(getattr(item, "entity", {}) or {})
                entity[schema.PRIMARY_FIELD] = getattr(item, "id")
                score = self.normalize_score(float(getattr(item, "distance", 0.0)), metric)
                hits.append(self.row_to_hit(entity, score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def close(self) -> None:
        """释放集合引用并断开连接。"""
        self._collection = None
        if not self._connected:
            return
        try:
            pymilvus = import_pymilvus()
            pymilvus.connections.disconnect(self._alias)
        except Exception as exc:  # noqa: BLE001 断开失败仅记录，不影响调用方
            self._logger.warning("Milvus 断开连接失败", error=str(exc))
        finally:
            self._connected = False
