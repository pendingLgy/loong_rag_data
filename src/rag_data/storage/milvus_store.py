# Milvus 向量库实现：记录模型、建表与写入。
#
# MilvusRecord 与建表逻辑同处本文件：表结构由记录类掌握，
# 子类覆盖 build_collection_schema 即可扩展或替换物理列。

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_data.config import Settings
from rag_data.exceptions import OptionalDependencyError, StoreError
from rag_data.logging.base import LoggerAdapter

DEFAULT_ALIAS = "rag_data"

# 记录显式声明的字段名。
BASE_FIELD_NAMES: Tuple[str, ...] = (
    "id",
    "text_payload",
    "vector",
    "entities",
    "created_at",
)

# 存储行中的向量列名。
VECTOR_FIELD = "vector"

# 各字段的标量索引类型；未列出的字段不建标量索引。
SCALAR_INDEX_TYPES: Dict[str, str] = {"entities": "INVERTED", "created_at": "STL_SORT"}

# 各向量索引类型的默认构建参数；未登记的类型由 Milvus 采用自身默认值。
VECTOR_INDEX_PARAMS: Dict[str, Dict[str, Any]] = {
    "HNSW": {"M": 16, "efConstruction": 200},
    "IVFLAT": {"nlist": 1024},
}


class MilvusRecord(BaseModel):
    """Milvus 记录模型：承载基础字段、存储行编解码与集合表结构生成。"""

    # extra 允许，构造时不拒绝未声明字段；未声明字段不落库。
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    text_payload: str = Field(min_length=1)
    vector: List[float] = Field(min_length=1)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)

    @field_validator("vector")
    @classmethod
    def _check_vector(cls, value: List[float]) -> List[float]:
        """保证向量非空；维度一致性在装配层结合 Settings 校验。"""
        if not value:
            raise ValueError("vector 不能为空")
        return value

    @classmethod
    def storage_fields(cls) -> List[str]:
        """返回存储行的平铺列名。"""
        return list(BASE_FIELD_NAMES)

    def to_storage_row(self) -> Dict[str, Any]:
        """记录到存储行：按平铺列展开。"""
        payload = self.model_dump()
        return {name: payload[name] for name in BASE_FIELD_NAMES if name in payload}

    @classmethod
    def from_storage_row(cls, row: Dict[str, Any]) -> "MilvusRecord":
        """存储行到记录：按平铺列还原。"""
        return cls(**{name: row[name] for name in BASE_FIELD_NAMES if name in row})

    @classmethod
    def build_collection_schema(cls, pymilvus: Any, vector_dim: int) -> Any:
        """按声明的字段创建集合表结构；子类覆盖即可扩展或替换。"""

        # 子类示例：
        #   class TenantRecord(MilvusRecord):
        #       user_id: str
        #
        #       @classmethod
        #       def build_collection_schema(cls, pymilvus, vector_dim):
        #           schema = super().build_collection_schema(pymilvus, vector_dim)
        #           schema.fields.append(pymilvus.FieldSchema(...))
        #           return schema
        data_type = pymilvus.DataType
        fields = [
            pymilvus.FieldSchema(name="id", dtype=data_type.VARCHAR, is_primary=True, max_length=64),
            pymilvus.FieldSchema(name="text_payload", dtype=data_type.VARCHAR, max_length=65535),
            pymilvus.FieldSchema(name="vector", dtype=data_type.FLOAT_VECTOR, dim=int(vector_dim)),
            # ARRAY 除元素个数外，元素为 VARCHAR 时还须声明元素最大长度。
            pymilvus.FieldSchema(
                name="entities",
                dtype=data_type.ARRAY,
                element_type=data_type.VARCHAR,
                max_capacity=64,
                max_length=256,
            ),
            pymilvus.FieldSchema(name="created_at", dtype=data_type.DOUBLE),
        ]
        return pymilvus.CollectionSchema(fields=fields, description="rag-data records for " + cls.__name__)


def import_pymilvus() -> Any:
    """导入 pymilvus，未安装时抛出可读异常。"""
    try:
        import pymilvus
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("pymilvus 未安装，请执行 pip install pymilvus") from exc
    return pymilvus


class MilvusVectorStore:
    """基于 Milvus 的实现：惰性建连，建表委托给 MilvusRecord。"""

    backend = "milvus"

    def __init__(
        self,
        settings: Settings,
        logger: LoggerAdapter,
        alias: str = DEFAULT_ALIAS,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._alias = alias
        self._collection: Optional[Any] = None
        self._connected = False

    # ---------------- 表结构 ----------------

    def column_names(self) -> List[str]:
        """返回集合中的平铺列名；未建表时只有基础字段可知。"""
        if self._collection is None:
            return list(BASE_FIELD_NAMES)
        fields = getattr(getattr(self._collection, "schema", None), "fields", None) or []
        return [name for name in (getattr(field, "name", None) for field in fields) if name]

    def vector_index_params(self) -> Dict[str, Any]:
        """返回向量索引参数：类型与度量取自配置，参数取该类型的默认值。"""
        index_type = self._settings.storage.index_type
        return {
            "index_type": index_type,
            "metric_type": self._settings.storage.metric,
            "params": dict(VECTOR_INDEX_PARAMS.get(index_type, {})),
        }

    def scalar_index_specs(self) -> List[Tuple[str, Dict[str, Any]]]:
        """返回 (字段名, 索引参数) 列表；未登记的字段不建标量索引。"""
        return [
            (name, {"index_type": SCALAR_INDEX_TYPES[name]})
            for name in self.column_names()
            if name in SCALAR_INDEX_TYPES
        ]

    def record_to_row(self, record: MilvusRecord) -> Dict[str, Any]:
        """记录转存储行。"""
        return record.to_storage_row()

    # ---------------- 连接与建表 ----------------

    def _connect(self) -> Any:
        """惰性建立连接，重复调用幂等。"""
        pymilvus = import_pymilvus()
        if self._connected:
            return pymilvus
        try:
            pymilvus.connections.connect(
                alias=self._alias,
                uri=self._settings.storage.milvus_uri,
                db_name=self._settings.storage.milvus_db,
            )
        except Exception as exc:  # noqa: BLE001 连接失败统一转领域异常
            raise StoreError("Milvus 连接失败：" + str(exc)) from exc
        self._connected = True
        self._logger.info(
            "Milvus 连接建立",
            uri=self._settings.storage.milvus_uri,
            db=self._settings.storage.milvus_db,
            alias=self._alias,
        )
        return pymilvus

    def _build_collection_schema(self, pymilvus: Any) -> Any:
        """建表委托给 MilvusRecord；子类覆盖 build_collection_schema 即可扩展表结构。"""
        return MilvusRecord.build_collection_schema(pymilvus, self._settings.storage.vector_dim)

    def _create_collection(self, pymilvus: Any, name: str) -> Any:
        """建表、建索引，返回集合实例。"""
        schema = self._build_collection_schema(pymilvus)
        collection = pymilvus.Collection(name=name, schema=schema, using=self._alias)
        collection.create_index(field_name=VECTOR_FIELD, index_params=self.vector_index_params())
        for field_name, index_params in self.scalar_index_specs():
            collection.create_index(field_name=field_name, index_params=index_params)
        self._logger.info("Milvus 集合创建完成", collection=name, columns=self.column_names())
        return collection

    def ensure_collection(self) -> None:
        """确保集合存在：不存在则建表建索引，重复调用幂等。"""
        pymilvus = self._connect()
        name = self._settings.storage.collection_name
        if pymilvus.utility.has_collection(name, using=self._alias):
            collection = pymilvus.Collection(name=name, using=self._alias)
            self._logger.debug("集合已存在，跳过建表", collection=name)
        else:
            collection = self._create_collection(pymilvus, name)
        self._collection = collection
        collection.load()

    def _require_collection(self) -> Any:
        """返回可用集合；未初始化时先建表。"""
        if self._collection is None:
            self.ensure_collection()
        return self._collection

    # ---------------- 写入 ----------------

    def _rows_to_columns(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """将行列表转为按列组织的写入结构。"""
        columns: Dict[str, List[Any]] = {}
        for row in rows:
            for key, value in row.items():
                columns.setdefault(key, []).append(value)
        return columns

    def upsert(self, records: List[MilvusRecord]) -> int:
        """按主键幂等写入，分批提交，返回写入条数。"""
        if not records:
            return 0
        collection = self._require_collection()
        batch_size = self._settings.embedding.batch_size
        written = 0
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            rows = [self.record_to_row(record) for record in chunk]
            try:
                collection.upsert(self._rows_to_columns(rows))
            except Exception as exc:  # noqa: BLE001 写入失败统一转领域异常
                raise StoreError("Milvus 写入失败：" + str(exc)) from exc
            written += len(chunk)
        collection.flush()
        self._logger.info("Milvus 写入完成", written=written, collection=self._settings.storage.collection_name)
        return written

    def close(self) -> None:
        """释放集合引用并断开连接。"""
        self._collection = None
        if not self._connected:
            return
        try:
            import_pymilvus().connections.disconnect(self._alias)
        except Exception as exc:  # noqa: BLE001 断开失败仅记录，不影响调用方
            self._logger.warning("Milvus 断开连接失败", error=str(exc))
        finally:
            self._connected = False
