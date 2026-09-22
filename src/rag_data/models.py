# pydantic 数据模型，作为跨层数据契约。

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Protocol, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_data.exceptions import ConfigError

# 基类显式声明的字段名；之外的一律视为扩展字段。
BASE_FIELD_NAMES: Tuple[str, ...] = (
    "id",
    "text_payload",
    "vector",
    "entities",
    "created_at",
)

# 默认记录类的点分路径；可通过配置 models.record_class 覆盖。
DEFAULT_RECORD_CLASS_PATH = "rag_data.models.MemoryRecord"

class MilvusDataType(Protocol):
    """pymilvus.DataType 的结构约束；其成员为整型枚举值。"""

    VARCHAR: int
    FLOAT_VECTOR: int
    ARRAY: int
    DOUBLE: int
    INT64: int
    BOOL: int
    JSON: int


class MilvusFieldSchema(Protocol):
    """pymilvus.FieldSchema 实例的最小接口。"""

    name: str


class MilvusCollectionSchema(Protocol):
    """pymilvus.CollectionSchema 实例的最小接口。"""

    fields: List[MilvusFieldSchema]


class MilvusModule(Protocol):
    """pymilvus 模块中建表所需的最小接口。"""

    # 以 Protocol 描述外部结构，既不在运行期依赖可选库 pymilvus，
    # 也无需在签名中使用 Any。
    DataType: Type[MilvusDataType]

    def FieldSchema(self, name: str, dtype: int, **kwargs: object) -> MilvusFieldSchema:
        ...

    def CollectionSchema(
        self, fields: List[MilvusFieldSchema], description: str = ...
    ) -> MilvusCollectionSchema:
        ...


class DocumentChunk(BaseModel):
    """切块后的中间态。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)


class MemoryRecord(BaseModel):
    """写入向量库的最终形态，可作为基类被继承以扩展字段。"""

    # extra 设为 allow，使其可继承、可扩展：
    # 1. 子类可声明类型化字段，例如 user_id、tags，随记录一并存储
    # 2. 也可直接传入未声明字段，落入扩展字段集合
    # 扩展字段统一通过 extra_fields 或 to_row 获取，便于存储层落库。
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

    @property
    def extra_fields(self) -> Dict[str, Any]:
        """返回基类字段之外的扩展字段（含子类声明与动态传入）。"""
        return {
            key: value
            for key, value in self.model_dump().items()
            if key not in BASE_FIELD_NAMES
        }

    def to_row(self) -> Dict[str, Any]:
        """展开为存储层可用的扁平字典，含扩展字段。"""
        return self.model_dump()

    @classmethod
    def build_collection_schema(
        cls, pymilvus: MilvusModule, vector_dim: int
    ) -> MilvusCollectionSchema:
        """按 MemoryRecord 声明的字段硬编码创建集合表结构。"""

        # 子类可覆盖本方法以扩展或替换表结构，示例：
        #   class TenantRecord(MemoryRecord):
        #       user_id: str
        #
        #       @classmethod
        #       def build_collection_schema(cls, pymilvus, vector_dim):
        #           schema = super().build_collection_schema(pymilvus, vector_dim)
        #           schema.fields.append(pymilvus.FieldSchema(
        #               name="user_id", dtype=pymilvus.DataType.VARCHAR, max_length=128
        #           ))
        #           return schema
        #
        # pymilvus 由调用方传入，避免模型层依赖可选第三方库。
        if vector_dim <= 0:
            raise ConfigError("向量维度需大于 0：" + str(vector_dim))
        data_type = pymilvus.DataType
        fields = [
            pymilvus.FieldSchema(
                name="id", dtype=data_type.VARCHAR, is_primary=True, max_length=64
            ),
            pymilvus.FieldSchema(name="text_payload", dtype=data_type.VARCHAR, max_length=65535),
            pymilvus.FieldSchema(name="vector", dtype=data_type.FLOAT_VECTOR, dim=int(vector_dim)),
            pymilvus.FieldSchema(
                name="entities",
                dtype=data_type.ARRAY,
                element_type=data_type.VARCHAR,
                max_capacity=64,
            ),
            pymilvus.FieldSchema(name="created_at", dtype=data_type.DOUBLE),
            pymilvus.FieldSchema(name="metadata", dtype=data_type.JSON),
        ]
        return pymilvus.CollectionSchema(
            fields=fields, description="rag-data records for " + cls.__name__
        )


class QueryHit(BaseModel):
    """召回返回的候选。"""

    id: str = Field(min_length=1)
    text_payload: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)


def resolve_record_class(path: str = DEFAULT_RECORD_CLASS_PATH) -> Type[MemoryRecord]:
    """按点分路径导入记录类，并校验其为 MemoryRecord 的子类。"""
    # 扩展字段只需两步：继承 MemoryRecord 定义字段，再把类路径写入配置。
    if not path or not isinstance(path, str):
        raise ConfigError("记录类路径不能为空")
    module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ConfigError("记录类路径需为点分形式，如 myapp.models.TenantRecord：" + path)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError("无法导入记录类所在模块：" + module_path + "（" + str(exc) + "）") from exc
    candidate = getattr(module, attr, None)
    if candidate is None:
        raise ConfigError("模块 " + module_path + " 中不存在属性 " + attr)
    if not (isinstance(candidate, type) and issubclass(candidate, MemoryRecord)):
        raise ConfigError("记录类需继承 MemoryRecord：" + path)
    return candidate
