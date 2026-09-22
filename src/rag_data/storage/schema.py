# 集合字段名常量、索引参数生成，以及记录与存储行之间的编解码。
#
# 建表字段由记录类自身声明（见 models.MemoryRecord.build_collection_schema），
# 本模块只维护字段名常量与索引参数，避免表结构出现两处定义。

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

from rag_data.exceptions import ConfigError
from rag_data.models import MemoryRecord

# 基类字段，与 MemoryRecord.build_collection_schema 硬编码的字段一一对应。
BASE_COLUMNS: List[str] = [
    "id",
    "text_payload",
    "vector",
    "entities",
    "created_at",
]

VECTOR_FIELD = "vector"
PRIMARY_FIELD = "id"
METADATA_FIELD = "metadata"
DEFAULT_METRIC = "COSINE"
DEFAULT_INDEX_TYPE = "HNSW"

# 各字段的标量索引类型；未列出的字段不建标量索引。
SCALAR_INDEX_TYPES: Dict[str, str] = {
    "entities": "INVERTED",
    "created_at": "STL_SORT",
}
SCALAR_INDEXES: List[str] = list(SCALAR_INDEX_TYPES)

# 向量索引的默认构建参数。
HNSW_INDEX_PARAMS: Dict[str, int] = {"M": 16, "efConstruction": 200}
IVFLAT_INDEX_PARAMS: Dict[str, int] = {"nlist": 1024}

# 允许提升为独立列的扩展字段类型。
PROMOTABLE_TYPES = ("VARCHAR", "DOUBLE", "INT64", "BOOL", "JSON")


def field_names() -> List[str]:
    """返回集合中的全部字段名，含扩展字段容器 metadata。"""
    return list(BASE_COLUMNS) + [METADATA_FIELD]


def is_base_field(name: str) -> bool:
    """判断字段名是否属于平铺列。"""
    return name in BASE_COLUMNS


def promoted_names(extra_columns: Optional[Sequence[Mapping[str, Any]]]) -> List[str]:
    """返回被提升为独立列的扩展字段名。"""
    return [str(item["name"]) for item in (extra_columns or ()) if item.get("name")]


def build_scalar_index_specs(field_names_in_collection: Sequence[str]) -> List[Dict[str, Any]]:
    """按字段名生成标量索引配置。"""
    specs: List[Dict[str, Any]] = []
    for name in field_names_in_collection:
        index_type = SCALAR_INDEX_TYPES.get(name)
        if index_type:
            specs.append({"field_name": name, "index_params": {"index_type": index_type}})
    return specs


def build_vector_index_params(index_type: str, metric: str) -> Dict[str, Any]:
    """生成向量索引参数。"""
    if index_type == "HNSW":
        params: Dict[str, Any] = dict(HNSW_INDEX_PARAMS)
    elif index_type == "IVFLAT":
        params = dict(IVFLAT_INDEX_PARAMS)
    else:
        raise ConfigError("不支持的向量索引类型：" + str(index_type))
    return {"index_type": index_type, "metric_type": metric, "params": params}


def metadata_fields(
    record: MemoryRecord,
    promoted: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """提取需写入 metadata 的扩展字段；已提升为独立列的字段不再重复入库。"""
    skip = set(BASE_COLUMNS) | set(promoted or ())
    return {key: value for key, value in record.to_row().items() if key not in skip}


def encode_metadata(extras: Dict[str, Any]) -> str:
    """将扩展字段序列化为 JSON 文本；Milvus 的 JSON 列按字符串写入。"""
    return json.dumps(extras, ensure_ascii=False, sort_keys=True)


def decode_metadata(raw: Any) -> Dict[str, Any]:
    """将 metadata 还原为字典，兼容字符串、字节与已解析的字典。"""
    if raw is None or raw == "" or raw == b"":
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        parsed = json.loads(raw)
        return dict(parsed) if isinstance(parsed, dict) else {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def to_storage_row(
    record: MemoryRecord,
    promoted: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """记录到存储行：基类字段与提升列平铺，其余扩展字段序列化进 metadata。"""
    payload = record.to_row()
    columns = list(BASE_COLUMNS) + [name for name in (promoted or ()) if name not in BASE_COLUMNS]
    row: Dict[str, Any] = {name: payload[name] for name in columns if name in payload}
    row[METADATA_FIELD] = encode_metadata(metadata_fields(record, promoted))
    return row


def from_storage_row(
    row: Dict[str, Any],
    record_class: Optional[Type[MemoryRecord]] = None,
    promoted: Optional[Sequence[str]] = None,
) -> MemoryRecord:
    """存储行到记录：提升列与 metadata 一并还原为扩展字段。"""
    cls = record_class or MemoryRecord
    columns = list(BASE_COLUMNS) + [name for name in (promoted or ()) if name not in BASE_COLUMNS]
    data: Dict[str, Any] = {name: row[name] for name in columns if name in row}
    data.update(decode_metadata(row.get(METADATA_FIELD)))
    return cls(**data)


def flatten_row(
    row: Dict[str, Any],
    promoted: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """将存储行摊平为单层字典，便于按字段过滤。"""
    columns = list(BASE_COLUMNS) + [name for name in (promoted or ()) if name not in BASE_COLUMNS]
    merged: Dict[str, Any] = {name: row[name] for name in columns if name in row}
    merged.update(decode_metadata(row.get(METADATA_FIELD)))
    return merged


def row_matches(
    row: Dict[str, Any],
    filters: Dict[str, Any],
    promoted: Optional[Sequence[str]] = None,
) -> bool:
    """在存储行上做字段精确匹配，涵盖基类字段、提升列与扩展字段。"""
    if not filters:
        return True
    merged = flatten_row(row, promoted)
    return all(merged.get(key) == value for key, value in filters.items())
