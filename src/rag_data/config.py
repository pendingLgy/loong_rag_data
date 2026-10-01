# 配置模型：按模块分区，默认读取环境变量，也支持在代码中硬编码。
#
# 优先级由高到低：
#   1. 代码硬编码：Settings.load(storage={"backend": "milvus"})
#   2. 环境变量：RAG_<分区>__<字段>，如 RAG_STORAGE__VECTOR_DIM=1024
#   3. 字段默认值
#
# 环境变量命名规则：RAG_ 前缀，分区名与字段名大写，双下划线分隔，例如：
#   RAG_STORAGE__BACKEND=milvus
#   RAG_STORAGE__VECTOR_DIM=1024
#   RAG_STORAGE__STORE_MODULES=["myapp.stores", "my_stores.py"]
#   RAG_LOGGING__LEVEL=DEBUG
#   RAG_LOGGING__JSON=true

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_data.exceptions import ConfigError

SECTION_STORAGE = "storage"
SECTION_CHUNKING = "chunking"
SECTION_EMBEDDING = "embedding"
SECTION_NLP = "nlp"
SECTION_LOGGING = "logging"

MODULE_SECTIONS: List[str] = [
    SECTION_STORAGE,
    SECTION_CHUNKING,
    SECTION_EMBEDDING,
    SECTION_NLP,
    SECTION_LOGGING,
]

# 环境变量前缀与分区分隔符，与 SettingsConfigDict 的 env_prefix、env_nested_delimiter 一致。
ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"


class StorageSettings(BaseModel):
    """向量库相关配置。"""

    model_config = ConfigDict(extra="forbid")

    # 后端名对应存储后端注册表中的键；内置 milvus，
    # 可用 rag_data.available_backends() 查看。
    backend: str = Field(default="milvus")
    # 额外的存储后端模块：点分模块路径（myapp.stores）或以 .py 结尾的文件路径。
    # 模块在导入时调用 register_store 或 register_backend 完成登记；
    # 装配向量库时按需导入，重复导入幂等。
    store_modules: List[str] = Field(default_factory=list)
    milvus_uri: str = Field(default="localhost:19530")
    # 目标数据库名。Milvus 支持多库隔离，需与 collection_name 所在库一致；
    # 默认库名为 default，可用自定义库隔离不同项目的数据。
    milvus_db: str = Field(default="default")
    collection_name: str = Field(default="memory_store")
    vector_dim: int = Field(default=1024, gt=0)
    metric: Literal["COSINE", "L2", "IP"] = "COSINE"
    index_type: Literal["HNSW", "IVFLAT"] = "HNSW"

class ChunkingSettings(BaseModel):
    """两阶段语义切块参数。"""

    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=250, ge=150, le=300)
    overlap_sents: int = Field(default=1, ge=0)

class EmbeddingSettings(BaseModel):
    """向量化参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # provider 名对应嵌入模型注册表中的键；继承 BaseEmbeddingProvider 即自动注册，
    # 内置 openai 与 qwen，可用 rag_data.available_embedding_providers() 查看。
    provider: str = Field(default="openai")
    # 模型名。留空时用 provider 的默认模型，见各 provider 的 default_model。
    model: str = Field(default="")
    batch_size: int = Field(default=128, gt=0)
    # 期望输出维度，0 表示用 provider 默认维度；需与 storage.vector_dim 对齐。
    dim: int = Field(default=0, ge=0)
    # API Key。留空时读取 provider 约定的环境变量，如 OPENAI_API_KEY、DASHSCOPE_API_KEY。
    api_key: str = Field(default="")
    # 接口地址。留空时用 provider 默认地址，可指向自建网关或代理。
    base_url: str = Field(default="")
    timeout: float = Field(default=60.0, gt=0)

class NLPSettings(BaseModel):
    """分句与实体抽取参数。"""

    model_config = ConfigDict(extra="forbid")

    spacy_model: str = Field(default="zh_core_web_sm")

class LoggingSettings(BaseModel):
    """日志适配层参数。json 为保留名，故字段名为 json_output 并设置别名。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    backend: Literal["stdlib", "loguru", "structlog", "auto"] = "auto"
    level: str = Field(default="INFO")
    json_output: bool = Field(default=False, alias="json")

# 分区名到分区模型的映射。
# 关闭环境变量时用它补齐字段默认值，使构造实参覆盖全部字段。
SECTION_MODEL_CLASSES: Dict[str, Type[BaseModel]] = {
    SECTION_STORAGE: StorageSettings,
    SECTION_CHUNKING: ChunkingSettings,
    SECTION_EMBEDDING: EmbeddingSettings,
    SECTION_NLP: NLPSettings,
    SECTION_LOGGING: LoggingSettings,
}

class Settings(BaseSettings):
    """根配置：按模块分区的嵌套模型，默认读取 RAG_ 前缀的环境变量。"""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        # 环境变量与 .env 处于 RAG_ 命名空间，可能出现与字段无关的变量，
        # 故根层忽略未知键；分区内仍为严格模式，分区名的合法性由 load 校验。
        extra="ignore",
    )

    storage: StorageSettings = Field(default_factory=StorageSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def load(
        cls,
        source: Optional[Mapping[str, Any]] = None,
        *,
        use_env: bool = True,
        **overrides: Any,
    ) -> "Settings":
        """加载配置，优先级由高到低：代码硬编码 > 环境变量 > 字段默认值。"""
        # source 为整体配置字典，overrides 以分区为单位，二者都是代码硬编码，优先级最高。
        # 环境变量先作为基线，再由硬编码逐字段覆盖，因此同时存在时以硬编码为准。
        # use_env 为 False 时忽略环境变量，便于测试与结果复现。
        if source is not None and not isinstance(source, Mapping):
            # 配置文件读取已移除，传路径时给出明确提示。
            raise ConfigError("配置源需为分区的字典；已移除配置文件读取，请改用环境变量或代码硬编码")
        env_data = _filter_sections(load_env_overrides()) if use_env else {}
        hardcoded = dict(source) if source else {}
        # 硬编码中的分区名必须合法，避免拼写错误被静默忽略。
        _validate_sections(hardcoded, overrides)
        layered = _merge_sections(env_data, hardcoded)
        layered = _merge_sections(layered, overrides)
        if use_env:
            # 环境变量作为基线，未指定的字段由 pydantic-settings 从环境与 .env 补齐。
            return cls(**layered)
        # 关闭环境变量：先用各分区默认值补齐每个字段，再整体作为构造实参传入。
        # 显式实参在 pydantic-settings 中优先级最高，环境变量因此无从渗入；
        # 不依赖 model_validate 是否绕过环境源这一版本相关行为。
        return cls(**_with_defaults(layered))

def load_env_overrides(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """把 RAG_ 前缀的环境变量组装为分区字典，键形如 RAG_STORAGE__VECTOR_DIM。"""
    source = os.environ if environ is None else environ
    data: Dict[str, Any] = {}
    for key, raw in source.items():
        if not key.startswith(ENV_PREFIX):
            continue
        remainder = key[len(ENV_PREFIX):]
        parts = [part.lower() for part in remainder.split(ENV_NESTED_DELIMITER) if part]
        if parts:
            _assign_nested(data, parts, _parse_env_value(raw))
    return data

def _filter_sections(data: Dict[str, Any]) -> Dict[str, Any]:
    """仅保留已知分区，忽略环境中无关的 RAG_ 变量。"""
    return {name: value for name, value in data.items() if name in MODULE_SECTIONS}

def _validate_sections(*groups: Mapping[str, Any]) -> None:
    """校验代码硬编码中的分区名合法。"""
    for group in groups:
        for name in group:
            if name not in MODULE_SECTIONS:
                raise ConfigError(
                    "未知的配置分区：" + str(name) + "；可用分区：" + ", ".join(MODULE_SECTIONS)
                )

def _merge_sections(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """按分区合并：同名分区逐字段覆盖，其余字段保留。"""
    merged: Dict[str, Any] = dict(base)
    for name, value in overrides.items():
        current = merged.get(name)
        if isinstance(current, dict) and isinstance(value, Mapping):
            section = dict(current)
            section.update(value)
            merged[name] = section
        else:
            merged[name] = value
    return merged

def _with_defaults(layered: Mapping[str, Any]) -> Dict[str, Any]:
    """用各分区默认值补齐字段，使构造实参覆盖全部字段，从而屏蔽环境变量。"""
    filled: Dict[str, Any] = {}
    for name, model in SECTION_MODEL_CLASSES.items():
        # 分区模型是纯 pydantic 模型，不读取环境；by_alias 保证键名与配置一致。
        section: Dict[str, Any] = dict(model().model_dump(by_alias=True))
        override = layered.get(name)
        if isinstance(override, Mapping):
            section.update(override)
        filled[name] = section
    return filled

def _assign_nested(data: Dict[str, Any], parts: List[str], value: Any) -> None:
    """按路径写入嵌套字典，如 ["storage", "vector_dim"]。"""
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value

def _parse_env_value(raw: Any) -> Any:
    """解析环境变量取值：JSON 对象或数组按 JSON 解析，其余交给 pydantic 转换。"""
    text = str(raw).strip()
    if text[:1] in ("{", "["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw
