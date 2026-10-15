# 配置模型：按模块分区，默认读取环境变量，也支持在代码中硬编码。
#
# 优先级由高到低：
#   1. 代码硬编码：Settings.load_mapping(embedding={"provider": "qwen"})
#   2. 环境变量：RAG_<分区>__<字段>，如 RAG_EMBEDDING__DIM=1024
#   3. 字段默认值
#
# 环境变量命名规则：RAG_ 前缀，分区名与字段名大写，双下划线分隔，例如：
#   RAG_EMBEDDING__PROVIDER=qwen
#   RAG_EMBEDDING__DIM=1024
#   RAG_PARSING__SPACY_MODEL=zh_core_web_sm
#   RAG_PARSING__MAX_CHARS=1000
#   RAG_PARSING__SAFE_MAX_CHARS=2000
#   RAG_LOGGING__LEVEL=DEBUG
#   RAG_LOGGING__FORMAT=%(asctime)s [%(levelname)s] [%(pathname)s:%(lineno)d %(funcName)s()] %(message)s
#   RAG_LOGGING__TIMEZONE=Asia/Shanghai

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_data.exceptions import ConfigError, OptionalDependencyError

SECTION_PARSING = "parsing"
SECTION_EMBEDDING = "embedding"
SECTION_LOGGING = "logging"

MODULE_SECTIONS: List[str] = [
    SECTION_PARSING,
    SECTION_EMBEDDING,
    SECTION_LOGGING,
]

# 环境变量前缀与分区分隔符，与 SettingsConfigDict 的 env_prefix、env_nested_delimiter 一致。
ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"


class ParsingSettings(BaseModel):
    """文档解析与切块参数：spaCy 句柄模型与切块长度上限。"""

    model_config = ConfigDict(extra="forbid")

    spacy_model: str = Field(default="zh_core_web_sm")
    # 单块字符上限，逐块累加句子至该长度即收块，注入给各解析器。
    max_chars: int = Field(default=500, gt=0)
    # 单块安全上限，供解析器在极端长句时兜底，与 max_chars 一同注入。
    safe_max_chars: int = Field(default=2000, gt=0)

class EmbeddingSettings(BaseModel):
    """向量化参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # provider 名对应嵌入模型注册表中的键；继承 BaseEmbeddingProvider 即自动注册，
    # 内置 openai 与 qwen，可用 rag_data.available_embedding_providers() 查看。
    provider: str = Field(default="openai")
    # 模型名。留空时用 provider 的默认模型，见各 provider 的 default_model。
    model: str = Field(default="")
    batch_size: int = Field(default=128, gt=0)
    # 期望输出维度；向量化结果按此校验，默认与内置 qwen 模型一致。
    dim: int = Field(default=1024, gt=0)
    # API Key。留空时读取 provider 约定的环境变量，如 OPENAI_API_KEY、DASHSCOPE_API_KEY。
    api_key: str = Field(default="")
    # 接口地址。留空时用 provider 默认地址，可指向自建网关或代理。
    base_url: str = Field(default="")
    timeout: float = Field(default=60.0, gt=0)

class LoggingSettings(BaseModel):
    """日志适配层参数：后端、级别、格式与时区。"""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["stdliblog", "structlog", "loguru"] = "stdliblog"
    level: str = Field(default="INFO")
    # 留空时由工厂读取 pyproject 的 log_cli_format，再回退内置默认。
    format: str = Field(default="")
    # 时区：留空用系统本地时区，可填 UTC 或 Asia 开头的 IANA 名。
    timezone: str = Field(default="")

# 分区名到分区模型的映射。
# 关闭环境变量时用它补齐字段默认值，使构造实参覆盖全部字段。
SECTION_MODEL_CLASSES: Dict[str, Type[BaseModel]] = {
    SECTION_PARSING: ParsingSettings,
    SECTION_EMBEDDING: EmbeddingSettings,
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

    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def load_mapping(cls, source: Optional[Mapping[str, Any]] = None, **overrides: Any) -> "Settings":
        '''来源一：解析 JSON 对象（分区的字典），不读取环境变量。'''
        hardcoded = cls._read_mapping(source)
        _validate_sections(hardcoded, overrides)
        layered = _merge_sections(hardcoded, overrides)
        return cls(**_with_defaults(layered))

    @classmethod
    def load_env(cls, environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> "Settings":
        '''来源二：解析环境变量，RAG_ 前缀、双下划线分隔，键如 RAG_EMBEDDING__DIM。'''
        env_data = cls._read_env(environ)
        _validate_sections(overrides)
        layered = _merge_sections(env_data, overrides)
        return cls(**_with_defaults(layered))

    @classmethod
    def load_yaml(cls, path: str, **overrides: Any) -> "Settings":
        '''来源三：解析 config.yml 文件，内容需为分区的映射。'''
        file_data = cls._read_yaml(path)
        _validate_sections(file_data, overrides)
        layered = _merge_sections(file_data, overrides)
        return cls(**_with_defaults(layered))

    # ---------------- 三类来源的解析 ----------------

    @staticmethod
    def _read_mapping(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        '''解析 JSON 对象：确认为分区的映射后原样返回。'''
        if source is None:
            return {}
        if not isinstance(source, Mapping):
            raise ConfigError('''配置源需为分区的字典（JSON 对象）；如需读取文件请用 Settings.load_yaml''')
        return dict(source)

    @staticmethod
    def _read_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        '''解析环境变量：抽取 RAG_ 前缀并组装为分区字典。'''
        return _filter_sections(_load_env_overrides(environ))

    @staticmethod
    def _read_yaml(path: str) -> Dict[str, Any]:
        '''解析 config.yml：惰性导入 PyYAML，内容需为分区的映射。'''
        yaml = _import_yaml()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except OSError as exc:
            raise ConfigError('''无法读取配置文件：''' + str(path)) from exc
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise ConfigError('''配置文件需为分区的映射：''' + str(path))
        return dict(data)


def _load_env_overrides(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """把 RAG_ 前缀的环境变量组装为分区字典，键形如 RAG_EMBEDDING__DIM。"""
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
    """按路径写入嵌套字典，如 ["embedding", "dim"]。"""
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


def _import_yaml() -> Any:
    '''惰性导入 PyYAML；未安装时抛出可选依赖异常。'''
    try:
        import yaml
    except ImportError as exc:
        raise OptionalDependencyError('''解析 YAML 需要 PyYAML，请安装 pyyaml''') from exc
    return yaml
