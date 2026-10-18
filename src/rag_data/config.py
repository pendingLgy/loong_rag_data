# 配置模型：按模块分区，默认读取环境变量，也支持在代码中硬编码。
#
# 优先级由高到低：
#   1. 代码硬编码：RagSetting.load_mapping(embedding={"provider": "qwen"})
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
#
# 记忆分区独立于 RagSetting，前缀为 RAG_MEMORY_，例如：
#   RAG_MEMORY__ENABLE=true
#   RAG_MEMORY__DB__BACKEND=pg
#   RAG_MEMORY__DB__MYSQL__HOST=127.0.0.1
#   RAG_MEMORY__DB__MYSQL__PASSWORD=secret
#   RAG_MEMORY__DB__SQLITE__PATH=data\rag_data.db

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from rag_data.exceptions import ConfigError, OptionalDependencyError

SECTION_PARSING = "parsing"
SECTION_EMBEDDING = "embedding"
SECTION_LOGGING = "logging"

MODULE_SECTIONS: List[str] = [
    SECTION_PARSING,
    SECTION_EMBEDDING,
    SECTION_LOGGING,
]

# 环境变量前缀与嵌套分隔符：解析侧供 SettingsConfigDict 使用，记忆侧供手工解析使用。
ENV_PREFIX = "RAG_"
# 记忆分区独立装配，使用独立前缀，避免与 RagSetting 的分区键冲突。
MEMORY_ENV_PREFIX = ENV_PREFIX + "MEMORY_"
ENV_NESTED_DELIMITER = "__"


class ParsingSettings(BaseModel):
    """文档解析与切块参数：spaCy 句柄模型与切块长度上限。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    spacy_model: str = Field(default="zh_core_web_sm")
    # 单块字符上限，逐块累加句子至该长度即收块，注入给各解析器。
    max_chars: int = Field(default=500, gt=0)
    # 单块安全上限，供解析器在极端长句时兜底，与 max_chars 一同注入。
    safe_max_chars: int = Field(default=2000, gt=0)

class EmbeddingSettings(BaseModel):
    """向量化参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

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

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=True)

    backend: Literal["stdliblog", "structlog", "loguru"] = "stdliblog"
    level: str = Field(default="INFO")
    # 留空时由工厂读取 pyproject 的 log_cli_format，再回退内置默认。
    format: str = Field(default="")
    # 时区：留空用系统本地时区，可填 UTC 或 Asia 开头的 IANA 名。
    timezone: str = Field(default="")

class LLMSettings(BaseModel):
    """大模型参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # provider 名对应大模型注册表中的键；内置 openai、qwen、deepseek，qwen 与 deepseek 兼容 OpenAI。
    provider: str = Field(default="openai")
    # 模型名。留空时用 provider 的默认模型，见各 provider 的 default_model。
    model: str = Field(default="")
    # 采样温度，范围 0 到 2。
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # 单次生成的最大 token 数，0 表示不显式传入（由服务端决定）。
    max_tokens: int = Field(default=0, ge=0)
    # API Key。留空时读取 provider 约定的环境变量，如 OPENAI_API_KEY、DASHSCOPE_API_KEY、DEEPSEEK_API_KEY。
    api_key: str = Field(default="")
    # 接口地址。留空时用 provider 默认地址，可指向自建网关或代理。
    base_url: str = Field(default="")
    timeout: float = Field(default=60.0, gt=0)


class SQLiteSettings(BaseModel):
    """SQLite 连接参数：path 为库文件路径，必须由使用方显式配置。"""

    model_config = ConfigDict(extra="forbid")

    # 库文件路径；留空表示未配置，实例化时会报错。
    path: str = Field(default="")


class MySQLSettings(BaseModel):
    """MySQL 连接参数。"""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=3306, gt=0)
    user: str = Field(default="root")
    password: str = Field(default="")
    database: str = Field(default="rag_data")
    charset: str = Field(default="utf8mb4")


class PGSettings(BaseModel):
    """PostgreSQL 连接参数；connection_string 非空时优先于离散字段。"""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=5432, gt=0)
    user: str = Field(default="postgres")
    password: str = Field(default="")
    database: str = Field(default="rag_data")
    # 完整连接串；留空则由上面的离散字段拼装。
    connection_string: str = Field(default="")

class MilvusSettings(BaseModel):
    """Milvus 连接参数；留空的字段交由实现层使用自身默认值。"""

    model_config = ConfigDict(extra="forbid")

    # 服务地址；留空时用实现默认（本机 19530 端口）。
    uri: str = Field(default="")
    # 鉴权令牌；留空表示不鉴权。
    token: str = Field(default="")
    # 数据库名；留空表示使用 Milvus 默认库。
    db_name: str = Field(default="")


class PGVectorSettings(BaseModel):
    """PostgreSQL 向量库参数；connection_string 非空时优先于离散字段。"""

    model_config = ConfigDict(extra="forbid")

    connection_string: str = Field(default="")
    host: str = Field(default="localhost")
    port: int = Field(default=5432, gt=0)
    user: str = Field(default="postgres")
    password: str = Field(default="")
    database: str = Field(default="rag_data")
    # 是否为向量列建立 HNSW 索引。
    use_hnsw: bool = Field(default=True)
    # 是否尝试 DiskANN 索引（需 vectorscale 扩展；缺失时自动回退 HNSW）。
    use_diskann: bool = Field(default=False)


class VectorSettings(BaseModel):
    """向量库分区：按 milvus、pg 承载连接参数，并统一集合与检索默认值。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # 当前使用的向量后端名，对应 storage.vector_store 注册表中的键。
    backend: Literal["milvus", "pg"] = Field(default="milvus")
    # 集合名与向量维度。
    collection: str = Field(default="rag_data")
    vector_size: int = Field(default=1024, gt=0)
    # 检索默认返回条数。
    top_k: int = Field(default=5, gt=0)
    # 距离度量类型。
    metric_type: Literal["COSINE", "L2", "IP"] = Field(default="COSINE")

    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    pg: PGVectorSettings = Field(default_factory=PGVectorSettings)


class DBSettings(BaseModel):
    """数据库分区：按 sqlite、mysql、pg 分别承载连接参数。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # 当前使用的后端名，对应 storage.db 注册表中的键。
    backend: Literal["sqlite", "mysql", "pg"] = Field(default="sqlite")

    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    mysql: MySQLSettings = Field(default_factory=MySQLSettings)
    pg: PGSettings = Field(default_factory=PGSettings)


class _MemoryEnvSource(PydanticBaseSettingsSource):
    """读取 RAG_MEMORY_ 前缀环境变量，产出 MemorySettings 的字段字典。

    不复用 pydantic-settings 的原生前缀解析：前缀以单下划线结尾、嵌套又以双下划线
    分隔时，RAG_MEMORY__DB__BACKEND 会被切成 _DB 与 BACKEND，导致 db 等嵌套字段
    读不到值。这里改走与 load_env 相同的解析，两种下划线写法行为一致。
    """

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        """整份字段字典一次性产出，故不按单字段取值。"""
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        return _load_memory_env_overrides()


class MemorySettings(BaseSettings):
    """记忆分区：与 RagSetting 相互独立，自带 llm、db、vector 三类子配置。"""

    model_config = SettingsConfigDict(
        # 前缀以单下划线结尾、嵌套又以双下划线分隔，原生解析会把 RAG_MEMORY__DB__BACKEND
        # 切成 _DB 与 BACKEND；故关闭原生 env 解析与 .env 文件，改由
        # settings_customise_sources 挂载 _MemoryEnvSource，与 load_env 共用同一套解析。
        # 未知字段仍直接报错，避免拼写错误被静默忽略。
        extra="forbid",
    )

    # 分区开关；关闭时门面不装配 MemoryHandler。
    enable: bool = Field(default=False)

    # 三类依赖：大模型（事实抽取）、关系库（历史与消息）、向量库（记忆检索）。
    llm: LLMSettings = Field(default_factory=LLMSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)

    # ---------------- 三类来源 ----------------

    @classmethod
    def load_mapping(cls, source: Optional[Mapping[str, Any]] = None, **overrides: Any) -> "MemorySettings":
        """来源一：解析字典（JSON 对象），不读取环境变量。"""
        hardcoded = _as_mapping(source, cls.__name__)
        layered = _merge_sections(hardcoded, overrides)
        return cls(**_with_model_defaults(cls, layered))

    @classmethod
    def load_env(cls, environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> "MemorySettings":
        """来源二：解析 RAG_MEMORY_ 前缀的环境变量，键形如 RAG_MEMORY__DB__HOST。"""
        env_data = _load_memory_env_overrides(environ)
        layered = _merge_sections(env_data, overrides)
        return cls(**_with_model_defaults(cls, layered))

    @classmethod
    def load_yaml(cls, path: str, **overrides: Any) -> "MemorySettings":
        """来源三：解析 config.yml 文件，内容为 MemorySettings 字段的映射。"""
        file_data = _as_yaml_mapping(path)
        layered = _merge_sections(file_data, overrides)
        return cls(**_with_model_defaults(cls, layered))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type["MemorySettings"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """初始化实参优先，RAG_MEMORY_ 环境变量次之。

        不挂载原生 env 解析与 .env 源：前者受前缀与分隔符冲突影响，后者键规则
        与 load_env 不一致；需要文件来源时改用 load_yaml。
        """
        return (init_settings, _MemoryEnvSource(settings_cls))


# 分区名到分区模型的映射。
# 关闭环境变量时用它补齐字段默认值，使构造实参覆盖全部字段。
SECTION_MODEL_CLASSES: Dict[str, Type[BaseModel]] = {
    SECTION_PARSING: ParsingSettings,
    SECTION_EMBEDDING: EmbeddingSettings,
    SECTION_LOGGING: LoggingSettings,
}


class RagSetting(BaseSettings):
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
    def load_mapping(cls, source: Optional[Mapping[str, Any]] = None, **overrides: Any) -> "RagSetting":
        '''来源一：解析 JSON 对象（分区的字典），不读取环境变量。'''
        hardcoded = cls._read_mapping(source)
        _validate_sections(hardcoded, overrides)
        layered = _merge_sections(hardcoded, overrides)
        return cls(**_with_defaults(layered))

    @classmethod
    def load_env(cls, environ: Optional[Mapping[str, str]] = None, **overrides: Any) -> "RagSetting":
        '''来源二：解析环境变量，RAG_ 前缀、双下划线分隔，键如 RAG_EMBEDDING__DIM。'''
        env_data = cls._read_env(environ)
        _validate_sections(overrides)
        layered = _merge_sections(env_data, overrides)
        return cls(**_with_defaults(layered))

    @classmethod
    def load_yaml(cls, path: str, **overrides: Any) -> "RagSetting":
        '''来源三：解析 config.yml 文件，内容需为分区的映射。'''
        file_data = cls._read_yaml(path)
        _validate_sections(file_data, overrides)
        layered = _merge_sections(file_data, overrides)
        return cls(**_with_defaults(layered))

    # ---------------- 三类来源的解析 ----------------

    @staticmethod
    def _read_mapping(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """解析 JSON 对象：确认为字典映射后原样返回。"""
        return _as_mapping(source, "RagSetting")


    @staticmethod
    def _read_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        '''解析环境变量：抽取 RAG_ 前缀并组装为分区字典。'''
        return _filter_sections(_load_env_overrides(environ))

    @staticmethod
    def _read_yaml(path: str) -> Dict[str, Any]:
        """解析 config.yml：惰性导入 PyYAML，内容需为字典映射。"""
        return _as_yaml_mapping(path)


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

def _load_memory_env_overrides(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """把 RAG_MEMORY_ 前缀的环境变量组装为字段字典，键形如 RAG_MEMORY__DB__HOST。"""
    source = os.environ if environ is None else environ
    data: Dict[str, Any] = {}
    for key, raw in source.items():
        if not key.startswith(MEMORY_ENV_PREFIX):
            continue
        # 去掉前缀后再剥掉一个分隔下划线，兼容 RAG_MEMORY__X 与 RAG_MEMORY_X 两种写法。
        remainder = key[len(MEMORY_ENV_PREFIX):].lstrip("_")
        parts = [part.lower() for part in remainder.split(ENV_NESTED_DELIMITER) if part]
        if parts:
            _assign_nested(data, parts, _parse_env_value(raw))
    return data


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

def _as_mapping(source: Optional[Mapping[str, Any]], loader: str) -> Dict[str, Any]:
    """校验配置源为字典后原样复制；非字典时提示改用 load_yaml。"""
    if source is None:
        return {}
    if not isinstance(source, Mapping):
        raise ConfigError("配置源需为字典映射（JSON 对象）；如需读取文件请用 " + loader + ".load_yaml")
    return dict(source)

def _as_yaml_mapping(path: str) -> Dict[str, Any]:
    """读取 config.yml 并校验为字典映射；PyYAML 缺失时抛 OptionalDependencyError。"""
    yaml = _import_yaml()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError("无法读取配置文件：" + str(path)) from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError("配置文件需为字典映射：" + str(path))
    return dict(data)

def _with_model_defaults(model_cls: Type[BaseModel], layered: Mapping[str, Any]) -> Dict[str, Any]:
    """用字段默认值补齐全部顶层键，屏蔽环境变量，再叠加来源数据。

    MemorySettings 是 BaseSettings，直接实例化会读取环境变量；这里改为从
    model_fields 取默认值，保证 load_mapping 与 load_yaml 不读环境。
    """
    filled: Dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        factory = field.default_factory
        filled[name] = factory() if factory is not None else field.default
    filled.update(layered)
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
