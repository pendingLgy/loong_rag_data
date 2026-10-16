# rag-data 项目结构

> 版本 0.1.0，Python 3.8 及以上，src 布局，包根仅导出当前目录直接模块的 23 个符号。

本文档描述仓库的物理结构与模块职责，是阅读代码与二次开发的导航图。设计取舍与开发计划见 plan/DEVELOPMENT_PLAN.md，使用方式见 README.md，开发、构建与测试说明见 development.md。

> 存储层（关系库与向量库）已实现并接入门面开关；落库流程（把切块与向量写入向量库）仍由调用方自行编排。

---

## 1. 概览

rag-data 是历史文本导入与向量化管道：把文件解析为分块并编码为稠密向量，也支持字符串直接向量化；并提供关系库、向量库、大模型与日志的统一装配入口。整体分层，层间只依赖下层，方向单一。

| 层 | 包 | 职责 |
| :--- | :--- | :--- |
| 配置层 | config.py | 分区配置模型与三类来源加载（JSON 对象、环境变量、config.yml），各分区带 enable 开关 |
| 契约层 | exceptions.py | 领域异常 |
| 通用设施 | registry.py、logging | 通用注册表、日志适配层 |
| 领域服务 | parsers、embedding、llm | 解析切块、向量化、大模型调用 |
| 存储层 | storage/db、storage/vector_store | 关系库（历史与消息）与向量库（稠密与关键词检索） |
| 门面层 | facade.py | 一站式装配 RagData 与私有装配方法 |
| 公共入口 | __init__.py | 只导出当前目录（直接模块）的公共符号，不含子包 |

---

## 2. 目录结构

```text
rag-data/
├─ pyproject.toml          构建与依赖声明（hatchling + hatch 环境）
├─ README.md               使用说明：特性、快速开始、公共 API、配置、各子包、门面
├─ development.md          开发指南：环境与依赖管理、命令分类、测试与构建
├─ rag_artifact.md         架构设计源文档
├─ .vcl/struct.md          本文件：项目结构说明
├─ plan/
│  └─ DEVELOPMENT_PLAN.md   设计与开发计划
├─ src/
│  └─ rag_data/
│     ├─ __about__.py      版本号（hatch 版本来源）
│     ├─ __init__.py       包根，仅导出当前目录直接模块的 23 个公共符号
│     ├─ py.typed          类型标记（PEP 561）
│     ├─ config.py         分区配置模型与三类来源加载
│     ├─ exceptions.py     领域异常层次
│     ├─ registry.py       通用注册表（Registry、Ref）
│     ├─ facade.py         流程门面 RagData 与私有装配方法 _build_*
│     ├─ parsers/
│     │  ├─ __init__.py          汇总导出与默认处理器的绑定方法
│     │  ├─ base.py              DocumentParser 基类与 spaCy 句柄加载
│     │  ├─ registry_parser.py   纯注册表：register_parser、resolve_parser_class
│     │  ├─ parser_handler.py    ParserHandler：预实例化并按扩展名派发
│     │  ├─ markdown.py          MarkdownParser
│     │  ├─ txt.py               TxtParser
│     │  ├─ epub.py              EpubParser
│     │  ├─ pdf.py               PdfParser
│     │  └─ word.py              WordParser
│     ├─ embedding/
│     │  ├─ __init__.py          汇总导出 embedder 与注册函数
│     │  ├─ base.py              BaseEmbedder 抽象：声明 backend 即自动注册
│     │  ├─ registry_embedder.py 纯注册表：register_embedder、resolve_embedder
│     │  ├─ openai_embedder.py   OpenAIEmbedder
│     │  ├─ qwen_embedder.py     QwenEmbedder
│     │  └─ embedder_handler.py  EmbedderHandler：按配置实例化并校验
│     ├─ llm/
│     │  ├─ __init__.py          汇总导出 provider 与注册函数
│     │  ├─ base.py              BaseLLMProvider 抽象
│     │  ├─ registry_llm.py      纯注册表：register_llm、resolve_llm_provider
│     │  ├─ openai_llm.py        OpenAILLMProvider
│     │  ├─ qwen_llm.py          QwenLLMProvider
│     │  ├─ deepseek_llm.py      DeepSeekLLMProvider
│     │  └─ llm_handler.py       LLMHandler：按配置实例化并调用
│     ├─ storage/
│     │  ├─ db/
│     │  │  ├─ __init__.py          汇总导出接口、装配函数与数据对象
│     │  │  ├─ base.py              BaseDB 抽象接口与自动登记
│     │  │  ├─ registry_db.py       纯注册表（BUILTIN_DBS）
│     │  │  ├─ db_engine.py         SQLDatabase 与 build_database
│     │  │  ├─ schema.py            表结构、建表 SQL、HistoryRecord 与 Message
│     │  │  ├─ snowflake.py         雪花 ID 生成器
│     │  │  ├─ sqlite_db.py         SQLiteDB
│     │  │  ├─ pg_db.py             PGDB
│     │  │  └─ mysql_db.py          MySQLDB
│     │  └─ vector_store/
│     │     ├─ __init__.py          汇总导出接口、引擎与后端
│     │     ├─ base.py              VectorStoreBase、VectorStoreResult、MetricType
│     │     ├─ registry_vector.py   纯注册表（BUILTIN_STORES）
│     │     ├─ vector_engine.py     build_vector_store、VectorEngine、build_vector_engine
│     │     ├─ milvus_store.py      MilvusVectorStore（单一 BM25 schema）
│     │     └─ pg_store.py          PGVectorStore（建表与索引）
│     └─ logging/
│        ├─ __init__.py        汇总导出适配器与工厂
│        ├─ base.py            LoggerAdapter 抽象
│        ├─ factory.py         LoggerFactory：全局装配与 get_logger
│        ├─ stdlib_adapter.py  标准库适配与 StdlibTimezoneFormatter
│        ├─ loguru_adapter.py  loguru 适配（惰性导入）
│        └─ structlog_adapter.py structlog 适配（惰性导入）
└─ tests/
   ├─ conftest.py           路径注入、公共夹器与 --integration 选项
   ├─ test_config.py        配置三类来源与各分区
   ├─ test_facade.py        装配方法、RagData 构造与门控
   ├─ test_parsers_*.py     各格式契约、解析与切块、派发与注册表
   ├─ test_logging.py       工厂装配、时区格式化与调用点
   ├─ test_embedding.py     向量化注册表、内置 embedder 与 EmbedderHandler
   ├─ test_llm.py           大模型注册表、内置 provider 与 LLMHandler
   ├─ test_db.py            关系库注册表、表结构与 SQLite 端到端、MySQL 与 PG 集成
   ├─ test_vector_store.py  向量库注册表、DDL、引擎，以及真实服务集成
   └─ test_integration_qwen_epub_milvus.py  真实 EPUB 到 Milvus 端到端
```

---

## 3. 分层与依赖方向

依赖自上而下，无环；较低层不反向导入较高层。

```text
__init__（包根，仅导出直接模块符号）
     ▲
facade（流程门面 RagData）
     ├── config           分区配置与三类来源加载
     ├── logging          日志适配层
     ├── parsers          文档解析与切块（ParserHandler）
     ├── embedding        向量化（EmbedderHandler）
     ├── llm              大模型（LLMHandler）
     ├── storage/db        关系库（build_database）
     └── storage/vector_store  向量库（build_vector_engine）
```

关键约定：

1. 可选依赖（spacy、loguru、structlog、openai、PyYAML、pymilvus、psycopg、PyMySQL、SQLAlchemy）一律惰性使用，未安装时包仍可正常导入
2. 实现选择走注册表：声明 backend 即自动登记，新增实现不改动装配代码
3. 注册表只登记实现类，实例化在各自的 handler 或 engine 中进行
4. 包根只导出当前目录直接模块的公共符号，不跨模块、不列子包
5. config 的 YAML 来源由 _import_yaml 惰性导入 PyYAML，缺失时抛 OptionalDependencyError

---

## 4. 基础模块

### 4.1 __about__.py

仅保存 __version__，由 pyproject.toml 的 [tool.hatch.version] 读取。

### 4.2 __init__.py（包根）

只导出当前目录直接模块的公共符号，共 23 个，不含子包，也不做跨模块重导出：

```python
from rag_data.config import Settings, ParsingSettings, LLMSettings
from rag_data.facade import RagData
from rag_data.exceptions import RagDataError
from rag_data.registry import Registry, Ref
```

未纳入导出的符号（SECTION_DB、SECTION_VECTOR、DBSettings、VectorSettings、StorageError）按模块路径取用，分组见 7.2 节。

### 4.3 config.py

职责：以嵌套模型承载分区配置，实例化即校验；提供三类来源的加载方法与一组私有工具函数。每个分区模型都带 enable 开关（logging 默认开启，其余默认关闭）。

```python
class ParsingSettings(BaseModel): ...      # enable、spacy_model、max_chars、safe_max_chars
class EmbeddingSettings(BaseModel): ...    # enable、provider、model、batch_size、dim、api_key、base_url、timeout
class LoggingSettings(BaseModel): ...      # enable、backend、level、format、timezone
class LLMSettings(BaseModel): ...          # enable、provider、model、temperature、max_tokens、api_key、base_url、timeout
class SQLiteSettings / MySQLSettings / PGSettings
class MilvusSettings / PGVectorSettings
class VectorSettings(BaseModel): ...       # enable、backend、collection、vector_size、top_k、metric_type、milvus、pg
class DBSettings(BaseModel): ...           # enable、backend、sqlite、mysql、pg
class Settings(BaseSettings): ...          # parsing、embedding、logging、llm、db、vector
```

常量：SECTION_PARSING、SECTION_EMBEDDING、SECTION_LOGGING、SECTION_LLM、SECTION_DB、SECTION_VECTOR、MODULE_SECTIONS、SECTION_MODEL_CLASSES、ENV_PREFIX、ENV_NESTED_DELIMITER。

Settings 的类方法：

| 方法 | 来源 |
| :--- | :--- |
| load_mapping(source=None, **overrides) | JSON 对象（分区字典），不读环境变量 |
| load_env(environ=None, **overrides) | 环境变量（RAG_ 前缀、双下划线嵌套） |
| load_yaml(path, **overrides) | config.yml 文件，惰性导入 PyYAML |

模块级函数：

| 函数 | 作用 |
| :--- | :--- |
| _load_env_overrides(environ=None) | 把 RAG_ 前缀环境变量组装为分区字典 |
| _filter_sections(data) | 只保留已知分区，忽略无关的 RAG_ 变量 |
| _validate_sections(*groups) | 校验代码硬编码中的分区名合法 |
| _merge_sections(base, overrides) | 按分区逐字段合并，实现覆盖 |
| _with_defaults(layered) | 用分区默认值补齐字段，屏蔽环境变量 |
| _assign_nested(data, parts, value) | 按路径写入嵌套字典 |
| _parse_env_value(raw) | JSON 对象与数组按 JSON 解析 |
| _import_yaml() | 惰性导入 PyYAML，缺失时抛 OptionalDependencyError |

优先与合并：overrides 高于来源数据，来源数据高于分区默认值。

### 4.4 exceptions.py

单一根异常，便于上层统一捕获。

```text
RagDataError
├─ ConfigError            配置缺失或非法
├─ DataError              数据校验失败
├─ OptionalDependencyError 可选依赖未安装
│  └─ ParserDependencyError 文档解析依赖未安装
├─ ParseError             文档损坏或格式不支持
├─ EmbeddingError         向量化失败
├─ LLMError               大模型调用失败
└─ StorageError           存储读写失败
```

### 4.5 registry.py（通用注册表）

各子包注册表共用的 dict 子类，支持惰性导入与统一解析；Ref 为注册值类型别名（点分路径或类对象）。

```python
class Registry(dict):
    def __init__(self, kind: str, builtins: Mapping[str, Ref]) -> None: ...
    def register(self, name, ref) -> None: ...   # 登记类对象或点分路径
    def decorator(self, name): ...               # 装饰器形式登记
    def names(self) -> List[str]: ...            # 已登记名，字典序
    def resolve(self, name) -> Type[Any]: ...    # 路径登记则此时惰性导入
```

登记类对象而非路径，可让函数内定义的本地类也能注册；注册表不再提供实例化方法。

### 4.6 logging/

把三种日志库收敛到同一接口，调用方只依赖 LoggerAdapter。

| 文件 | 内容 |
| :--- | :--- |
| base.py | LoggerAdapter 抽象：bind 与五个级别方法 |
| factory.py | LoggerFactory：setup 全局装配与 get_logger 取用 |
| stdlib_adapter.py | StdlibLogAdapter 与 StdlibTimezoneFormatter（按 tz 格式化时间） |
| loguru_adapter.py | LoguruAdapter 与惰性导入 |
| structlog_adapter.py | StructlogAdapter 与惰性导入 |

后端名称为 stdliblog、structlog、loguru，缺省 stdliblog；格式 fmt 优先级：参数高于 pyproject 的 log_cli_format 高于内置默认（均含完整路径与行号）。

---

## 5. 领域服务层

### 5.1 parsers（解析与切块）

```text
文件 → parse_document → 文本 → chunk_document → 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| registry_parser.py | BUILTIN_PARSERS、register_parser、resolve_parser_class、parser_classes | 纯注册表，只登记不实例化 |
| parser_handler.py | ParserHandler、PARSER_CLASSES | 构造时预实例化全部解析器；按扩展名派发 |
| base.py | DocumentParser、load_nlp、clear_cache | 抽象基类：SUFFIXES、backend 与 parse、chunk 契约；句柄加载与缓存 |
| txt.py | TxtParser | .txt，整篇递归切块 |
| markdown.py | MarkdownParser | .md、.markdown，按行首井号分章后递归切块 |
| epub.py | EpubParser | .epub，按 spine 提取章节（排除目录页）后按井号分章递归切块 |
| pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取、过滤页眉页脚后整篇递归切块 |
| word.py | WordParser | .docx，python-docx 提取段落，整篇递归切块 |

ParserHandler：构造时可替换解析器类集合与默认长度；索引在构造时建立并做后缀唯一性断言。

| 方法 | 作用 |
| :--- | :--- |
| resolve_parser(path, max_chars=None, safe_max_chars=None, logger=None) | 取该格式实例；长度与缓存一致时复用，不一致则临时新建 |
| parse_document(path, logger=None) | 解析单个文档并返回纯文本 |
| chunk_document(path, text, nlp=None, logger=None, max_chars=None, safe_max_chars=None) | 按扩展名派发到该格式的 chunk |
| get_instance(path) | 取缓存实例，不新建 |
| parser_classes、instances（属性） | 解析器类集合、已预实例化的实例副本 |

切块由各解析器自持：基类提供递归切分与四分之一重叠前缀；单块上限 max_chars 与安全上限 safe_max_chars 为构造参数，由 parsing 分区注入（默认 500 与 2000）。句柄模型默认 zh_core_web_md，加载时关闭 tagger、ner 与 attribute_ruler。

子包 __init__ 汇总导出，并把默认处理器的绑定方法暴露为 resolve_parser、parse_document、chunk_document，便于直接调用。

### 5.2 embedding（向量化）

实现由配置 embedding.provider 选择，内置 openai 与 qwen。

```text
EmbedderHandler → resolve_embedder → Registry → OpenAIEmbedder
                                     └ QwenEmbedder
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseEmbedder 抽象：声明 backend 即自动登记 |
| registry_embedder.py | 纯注册表（BUILTIN_EMBEDDERS）与 register_embedder、resolve_embedder 等函数 |
| openai_embedder.py | OpenAIEmbedder，使用官方 openai SDK |
| qwen_embedder.py | QwenEmbedder，继承 OpenAI（DashScope 兼容模式） |
| embedder_handler.py | EmbedderHandler：按配置解析并实例化 embedder，统一批次与维度校验 |

抽象类可声明的类属性：backend、default_model、default_dim、default_base_url、api_key_env、max_batch_size。

| embedder | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY | 10 |

EmbedderHandler(settings, logger, model=None)：注入的 model 优先且绕过配置；provider 首次访问时实例化并缓存。批次策略：取 embedding.batch_size 与 embedder 的 max_batch_size 的较小者；编码结果按 embedding.dim 校验，不一致抛 EmbeddingError。

### 5.3 llm（大模型调用）

实现由配置 llm.provider 选择，内置 openai、qwen、deepseek。

```text
LLMHandler → resolve_llm_provider → Registry → OpenAILLMProvider
                                          ├ QwenLLMProvider
                                          └ DeepSeekLLMProvider
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseLLMProvider 抽象：声明 backend 即自动登记 |
| registry_llm.py | 纯注册表（BUILTIN_PROVIDERS）与 register_llm、resolve_llm_provider 等函数 |
| openai_llm.py | OpenAILLMProvider，使用官方 openai SDK |
| qwen_llm.py | QwenLLMProvider，继承 OpenAI（DashScope 兼容模式） |
| deepseek_llm.py | DeepSeekLLMProvider，继承 OpenAI |
| llm_handler.py | LLMHandler：按配置解析并实例化 provider，统一调用入口 |

| provider | 默认模型 | Key 环境变量 |
| :--- | :--- | :--- |
| openai | gpt-4o-mini | OPENAI_API_KEY |
| qwen | qwen-plus | DASHSCOPE_API_KEY |
| deepseek | deepseek-chat | DEEPSEEK_API_KEY |

LLMHandler(settings, logger, model=None) 提供 provider（属性）、chat(messages, **overrides) 与 complete(prompt, system, **overrides)；注入的 model 优先且绕过配置。

---

## 6. 存储层

### 6.1 storage/db（关系库）

```text
build_database → SQLiteDB / MySQLDB / PGDB → SQLDatabase（SQLAlchemy Core）
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseDB 抽象接口；声明 backend 即自动登记；__init__(logger=None) |
| registry_db.py | 纯注册表（BUILTIN_DBS：sqlite、mysql、pg）与注册、解析函数 |
| db_engine.py | SQLDatabase 通用实现与 build_database 装配函数 |
| schema.py | METADATA、HISTORY_TABLE、MESSAGES_TABLE、CREATE_SQL、create_table_sql、HistoryRecord、Message |
| snowflake.py | SnowflakeGenerator、next_id 与位宽常量 |
| sqlite_db.py | SQLiteDB：db_path 必填，数据一律落盘 |
| pg_db.py | PGDB：psycopg3 优先、psycopg2 回退；缺库时实例化抛 OptionalDependencyError |
| mysql_db.py | MySQLDB：PyMySQL 优先、mysql-connector-python 回退 |

表结构：主键均为 BIGINT（雪花 ID，天然按时间趋势递增）。

| 表 | 关键列 |
| :--- | :--- |
| history | id（主键）、memory_id（BIGINT，可空，ix_history_memory_id）、old_memory、new_memory、event、created_at、updated_at、is_deleted、actor_id、role |
| messages | id（主键）、session_scope、role、content、name、created_at，ix_messages_scope_created 为 session_scope 与 created_at 的复合索引 |

SQLDatabase 的公开方法：add_history、batch_add_history、get_history、save_messages、get_last_messages、reset、close；私有辅助：_record_row、_to_record、_message_row、_to_message、_evict_stmt。

建表与生命周期：__init__ 打开引擎后检查表是否存在，缺失的才创建（已存在不动）；save_messages 按 session_scope 归组，仅保留每会话最近 keep_last 条；支持 with 上下文管理，close 释放引擎。

build_database(settings, logger=None) 按 settings.backend 惰性导入并构造方言实现：sqlite 要求 settings.sqlite.path 非空，否则抛 ConfigError。

数据对象：HistoryRecord 与 Message 为 pydantic 模型（extra 为 ignore），支持传入字典构造。

### 6.2 storage/vector_store（向量库）

```text
build_vector_engine → VectorEngine → VectorStoreBase
                              ├ MilvusVectorStore（单一 BM25 schema）
                              └ PGVectorStore（pgvector 表与索引）
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | VectorStoreBase 抽象、VectorStoreResult、MetricType（COSINE、L2、IP） |
| registry_vector.py | 纯注册表（BUILTIN_STORES：milvus、pg）与注册、解析函数 |
| vector_engine.py | build_vector_store、VectorEngine、build_vector_engine、_backend_kwargs |
| milvus_store.py | MilvusVectorStore：集合与索引 |
| pg_store.py | PGVectorStore：建表与索引，DDL 辅助函数 |

VectorStoreBase 抽象方法：create_col、insert、search、delete、update、get、list_cols、delete_col、col_info、list、reset；keyword_search 默认返回 None。约定 search 返回的 score 越大越相似（COSINE 用 1 减距离、L2 取 1 加距离的倒数、IP 用内积）；list 与 get 等无排序场景 score 为 None。

| 后端 | 建集合或建表 | 索引 |
| :--- | :--- | :--- |
| milvus | 单一 schema：id、vectors、metadata、text（开启分析器）、sparse（由 BM25 函数按 text 生成） | vector_index（AUTOINDEX）、sparse_index（SPARSE_INVERTED_INDEX，BM25） |
| pg | CREATE TABLE IF NOT EXISTS：id、vector vector(维度)、payload JSONB；另有 table_ddl 等 DDL 辅助函数 | HNSW（默认）或 DiskANN（需 vectorscale，缺失回退 HNSW，维度大于 2000 时跳过）；全文 GIN 索引与 keyword_search 表达式一致 |

VectorEngine：属性 store、backend、collection_name、vector_size；写入 insert、update、delete；检索 search、keyword_search、get、list；维护 list_cols、col_info、reset、delete_col、close，支持 with 上下文管理。写入与检索前校验维度与批量长度，不一致抛 StorageError；默认检索条数取 default_top_k（来自 vector.top_k）。

build_vector_engine(settings, logger=None) 按 settings.backend 组装参数：集合名、维度、度量类型固定注入；连接参数按后端从 milvus 或 pg 分区取，留空字段不下传（交由实现自身默认值）。

---

## 7. 门面层与公共 API

### 7.1 facade.py

模块级私有装配方法，各对应一个分区，且都受该分区的 enable 门控（为 False 时返回 None）：

| 方法 | 流程 |
| :--- | :--- |
| _build_logger(logging_settings) | 装配日志适配层 |
| _build_nlp(parsing_settings, logger) | 加载 spaCy，不可用时返回 None |
| _build_embedder(embedding_settings, logger, model=None) | 装配向量化组件 |
| _build_db(db_settings, logger) | 按 db 分区装配关系库 |
| _build_vector(vector_settings, logger) | 按 vector 分区装配向量引擎 |

RagData 是一次装配、逐流程调用的门面：

```python
RagData(settings)
```

| 成员 | 说明 |
| :--- | :--- |
| settings、parser、logger、nlp、embedder、db、vector | 构造后挂载的实例变量；未启用的分区为 None |
| parser_handler（属性） | 解析派发入口，返回 ParserHandler 实例 |
| parse_document(path) | 委托 parser 解析单文件为纯文本 |
| chunk_document(path, text) | 委托 parser 切块，长度取自 parsing 分区 |

解析器不做门控：parser 始终装配，保证 parse_document 与 chunk_document 在任何配置下可用。

facade 的 __all__ 仅含 RagData。

### 7.2 包根导出（23 个符号）

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、ParsingSettings、EmbeddingSettings、LoggingSettings、LLMSettings、ENV_PREFIX、ENV_NESTED_DELIMITER、SECTION_PARSING、SECTION_EMBEDDING、SECTION_LOGGING、SECTION_LLM |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError、LLMError |
| 通用设施 | Registry、Ref |
| 流程门面 | RagData |

未纳入包根导出的实现按需从子包引入，例如 parsers 的 ParserHandler、embedding 的 EmbedderHandler、llm 的 LLMHandler、storage 的 build_database 与 build_vector_store、storage 的 StorageError。

---

## 8. 数据流

### 8.1 文件向量化

```text
1. RagData(settings)                装配 settings、logger、nlp、embedder、db、vector（按 enable 门控）
2. app.parse_document(path)         解析文件为纯文本
3. app.chunk_document(path, text)   按格式切块（ParserHandler 派发）
4. app.embedder.encode(chunks)      分块编码为稠密向量
5. engine = app.vector              写入与检索（调用方自行编排）
```

### 8.2 字符串向量化

```text
app.embedder.encode(texts)          批量字符串到向量列表（与输入同序）
    ├─ 分批：embedding.batch_size 与 embedder 上限取较小者
    └─ 校验：逐条按 embedding.dim 校验，不一致抛 EmbeddingError
```

### 8.3 存储读写

```text
app.db.add_history(HistoryRecord(...))        写入历史记录（id 缺省时生成雪花 ID）
app.db.get_history(memory_id)                按 memory_id 取历史，时间升序
app.db.save_messages([Message(...)])          写入消息，仅保留每会话最近 N 条
app.vector.insert(vectors, payloads, ids)    写入向量与元数据
app.vector.search(query, vector, top_k)      稠密检索；keyword_search 走 BM25 或全文检索
```

---

## 9. 配置项

分区与字段（括号内为默认值）：

| 分区 | 字段 |
| :--- | :--- |
| parsing | enable(false)、spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | enable(false)、provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |
| logging | enable(true)、backend(stdliblog)、level(INFO)、format(空)、timezone(空) |
| llm | enable(false)、provider(openai)、model(空)、temperature(0.7)、max_tokens(0)、api_key(空)、base_url(空)、timeout(60.0) |
| db | enable(false)、backend(sqlite)、sqlite.path(空)、mysql 与 pg 连接参数 |
| vector | enable(false)、backend(milvus)、collection(rag_data)、vector_size(1024)、top_k(5)、metric_type(COSINE)、milvus 与 pg 连接参数 |

三类来源（实现见 config.py）：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| Settings.load_mapping | JSON 对象（分区字典） | 否 |
| Settings.load_env | 环境变量 | 是（可显式传 environ） |
| Settings.load_yaml | config.yml 文件 | 否（惰性导入 PyYAML） |

环境变量命名：RAG_ 前缀，分区与字段大写，双下划线分隔，如 RAG_EMBEDDING__DIM 与 RAG_DB__ENABLE；取值按字段类型转换，对象与数组按 JSON 书写，分区名非法抛 ConfigError。

enable 语义：分区开关。关闭时门面不初始化对应组件（实例为 None），可避免加载 spaCy、请求外部模型或建立数据库连接。

---

## 10. 扩展点

各领域的实现选择都走注册表：继承抽象基类并声明 backend 即被收录，新增实现无需改动装配代码。

| 扩展点 | 抽象基类 | 声明方式 | 配置键 | 内置 |
| :--- | :--- | :--- | :--- | :--- |
| 文档解析 | DocumentParser | backend 与 SUFFIXES | 无（按扩展名派发） | txt、markdown、epub、pdf、word |
| 嵌入模型 | BaseEmbedder | backend 类属性 | embedding.provider | openai、qwen |
| 大模型 | BaseLLMProvider | backend 类属性 | llm.provider | openai、qwen、deepseek |
| 关系库 | BaseDB | backend 类属性 | db.backend | sqlite、mysql、pg |
| 向量库 | VectorStoreBase | backend 类属性 | vector.backend | milvus、pg |

新增 embedder 的最小骨架：

```python
from rag_data.embedding import BaseEmbedder

class LocalEmbedder(BaseEmbedder):
    backend = 'local'          # 声明名字，类定义时自动注册
    default_model = 'bge-large-zh'
    default_dim = 1024
    default_base_url = 'http://localhost:8000/v1'
    api_key_env = 'LOCAL_API_KEY'

    def encode(self, texts):
        # 返回 List[List[float]]，顺序与输入一致
        ...
```

自定义实现也可显式登记：register_parser、register_embedder、register_llm、register_db、register_vector_store，或使用对应的装饰器形式（名称以 _provider 结尾）。

---

## 11. 测试

```text
tests（共 383 项：单元 240、集成 143）
├─ conftest.py                     路径注入、settings 与 logger 夹器、--integration 选项
├─ test_config.py           38 项  三类来源与各分区（含 enable 开关）
├─ test_facade.py           33 项  装配方法、门控、RagData 构造与流程方法
├─ test_parsers_base.py     16 项  分句、重叠前缀与递归切块
├─ test_parsers_dispatch.py 20 项  派发、注册表与预实例化复用
├─ test_parsers_txt.py      12 项  TxtParser 契约、解析与切块
├─ test_parsers_markdown.py 12 项  MarkdownParser
├─ test_parsers_epub.py     10 项  EpubParser
├─ test_parsers_pdf.py      10 项  PdfParser
├─ test_parsers_word.py     12 项  WordParser
├─ test_logging.py          17 项  工厂装配、时区格式化与调用点
├─ test_embedding.py        21 项  注册表、内置 embedder 与 EmbedderHandler
├─ test_llm.py              39 项  注册表、内置 provider 与 LLMHandler
├─ test_db.py               64 项  注册表、表结构、SQLite 端到端、MySQL 与 PG 集成
├─ test_vector_store.py     73 项  注册表、DDL、引擎、真实 Milvus 与 PG 集成
└─ test_integration_qwen_epub_milvus.py  6 项  真实 EPUB 到 Milvus 端到端
```

集成用例分布在 test_db.py、test_vector_store.py、test_integration_qwen_epub_milvus.py：这三个文件在模块级声明 pytestmark 为 integration，并检查 --integration 参数，未提供时以 allow_module_level 整体跳过并打印 [SKIP INFO]；默认 addopts 另加 -m not integration 反选。

跳过信息由各测试模块内的局部辅助函数输出，包含依赖名、目标地址与原始异常，配合 -rs 可查看跳过清单；与本机示例文件相关的环境变量：RAG_TEST_TXT、RAG_TEST_MD、RAG_TEST_PDF、RAG_TEST_DOCX、RAG_TEST_EPUB。

---

## 12. 依赖

核心依赖仅两项：pydantic 与 pydantic-settings。其余按用途分组为可选依赖（extras），声明见 pyproject.toml。

| 组 | 包 |
| :--- | :--- |
| 核心 | pydantic、pydantic-settings |
| parsers | unstructured、spacy、zh_core_web_sm、zh_core_web_md |
| epub | EbookLib、beautifulsoup4 |
| pdf | pdfplumber |
| word | python-docx |
| embedding | sentence-transformers（预留，尚无对应后端） |
| openai | openai SDK（内置 openai 与 qwen provider 依赖） |
| sqlalchemy | SQLAlchemy（关系库 SQLAlchemy Core 所需） |
| pg | psycopg（关系库与向量库的 PG 后端） |
| mysql | PyMySQL（关系库 MySQL 后端） |
| milvus | pymilvus（向量库 Milvus 后端，建议 2.5 及以上以支持 BM25） |
| loguru | loguru |
| structlog | structlog |
| dev | pytest、pytest-cov、mypy、ruff |

环境装配：pyproject.toml 定义 default 与 rag_data_dev 两套 hatch 环境，按 features 自动安装上述 extras；rag_data_dev 额外包含 sqlalchemy、pg、mysql、milvus、word、pdf、openai。

注意：PyYAML 用于 Settings.load_yaml，当前未在 pyproject.toml 声明，生产使用前建议补充依赖或新增 yaml extra。

---

## 13. 待补全

| 位置 | 事项 |
| :--- | :--- |
| parsers | 版面级解析细化，如 PDF 双栏、Word 表格 |
| embedding | 兼容本地模型的 embedder（如 sentence-transformers 后端） |
| 数据保存 | 向量落库流程与批量导入编排，需调用方自行组合 parser、embedder 与 vector |
| facade | 是否把 llm、db、vector 纳入更多流程方法（如直接暴露检索与写入入口） |
| config | pyproject.toml 未声明 PyYAML；llm 与 db、vector 分区已接入配置但与门面的联动仍以显式调用为主 |
| 子包导出 | 各子包 __init__.py 的重导出策略尚未完全统一 |
| 遗留目录 | src/rag_data 下的 memory、util 为空目录（仅 __pycache__），待清理 |

