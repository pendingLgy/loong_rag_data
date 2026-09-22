# rag-data 项目结构

> 版本 0.1.0 · Python >=3.8 · src 布局 · 公共导出 70 个符号

本文档描述仓库的物理结构与模块职责，是阅读代码与二次开发的导航图。
设计取舍与开发计划见 plan/DEVELOPMENT_PLAN.md，使用方式见 README.md。

---

## 1. 概览

rag-data 是「历史文本导入与向量化」管道：把文件解析为句子级切块，编码为向量，
写入向量库，并按相似度检索。整体分为七层，层间只依赖下层，方向单一。

| 层 | 包 | 职责 |
| :--- | :--- | :--- |
| 配置层 | config.py | 分区配置模型，环境变量与代码硬编码 |
| 契约层 | models.py、exceptions.py | 数据模型、类型协议、领域异常 |
| 通用设施 | registry.py、logging | 通用注册表、日志适配层 |
| 领域服务 | ingestion、embedding | 解析切块、实体抽取、向量化 |
| 存储层 | storage | 抽象接口、编解码、内存与 Milvus 实现 |
| 门面层 | facade.py | 分流程装配 + 一站式调用 |
| 公共入口 | __init__.py | 统一导出，外部只依赖包根 |

---

## 2. 目录结构

```text
rag-data/
├─ pyproject.toml             构建与依赖声明（hatchling + hatch 环境）
├─ README.md                  使用说明：安装、快速开始、公共 API、各专题
├─ rag_artifact.md            架构设计源文档
├─ struct.md                  本文件：项目结构说明
├─ plan/
│  └─ DEVELOPMENT_PLAN.md     设计与开发计划（含分模块设计要点）
├─ src/
│  └─ rag_data/
│     ├─ __about__.py         版本号（hatch 版本来源）
│     ├─ __init__.py          公共入口，统一导出 70 个符号
│     ├─ py.typed             类型标记（PEP 561）
│     ├─ config.py            分区配置模型与环境变量解析
│     ├─ models.py            pydantic 数据模型与 Milvus 结构协议
│     ├─ exceptions.py        领域异常层次
│     ├─ registry.py          通用类型注册表（存储与嵌入共用）
│     ├─ facade.py            流程门面与一键入口
│     ├─ ingestion/          离线导入：解析 - 切块 - 实体 - 编排
│     │  ├─ parsers.py        文件读取（md/txt 直读，PDF/Word 待补）
│     │  ├─ chunking.py       两阶段语义切块
│     │  ├─ entities.py       实体抽取（spaCy，含回退）
│     │  └─ pipeline.py       导入编排：切块 - 编码 - 构造记录 - 落库
│     ├─ embedding/          向量化：抽象、注册表、内置 provider
│     │  ├─ base.py            BaseEmbeddingProvider 抽象
│     │  ├─ registry.py        provider 注册表
│     │  ├─ openai_provider.py OpenAI 实现（标准库 HTTP）
│     │  ├─ qwen_provider.py   通义千问实现（继承 OpenAI）
│     │  └─ embedder.py        按配置装配 provider，统一批次
│     ├─ storage/            向量库：接口、编解码、注册表、实现
│     │  ├─ base.py            BaseVectorStore 抽象
│     │  ├─ schema.py          存储行编解码与字段提升规则
│     │  ├─ registry.py        后端注册表
│     │  ├─ memory_store.py    内存实现（零依赖）
│     │  └─ milvus_store.py    Milvus 实现（建表、写入、检索）
│     └─ logging/            日志适配层
│        ├─ base.py            LoggerAdapter 抽象与 LogContext
│        ├─ formatters.py      人类可读与 JSON 格式化器
│        ├─ factory.py         后端解析与装配
│        ├─ stdlib_adapter.py  标准库 logging 适配
│        ├─ loguru_adapter.py  loguru 适配
│        └─ structlog_adapter.py structlog 适配
└─ tests/                   单元测试（15 个测试文件，213 项）
```

---

## 3. 分层与依赖方向

依赖自上而下，无环。较低层不反向导入较高层。

```text
__init__（公共入口）
    │
facade（流程门面）
    ├── config          分区配置
    ├── models          数据模型与协议
    ├── registry        通用注册表
    ├── logging         日志适配层
    ├── ingestion       解析、切块、实体、编排
    │      └── embedding 向量化
    └── storage         接口、编解码、实现
           ▲
           │  注册表反向登记：
     embedding/base 延后导入 registry，storage/base 同样，
     避免包导入期循环依赖
```

关键约定：

1. 可选依赖（pymilvus、spacy、loguru、structlog）一律惰性导入，未安装时包仍可正常导入
2. 存储与嵌入的后端选择都走注册表，新增实现不改动装配代码
3. 唯一的外部入口是包根 __init__.py，内部子包路径不作为稳定契约

---

## 4. 基础模块

### 4.1 __about__.py

仅保存 __version__，由 pyproject.toml 的 [tool.hatch.version] 读取。

### 4.2 __init__.py（公共入口）

只做统一导出，不含业务逻辑。外部调用形如：

```python
import rag_data
from rag_data import Settings, RagData, InMemoryVectorStore
from rag_data import register_store, BaseEmbeddingProvider
```

导出 70 个符号，分组见 7.2 节。Milvus 实现因依赖 pymilvus 未纳入顶层导出。

### 4.3 config.py

职责：以嵌套模型承载分区配置，实例化即校验；默认读环境变量，支持代码硬编码。

```python
class StorageSettings(BaseModel): ...     # 向量库
class ModelSettings(BaseModel): ...       # 数据模型
class ChunkingSettings(BaseModel): ...    # 切块
class EmbeddingSettings(BaseModel): ...   # 向量化
class NLPSettings(BaseModel): ...         # 分词与实体
class LoggingSettings(BaseModel): ...     # 日志

ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"

class Settings(BaseSettings):
    @classmethod
    def load(cls, source=None, *, use_env=True, **overrides): ...
```

模块级函数：

| 函数 | 作用 |
| :--- | :--- |
| load_env_overrides(environ=None) | 把 RAG_ 前缀变量组装为分区字典 |
| _filter_sections(data) | 只保留已知分区，忽略无关的 RAG_ 变量 |
| _validate_sections(*groups) | 校验硬编码中的分区名合法 |
| _merge_sections(base, overrides) | 按分区逐字段合并，实现优先级覆盖 |
| _assign_nested(data, parts, value) | 按路径写入嵌套字典 |
| _parse_env_value(raw) | JSON 对象与数组按 JSON 解析 |

优先级：代码硬编码 > 环境变量与 .env > 字段默认值。

### 4.4 models.py

职责：记录类与领域数据模型，以及描述 pymilvus 结构的协议。

| 名称 | 说明 |
| :--- | :--- |
| MemoryRecord | 写入向量库的记录，基类字段 id、text_payload、vector、entities、created_at |
| DocumentChunk | 切块中间产物，含 chunk_id、user_id、source_path、text、entities、created_at |
| QueryHit | 检索结果，含 id、text_payload、score、entities、created_at |
| resolve_record_class(path) | 按点分路径解析记录类，默认 MemoryRecord |
| DEFAULT_RECORD_CLASS_PATH | 默认记录类路径常量 |
| MilvusModule / MilvusDataType / MilvusFieldSchema / MilvusCollectionSchema | 以 Protocol 描述 pymilvus 的最小结构，避免签名使用 Any 且不引入运行期依赖 |

MemoryRecord 的两个要点：

1. extra 允许未声明字段，经 extra_fields/to_row 与存储层往返
2. build_collection_schema 硬编码基础字段建表，子类可覆盖以扩展表结构

### 4.5 exceptions.py

单一根异常，便于上层统一捕获。

```text
RagDataError
├─ ConfigError            配置缺失或非法
├─ DataError              数据校验失败
├─ OptionalDependencyError 可选依赖未安装
│  └─ ParserDependencyError 文档解析依赖未安装
├─ ParseError             文档损坏或格式不支持
├─ SchemaMismatchError    维度或字段与集合不一致
├─ StoreError             向量库连接或读写失败
└─ EmbeddingError         向量化失败
```

### 4.6 registry.py（通用注册表）

存储后端与嵌入模型共用的注册表，dict 子类，支持惰性导入与统一实例化。

```python
class Registry(dict):
    def __init__(self, kind: str, builtins: Mapping[str, Ref]): ...
    def register(self, name, ref) -> None: ...      # ref 为类对象或点分路径
    def decorator(self, name): ...                  # 装饰器形式
    def names(self) -> List[str]: ...
    def resolve(self, name) -> Type[Any]: ...       # 路径登记则此时导入
    def create(self, name, **kwargs) -> Any: ...
```

登记类对象而非路径，可让函数内定义的本地类也能注册（qualname 含 locals 无法再导入）。

### 4.7 logging/

把三种日志库收敛到同一接口，调用方只依赖 LoggerAdapter。

| 文件 | 内容 |
| :--- | :--- |
| base.py | LoggerAdapter 抽象（bind、context、debug/info/warning/error/exception）、LogContext |
| formatters.py | HumanFormatter、JsonFormatter |
| factory.py | resolve_backend、configure_logging、get_logger 与各后端装配 |
| stdlib_adapter.py | 标准库 logging 适配 |
| loguru_adapter.py | loguru 适配 |
| structlog_adapter.py | structlog 适配 |

backend 配置为 auto 时按可用性择一；两个可选后端未安装时回落到标准库。


---

## 5. 领域服务层

### 5.1 ingestion/（离线导入）


```text
文件 ─ parse_document ─ 文本 ─ build_semantic_chunks ─ 切块
                                          │
                                extract_entities 抽实体
                                          │
                                     Embedder.encode
                                          │
                              构造记录 ─ store.upsert
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| parsers.py | parse_document(path, logger) | 读取文件为文本；md 与 txt 直读，PDF 与 Word 待补 |
| chunking.py | build_semantic_chunks(...) | 两阶段切块：先按版面结构分段，再句级切分并保留重叠 |
| entities.py | extract_entities(text, nlp=None) | 抽取实体；nlp 为空时走回退实现 |
| pipeline.py | IngestionPipeline | 导入编排，方法 run(paths, user_id) 与 ingest_file(path, user_id) |

切块参数来自 chunking 分区：max_chars（150 到 300）、overlap_sents。

Pipeline 写入约定：

1. 主键由 user_id、source_path 与切块文本做 sha256 生成，重复导入幂等
2. 按 embedding.batch_size 分批编码，降低单次请求压力
3. 用配置指定的记录类构造记录，自定义字段获得校验并随记录落库
4. 每个阶段绑定 source_path 与 user_id，便于链路定位

### 5.2 embedding/（向量化）

实现由配置 embedding.provider 选择，内置 openai 与 qwen。

```text
Embedder ─ create_embedding_provider ─ Registry ─┬─ OpenAIEmbeddingProvider
                                                └─ QwenEmbeddingProvider
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseEmbeddingProvider 抽象：声明 backend 即自动登记 |
| registry.py | provider 注册表与 register/resolve/create 系列函数 |
| openai_provider.py | OpenAIEmbeddingProvider，走标准库 urllib，无额外依赖 |
| qwen_provider.py | QwenEmbeddingProvider，继承 OpenAI（DashScope 兼容模式） |
| embedder.py | Embedder，按配置装配 provider，统一批次与维度校验 |

抽象类可声明的类属性：

| 属性 | 作用 |
| :--- | :--- |
| backend | 配置中的 provider 名，声明即自动登记 |
| default_model | 未配置 model 时使用的模型名 |
| default_dim | 该默认模型的输出维度 |
| default_base_url | 未配置 base_url 时的接口地址 |
| api_key_env | 未配置 api_key 时读取的环境变量名 |
| max_batch_size | 单次请求上限，0 表示不限制 |

内置 provider 对照：

| provider | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | text-embedding-v3 | 1024 | DASHSCOPE_API_KEY | 10 |

网络细节集中在 openai_provider 的四个接缝方法，子类与测试可覆盖：

| 方法 | 作用 |
| :--- | :--- |
| _endpoint() | 拼出 embeddings 端点 |
| _payload(texts) | 构造请求体，仅在维度被覆盖时传 dimensions |
| _headers() | 构造请求头（Bearer 令牌） |
| _post_json(url, payload) | 实际发送请求，可覆盖以隔离网络 |
| _parse(data) | 按 index 排序还原顺序，保证与输入一一对应 |

Embedder 的批次策略：取 embedding.batch_size 与 provider 的 max_batch_size 的较小者，
因此 Qwen 的 10 条上限自动生效，无需用户手动调小。


---

## 6. 存储层


```text
BaseVectorStore（抽象）
├─ InMemoryVectorStore   零依赖，字典存行，用于测试与本地验证
└─ MilvusVectorStore     惰性建连，建表委托记录类
        共用 schema.py 的编解码，落库行为一致
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseVectorStore 抽象：ensure_collection、upsert、query、close；声明 backend 即自动登记 |
| schema.py | 存储行编解码、字段提升规则、过滤匹配、索引参数 |
| registry.py | 后端注册表与 register/resolve/create 系列函数 |
| memory_store.py | 内存实现，含 cosine_similarity |
| milvus_store.py | Milvus 实现：建表、写入、检索、过滤表达式 |

schema.py 的函数：

| 函数 | 作用 |
| :--- | :--- |
| field_names() | 返回基类平铺字段名 |
| is_base_field(name) | 判断是否基类字段 |
| promoted_names(extra_columns) | 取出被提升为独立列的扩展字段名 |
| build_scalar_index_specs(names) | 生成标量索引声明 |
| build_vector_index_params(index_type, metric) | 生成向量索引参数 |
| metadata_fields(record) | 仅提取扩展字段 |
| encode_metadata(extras) | 扩展字段序列化为 JSON |
| decode_metadata(raw) | JSON 反序列化为扩展字段 |
| to_storage_row(record) | 记录到存储行 |
| from_storage_row(row) | 存储行还原为记录 |
| flatten_row(row) | 存储行摊平为单层字典 |
| row_matches(row, filters) | 按字段匹配，涵盖基类与扩展字段 |

落库策略：

| 场景 | 落库位置 |
| :--- | :--- |
| 普通扩展字段 | metadata JSON 列，表结构不变 |
| 子类建表新增的列 | 独立物理列，写入时平铺 |
| 配置提升的列 | 独立物理列，写入时平铺 |

milvus_store.py 的模块常量：DEFAULT_ALIAS（连接别名）、OUTPUT_FIELDS（回传字段）、
METADATA_PATH（JSON 路径模板）、HNSW_SEARCH_PARAMS 与 IVFLAT_SEARCH_PARAMS。

---

## 7. 门面层与公共 API

### 7.1 facade.py

分流程装配方法，每个方法对应一个可独立调用的流程：

| 方法 | 流程 |
| :--- | :--- |
| build_settings(source, **overrides) | 加载配置 |
| build_logger(settings) | 装配日志 |
| build_record_class(settings) | 解析记录类 |
| build_store(settings, logger, record_class=None) | 装配向量库 |
| build_nlp(settings, logger) | 加载 spaCy，不可用时降级为 None |
| build_embedder(settings, logger, model=None) | 装配向量化 |
| build_pipeline(settings, store, embedder, logger, nlp=None, record_class=None) | 装配导入管道 |

RagData 是一次装配、逐流程调用的门面：

```python
RagData(settings, *, store=None, embedder=None, logger=None,
        nlp=UNSET, model=None, record_class=None)
RagData.create(source=None, *, overrides=None, **kwargs)
```

| 实例方法 | 说明 |
| :--- | :--- |
| init_collection() | 建表建索引并 load，幂等 |
| ingest(paths, user_id=None) | 批量导入，返回写入条数 |
| ingest_file(path, user_id=None) | 单文件导入 |
| query(vector, top_n=None, user_id=None, filters=None) | 向量检索，默认取前 5 条 |
| close() | 释放资源；支持 with 语句 |

全部依赖可注入，故测试可替换 store、embedder、logger、nlp、record_class。
模块级一键入口 init_collection 与 ingest 内部自行装配后调用。

### 7.2 公共 API 导出（70 个符号）

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 子包 | config、models、ingestion、embedding、storage、logging |
| 配置 | Settings、StorageSettings、ChunkingSettings、EmbeddingSettings、NLPSettings、LoggingSettings、load_env_overrides、ENV_PREFIX、ENV_NESTED_DELIMITER |
| 数据模型 | DocumentChunk、MemoryRecord、QueryHit、resolve_record_class、DEFAULT_RECORD_CLASS_PATH、MilvusModule、MilvusDataType、MilvusFieldSchema、MilvusCollectionSchema |
| 导入管道 | IngestionPipeline、parse_document、build_semantic_chunks、extract_entities |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 存储 | BaseVectorStore、InMemoryVectorStore、register_store、register_backend、available_backends、is_registered、resolve_store、create_store |
| 流程门面 | RagData、init_collection、ingest、build_settings、build_logger、build_store、build_record_class、build_nlp、build_embedder、build_pipeline、DEFAULT_TOP_N |
| 日志 | LoggerAdapter、configure_logging、get_logger |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、SchemaMismatchError、StoreError、EmbeddingError |

未纳入顶层导出的实现按需从子包引入，如 MilvusVectorStore 与 logging 三个适配器。


---

## 8. 数据流

### 8.1 写入（离线导入）

```text
1. RagData.create(overrides)      装配 settings、logger、record_class、store、nlp、embedder、pipeline
2. app.init_collection()          确保集合与索引存在
3. app.ingest(paths, user_id)     逐文件导入，返回写入条数
     ├─ parse_document             读取文件为文本
     ├─ build_semantic_chunks      两阶段切块
     ├─ extract_entities           抽取实体
     ├─ Embedder.encode            分批编码为向量
     ├─ 构造 MemoryRecord 子类     校验并由 schema 编解码
     └─ store.upsert               幂等写入
```

### 8.2 读取（向量检索）

```text
1. 外部完成 query 文本编码（Embedder.encode 或自有模型）
2. app.query(vector, top_n, user_id, filters)
     ├─ 内存实现按余弦排序
     └─ Milvus 实现转过滤表达式并检索
3. 返回检索结果列表，按相似度降序
```

filters 为字段精确匹配，键可指向基类字段、提升列或扩展字段。

---

## 9. 配置项

分区与字段（括号内为默认值）：

| 分区 | 字段 |
| :--- | :--- |
| storage | backend(memory)、milvus_uri(localhost:19530)、collection_name(memory_store)、vector_dim(1024)、metric(COSINE)、index_type(HNSW) |
| models | record_class(rag_data.models.MemoryRecord)、promoted_fields(空) |
| chunking | max_chars(250，限 150 到 300)、overlap_sents(1) |
| embedding | provider(openai)、model(空)、batch_size(128)、dim(0)、api_key(空)、base_url(空)、timeout(60.0) |
| nlp | spacy_model(zh_core_web_sm) |
| logging | backend(auto)、level(INFO)、json(false) |

三层来源，优先级由高到低：代码硬编码、环境变量与 .env、字段默认值。
环境变量命名：RAG_ 前缀，分区与字段大写，双下划线分隔，如 RAG_STORAGE__VECTOR_DIM。

维度一致性：embedding 的输出维度需等于 storage.vector_dim。
默认 openai 为 1536 维、默认 vector_dim 为 1024 维，二者不一致，
故选择模型时须同时对齐集合维度。

---

## 10. 扩展点

三处登记机制让新增实现无需改动装配代码，均为继承或填路径即生效。

| 扩展点 | 抽象基类 | 声明方式 | 配置键 | 内置 |
| :--- | :--- | :--- | :--- | :--- |
| 向量库 | BaseVectorStore | backend 类属性 | storage.backend | memory、milvus |
| 嵌入模型 | BaseEmbeddingProvider | backend 类属性 | embedding.provider | openai、qwen |
| 记录类 | MemoryRecord | 填写类路径 | models.record_class | MemoryRecord |

新增向量库的最小骨架：

```python
from rag_data import BaseVectorStore


class SqliteVectorStore(BaseVectorStore):
    backend = "sqlite"

    def ensure_collection(self) -> None: ...
    def upsert(self, records) -> int: ...
    def query(self, vector, top_n, filters=None): ...
```

记录类另有两个可覆盖点：build_collection_schema 自定义表结构，
class 与 to_row 相关的编解码由 storage/schema 统一处理。

---

## 11. 测试

```text
tests/
├─ conftest.py                    路径注入与 settings、logger 夹具
├─ test_config.py         27 项   默认值、环境变量、硬编码优先、分区校验
├─ test_models.py         12 项   数据模型与字段校验
├─ test_schema.py         25 项   存储行编解码、提升列、过滤匹配
├─ test_store_contract.py 14 项   存储接口契约（内存实现）
├─ test_store_registry.py 15 项   后端注册表与配置驱动切换
├─ test_milvus_mapping.py 33 项   Milvus 建表、映射与过滤表达式
├─ test_record_class.py   16 项   记录类解析、子类建表、扩展字段往返
├─ test_chunking.py        8 项   两阶段切块与重叠
├─ test_entities.py        3 项   实体抽取与回退
├─ test_pipeline.py        4 项   导入编排、幂等、缺失文件
├─ test_embedder.py        4 项   批次切分、维度校验、注入优先
├─ test_embedding_provider.py 26 项 provider 注册、配置解析、请求构造、错误包装
├─ test_logging.py         6 项   后端选择与适配器行为
├─ test_facade.py         16 项   分流程装配与门面调用
└─ test_public_api.py      6 项   公共入口端到端
```

当前合计 213 项通过、2 项跳过（需外部服务的集成测试）。

---

## 12. 依赖

核心依赖仅两项：pydantic 与 pydantic-settings。

| 组 | 包 |
| :--- | :--- |
| 核心 | pydantic、pydantic-settings |
| parsers | unstructured、spacy |
| embedding | sentence-transformers |
| milvus | pymilvus |
| qdrant | qdrant-client |
| loguru | loguru |
| structlog | structlog |
| dev | pytest、pytest-cov、mypy、ruff |

嵌入模型走标准库 HTTP，故选用 openai 或 qwen 时无需额外依赖。

---

## 13. 待补全

| 位置 | 事项 |
| :--- | :--- |
| ingestion/parsers.py | PDF 与 Word 解析，基于 unstructured 或 MinerU |
| embedding/embedder.py | 依据 storage.metric 决定是否做 L2 归一化 |
| embedding | 兼容本地模型的 provider，如 sentence-transformers |
| facade.py | build_nlp 的管道细化，按流程启用 ner 与句子边界 |
| storage/milvus_store.py | format_literal 的注入防护 |

