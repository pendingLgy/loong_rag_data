# rag-data 代码开发计划（向量化与数据导入）

依据文档：rag_artifact.md（历史文本数据导入与长期记忆构建架构设计）

修订说明：

1. 项目范围收窄，移除在线记忆召回模块，仅保留向量化与离线数据导入
2. 数据模型与配置统一采用 pydantic 做类型校验
3. 新增日志适配层，支持多种日志框架切换
4. 配置改为按模块分区，默认读取环境变量，并支持代码硬编码
5. 新增流程门面，各流程调用集中封装
6. 记录类支持通过配置切换，便于继承扩展字段

## 0. 项目现状盘点

- pyproject.toml：hatchling 构建，源码根为 src 下的 rag_data 包，Python 需 3.8 以上；已声明核心与可选依赖、mypy 的 pydantic 插件、pytest 与 coverage 配置；hatch 环境分 default 与 rag_data_dev 两套，依赖由 uv 处理
- src 下的 rag_data：核心包已生成，含 config、models、exceptions、facade、ingestion、embedding、storage、logging
- ingestion：解析、两阶段切块、实体抽取与导入编排
- embedding：向量化封装，含批次与维度校验
- storage：Schema 声明式定义与存储行编解码、抽象接口、后端注册表、内存实现、Milvus 实现（建表、写入与检索已落地）
- logging：日志适配层，含 stdlib、loguru、structlog 三后端
- facade：流程门面，一站式装配与调用
- tests：单元测试覆盖模型、配置、日志、切块、实体、向量化、Schema、存储契约、导入管道、公共入口、记录类解析、流程门面与 Milvus 建表
- README.md：项目说明、公共 API、流程门面、Milvus 建表、数据模型与扩展、配置
- plan：本设计与开发计划
- rag_artifact.md：架构设计源文档
- 依赖现状：核心依赖 pydantic 与 pydantic-settings；可选依赖含 pymilvus、spacy、unstructured、loguru、structlog 等
- 待补全：涉及外部服务的复杂逻辑以 TODO 标注，主要集中在文档解析（PDF 与 Word）、真实模型加载、spaCy 默认加载
- 当前目录不是 git 仓库

结论：骨架、类型模型、日志适配层、流程门面与 Milvus 建表已就绪，单元测试可运行；文档解析与真实模型加载待人工补全。

## 1. 目标与范围

### 1.1 目标

实现文档描述的离线批量导入与向量化管道：

1. 类型模型：以 pydantic 定义并校验数据模型与配置，边界处拒绝脏数据
2. 文档解析：将 PDF、Word、Markdown 等历史文档转为带层级的纯文本
3. 两阶段语义切块：版面结构切分后再做句级切分与重叠
4. 实体抽取：抽取实体并写入标量字段，便于后续过滤
5. 向量化：批量将切块文本编码为稠密向量
6. 批量写入：将向量与元数据写入向量数据库
7. 日志适配：以统一接口适配多种日志框架，业务代码与框架解耦
8. 流程门面：把各流程调用集中封装，外部调用无需逐个导入子模块

### 1.2 范围内

- 模块一：批量切块 Batch Chunking
- 模块二：向量库 Schema，Milvus 为主、Qdrant 可插拔
- 向量化模块：Embedding 封装与批量编码
- 类型层：pydantic 模型与 pydantic-settings 配置
- 日志适配层：stdlib logging、loguru、structlog 多后端
- 离线导入管道（以库形式对外提供，不含命令行）

### 1.3 明确不涉及

- 在线记忆召回引擎（向量粗筛、重排）
- 时间衰减与召回频次巩固算法
- 异步反馈闭环
- 知识图谱与 Entity Store 服务化
- 多租户鉴权与配额
- 在线 HTTP 或 gRPC 网关
- 分布式链路追踪与指标上报

## 2. 技术栈与依赖

| 领域 | 选型 | 备注 |
| :--- | :--- | :--- |
| 类型校验 | pydantic v2 | 数据模型统一继承 BaseModel |
| 配置管理 | pydantic-settings | BaseSettings 支持环境变量与 .env |
| 文档解析 | Unstructured、MinerU | 需带 Markdown 层级输出 |
| 分句与实体 | spaCy、zh_core_web_sm | 停用 ner 与 parser，启用 sentencizer |
| 向量化 | 外部 Embedding 服务（openai、qwen）或自建 provider | 默认 openai；维度需与 storage.vector_dim 对齐 |
| 向量库 | pymilvus、可选 qdrant-client | 抽象 BaseVectorStore 接口 |
| 日志框架 | stdlib logging、loguru、structlog | 由适配层统一封装，可切换 |
| 构建与依赖 | hatch 管理构建与运行环境，uv 解析与安装依赖 | 构建后端为 hatchling |
| 测试 | pytest、pytest-cov | 已配置 coverage |
| 质量 | mypy、ruff | 配合 pydantic 插件做类型检查 |

依赖分层：pydantic 与 pydantic-settings 为核心必装；loguru、structlog、qdrant-client、sentence-transformers 均为可选依赖，按需安装。

## 3. 目标目录结构

计划采用如下结构，与现有 src 布局一致：

```text
rag-data
├─ pyproject.toml
├─ README.md
├─ DEVELOPMENT_PLAN.md
├─ src
│  └─ rag_data
│     ├─ __about__.py        版本号
│     ├─ __init__.py         公共入口，统一导出 API
│     ├─ config.py           pydantic-settings 全局配置
│     ├─ models.py           pydantic 数据模型与记录类解析
│     ├─ exceptions.py       领域异常基类
│     ├─ facade.py           流程门面，封装各流程调用方法
│     ├─ ingestion           解析与切块
│     │  ├─ parsers.py       文档解析与结构化
│     │  ├─ chunking.py      两阶段语义切块
│     │  ├─ entities.py      spaCy 实体抽取
│     │  └─ pipeline.py      离线导入编排
│     ├─ embedding           向量化接口、provider 注册表与 openai、qwen 实现
│     │  └─ embedder.py      Embedding 封装
│     ├─ storage             向量库
│     │  ├─ base.py          BaseVectorStore 接口
│     │  ├─ schema.py        Schema 声明与存储行编解码
│     │  └─ milvus_store.py  Milvus 实现（建表、写入、检索）
│     └─ logging             日志适配层
│        ├─ base.py          LoggerAdapter 抽象
│        ├─ stdlib_adapter.py
│        ├─ loguru_adapter.py
│        ├─ structlog_adapter.py
│        └─ factory.py       get_logger 与自动探测
└─ tests
   ├─ test_models.py
   ├─ test_config.py
   ├─ test_logging.py
   ├─ test_chunking.py
   ├─ test_entities.py
   ├─ test_embedder.py
   ├─ test_schema.py
   ├─ test_store_contract.py
   ├─ test_pipeline.py
   ├─ test_public_api.py
   ├─ test_record_class.py
   ├─ test_facade.py
   └─ test_milvus_mapping.py
```

说明：日志适配层目录名为 logging，位于 rag_data 包内，采用绝对导入避免与标准库 logging 冲突；如仍担心歧义，可改名为 observability。

## 4. 里程碑与任务分解

### M0 项目骨架与环境（0.5 人天）

任务：

- 创建 src 下的 rag_data 包及 __about__.py，定义版本号供 hatch 读取
- 创建各子包 __init__.py
- 在 pyproject.toml 声明核心依赖 pydantic、pydantic-settings 与可选依赖 loguru、structlog、pymilvus 等
- 声明 dev 依赖 pytest、mypy、ruff 以及 pydantic 的 mypy 插件
- 配置 hatch 环境：default 含 dev 工具，rag_data_dev 叠加全部可选依赖，并定义 test、types、check、format 脚本
- 依赖安装与锁定交由 uv 处理，uv 直接读取 pyproject.toml 的依赖声明
- 创建 README.md

交付：可本地以可编辑方式安装并可导入 rag_data
验收：mypy 对空包无报错

### M1 类型模型与配置（1 人天）

任务：

- models.py：以 pydantic BaseModel 定义 DocumentChunk、MemoryRecord、QueryHit
- 为关键字段加约束：text 非空、vector 维度校验、score 取值 0 到 1
- config.py：以 pydantic-settings BaseSettings 定义 Settings，按模块分区，含 env_prefix、嵌套分隔符、.env 加载
- 对互相约束的字段使用 model_validator，对单字段使用 field_validator
- 配置来源优先级：初始化参数 大于 环境变量 大于 .env 大于 默认值

交付：models.py、config.py
验收：非法输入抛 ValidationError 且信息可读；环境变量可覆盖默认值

### M2 日志适配层（1 人天）

任务：

- logging 子包的 base.py：定义 LoggerAdapter 抽象接口与 LogContext
- 实现 stdlib_adapter、loguru_adapter、structlog_adapter，各自惰性导入对应框架
- logging 子包的 factory.py：configure_logging(settings) 与 get_logger(name)
- 支持自动探测：按 settings.logging.backend 指定，或 auto 时按已安装框架择优选 stdlib 回退
- 统一支持结构化字段绑定 bind、上下文 context、级别过滤与可选 JSON 输出

交付：logging 子包
验收：切换 logging.backend 输出格式随之变化；未安装某框架时给出清晰提示而非报错

### M3 两阶段语义切块（1 人天）

对应文档 模块一。

任务：

- 在 ingestion 的 chunking.py 实现 build_semantic_chunks（text、max_chars、overlap_sents）
- 复用文档示例逻辑，补齐边界处理：空文本、超长单句、overlap 大于已累积句数
- 保持纯函数、无 IO，便于单元测试
- 以 LoggerAdapter 记录超长单句等告警，不直接依赖某个日志框架

交付：chunking.py
验收：单测覆盖 空文本、单句、恰好等于上限、需 overlap 的跨块、超长单句

### M4 向量库 Schema（1 人天）

对应文档 模块二，字段按向量化范围裁剪。

任务：

- schema.py 定义字段：id、text_payload、vector、entities、created_at、metadata
- 定义索引：主键、标量索引（entities、created_at）、向量索引 HNSW、度量 COSINE
- 提供存储行编解码：记录与行之间的双向映射，扩展字段进 metadata
- 提供建表所需生成器：字段定义、向量索引参数、标量索引配置
- base.py 抽象接口：ensure_collection、upsert、query
- milvus_store.py 实现，建表前用 Settings.vector_dim 与模型维度做校验

交付：schema.py、base.py、milvus_store.py
验收：本地 Milvus 可建表并写入读取；接口用内存 Fake 实现做契约测试

### M5 向量化模块与导入管道（2 人天）

任务：

- embedding：provider 注册表与 openai、qwen 实现，批量 embedding，输出维度需与 storage.vector_dim 一致
- ingestion 的 entities.py：spaCy NER 抽取实体列表
- ingestion 的 parsers.py：PDF、Word、Markdown 转带层级文本，清理页眉页脚与断词
- ingestion 的 pipeline.py：解析 到 切块 到 实体抽取 到 向量化 到 upsert，支持分批与幂等
- 全链路以 pydantic 模型传递数据，构造 MemoryRecord 时自动校验维度与字段
- 记录类可由配置指定，写入与读回共用同一个类

交付：embedder.py、entities.py、parsers.py、pipeline.py
验收：样例文档端到端跑通，写入条数正确，重复导入幂等

### M6 公共入口与流程门面（1 人天）

任务：

- 包根 __init__.py 统一导出全部公共 API，并登记 __all__
- facade.py 封装各流程调用：配置、日志、存储、句柄、向量化、管道
- RagData 提供 init_collection、ingest、ingest_file、query、close，支持依赖注入与上下文管理
- build_record_class 解析配置中的记录类路径，支持继承扩展

交付：__init__.py、facade.py
验收：外部仅导入 rag_data 即可完成全部流程；__all__ 全部可解析

### M7 测试与文档（1 人天）

任务：

- 补齐测试与覆盖率、README、本计划
- 通过 mypy 与 ruff

验收：pytest 全绿且覆盖核心模块，mypy 无错误

## 5. 关键接口契约

- config.Settings：pydantic-settings 的 BaseSettings，按模块分区，实例化即完成校验
- models.DocumentChunk、models.MemoryRecord、models.QueryHit：pydantic BaseModel
- models.MemoryRecord：extra 允许，支持子类声明扩展字段或运行时传入，通过 extra_fields 与 to_row 暴露
- models.resolve_record_class（path）返回 MemoryRecord 的子类；非法路径抛 ConfigError
- logging.factory.configure_logging（settings）返回 LoggerAdapter
- logging.factory.get_logger（name）返回 LoggerAdapter
- logging.base.LoggerAdapter：debug、info、warning、error、exception、bind、context
- ingestion.parsers.parse_document（path）返回纯文本字符串
- ingestion.chunking.build_semantic_chunks（text，max_chars 默认 250，overlap_sents 默认 1，nlp 可选，logger 可选）返回切块列表
- ingestion.entities.extract_entities（text，nlp 可选）返回实体列表
- embedding.embedder.Embedder.encode（texts）返回向量列表
- storage.schema：to_storage_row、from_storage_row、flatten_row、row_matches、build_collection_fields
- storage.base.BaseVectorStore：ensure_collection、upsert（records）、query（vector，top_n，filters 可选）
- ingestion.pipeline.IngestionPipeline：ingest_file（path，user_id 可选）、run（paths，user_id 可选）返回写入条数
- facade.RagData：init_collection、ingest、ingest_file、query、close，支持上下文管理
- 一键入口：init_collection（source）、ingest（paths，source，user_id）

## 6. 配置与默认值

配置按模块分区，与 JSON 结构一一对应；模型为 pydantic 嵌套模型，环境变量与 JSON 均可覆盖。

| 模块 | 字段 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| storage | backend | memory | 存储后端名，对应注册表键；继承 BaseVectorStore 即自动注册 |
| storage | milvus_uri | localhost:19530 | Milvus 服务地址 |
| storage | collection_name | memory_store | 集合名 |
| storage | vector_dim | 1024 | 向量维度，建表前校验 |
| storage | metric | COSINE | 距离度量 |
| storage | index_type | HNSW | 向量索引类型 |
| models | record_class | rag_data.models.MemoryRecord | 记录类点分路径，切换自定义扩展 |
| models | promoted_fields | 空列表 | 需提升为独立列的扩展字段，形如 name、type、max_length |
| chunking | max_chars | 250 | 切块字符上限，区间 150 至 300 |
| chunking | overlap_sents | 1 | 切块重叠句数 |
| embedding | model | 空字符串 | Embedding 模型标识 |
| embedding | batch_size | 128 | 批量写入与编码大小 |
| nlp | spacy_model | zh_core_web_sm | 句法与实体模型 |
| logging | backend | auto | stdlib、loguru、structlog 或 auto |
| logging | level | INFO | 日志级别 |
| logging | json | false | 是否结构化 JSON 输出 |

配置来源优先级：代码硬编码 大于 环境变量 大于 .env 大于 默认值。

已移除：half_life_days、beta、reinforcement_cap、top_n、top_k（原属召回模块）。

## 7. 测试策略

- 类型层：models 与 config 的校验用例，覆盖非法值、边界值、环境变量覆盖
- 日志层：以参数化用例逐一验证 stdlib、loguru、structlog 适配器输出与字段绑定
- 纯逻辑：chunking、entities、embedder，外部依赖最小化
- 存储层用内存 Fake 实现测接口契约；Milvus 集成测试标记为 slow，默认跳过
- 建表层：以假 pymilvus 注入 sys.modules，验证字段生成、索引参数与建表幂等，无需真实服务
- 端到端：小样例文档经内存 store 到写入断言
- 覆盖率目标：核心模块 85 以上

## 8. 依赖与风险

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| pydantic v1 与 v2 语法差异 | 配置与校验写法不兼容 | 锁定 v2，统一使用 ConfigDict、field_validator、model_validator |
| 日志框架为可选依赖 | 未安装时导入失败 | 适配器惰性导入，缺失时回退 stdlib 并给出提示 |
| 子包名 logging 与标准库同名 | 误导入歧义 | 包内一律绝对导入，必要时改名 observability |
| spaCy 中文模型体积与安装 | 首次环境搭建慢 | 模型下载脚本化与 CI 缓存 |
| Milvus 本地依赖较重 | 集成测试不便 | 抽象接口加内存 Fake，建表用假 pymilvus 测试，集成测试可选 |
| Unstructured 与 MinerU 版本差异 | 解析结果不稳定 | 锁版本并加解析回归样例 |
| Embedding 模型与维度不一致 | 索引与查询不匹配 | 维度写入配置，pydantic 与建表双重校验 |
| 过滤表达式拼接 | 存在注入风险 | 字面量转义已实现基础防护，生产需补强，已标 TODO |
| Python 3.8 运行期类型语法 | list 下标等新语法在 3.8 报错 | 文件首行加 from __future__ import annotations |

## 9. 建议排期（合计约 7.5 人天）

| 里程碑 | 内容 | 人天 |
| :--- | :--- | :--- |
| M0 | 骨架与环境 | 0.5 |
| M1 | 类型模型与配置（pydantic） | 1.0 |
| M2 | 日志适配层 | 1.0 |
| M3 | 两阶段切块 | 1.0 |
| M4 | 向量库 Schema | 1.0 |
| M5 | 向量化模块与导入管道 | 2.0 |
| M6 | 公共入口与流程门面 | 1.0 |
| M7 | 测试与文档 | 1.0 |
| 合计 | | 7.5 |

## 10. 落地顺序与依赖

M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7

说明：M1 类型层与 M2 日志层为横切基础，先行落地；M3 与 M4 无相互依赖，可并行；M5 依赖 M3 与 M4；M6 依赖 M5；M7 收口。本计划不含召回链路，故无召回分支。

## 11. 代码层级设计说明

### 11.1 分层结构与依赖方向

```text
RagData（facade）              表现层：流程门面
   |
IngestionPipeline            应用编排层
   |
parsers、chunking、entities、embedder   领域服务层
   |
BaseVectorStore              抽象接口层
   |
MilvusVectorStore 与 InMemoryVectorStore   基础设施层

config.Settings（pydantic-settings）   横切层：类型安全的只读配置
models（pydantic BaseModel）          横切层：数据契约与校验
logging（LoggerAdapter）              横切层：日志抽象，屏蔽具体框架
```

依赖原则：

1. 上层只依赖下层的抽象 BaseVectorStore，不直接依赖 pymilvus 等 SDK
2. 具体存储实现通过构造注入，即依赖反转，便于用 InMemoryVectorStore 做测试
3. 领域服务层只依赖 LoggerAdapter 抽象，不 import loguru 或 structlog
4. config 为不可变且已校验的配置对象，逐层向下传递，禁止模块级全局可变状态
5. 所有跨层数据以 pydantic 模型承载，边界处自动校验

### 11.2 数据模型（pydantic）

```python
class MemoryRecord(BaseModel):
    # extra=allow 支持继承与扩展：
    # 1. 子类可声明类型化字段，例如 user_id、tags
    # 2. 也可直接传入未声明字段，落入扩展字段集合
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    text_payload: str = Field(min_length=1)
    vector: List[float] = Field(min_length=1)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)

    @field_validator("vector")
    @classmethod
    def _check_vector(cls, value: List[float]) -> List[float]: ...

    @property
    def extra_fields(self) -> Dict[str, Any]: ...

    def to_row(self) -> Dict[str, Any]: ...


class QueryHit(BaseModel):
    id: str = Field(min_length=1)
    text_payload: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)
```

字段约定：

| 模型 | 字段 |
| :--- | :--- |
| MemoryRecord（基类） | id、text_payload、vector、entities、created_at |
| QueryHit | id、text_payload、score、entities、created_at |

继承与扩展用法：

```python
# 方式一：子类声明类型化字段，获得校验与 IDE 提示
class TenantRecord(MemoryRecord):
    user_id: str
    tags: List[str] = []

# 方式二：直接传入未声明字段，无需定义子类
record = MemoryRecord(id="m1", text_payload="t", vector=[0.1], created_at=1.0, user_id="u1")
record.extra_fields   # {"user_id": "u1"}
record.to_row()       # 含全部字段，供存储层落库
```

记录类切换：

```python
# 配置中指定，写入与读回共用该类
{"models": {"record_class": "myapp.models.TenantRecord"}}

from rag_data import resolve_record_class
resolve_record_class("myapp.models.TenantRecord")
```

设计要点：

1. 主键统一为 id，不再使用 memory_id；基类不再包含 user_id
2. extra 设为 allow，基类约束收敛，扩展能力交给子类或运行时传入
3. extra_fields 与 to_row 屏蔽扩展字段的具体名字，存储层无需硬编码
4. 基类字段仍保留约束（非空、向量非空、时间戳非负），子类自动继承校验
5. 原 extra=forbid 的拼写保护由测试覆盖，避免误传字段静默入库
6. 记录类可通过配置 models.record_class 指定，写入与读回共用同一个类
7. 建表权归属记录类：MemoryRecord.build_collection_schema 硬编码基础字段，子类可覆盖以自定义表结构

### 11.3 各模块职责与接口签名

#### 11.3.1 config.py（pydantic-settings，按模块分区）

职责：以嵌套模型承载各模块配置，实例化即完成校验；默认读取环境变量，也支持代码硬编码。

```python
class StorageSettings(BaseModel):
    backend: Literal["memory", "milvus"] = "memory"
    milvus_uri: str = "localhost:19530"
    collection_name: str = "memory_store"
    vector_dim: int = Field(default=1024, gt=0)
    metric: Literal["COSINE", "L2", "IP"] = "COSINE"
    index_type: Literal["HNSW", "IVFLAT"] = "HNSW"


class ModelSettings(BaseModel):
    record_class: str = DEFAULT_RECORD_CLASS_PATH
    promoted_fields: List[Dict[str, Any]] = []


ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        extra="ignore",
    )

    storage: StorageSettings = Field(default_factory=StorageSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def load(cls, source=None, *, use_env=True, **overrides) -> Settings: ...
```

加载入口与优先级：

| 入口 | 说明 |
| :--- | :--- |
| Settings() | 默认读取环境变量（含 .env），未提供时用字段默认值 |
| Settings.load() | 同上，显式表达按环境变量加载 |
| Settings.load(storage={...}) | 代码硬编码分区配置，逐字段覆盖环境变量 |
| Settings.load({...}) | 代码硬编码整体字典，与分区写法等价 |
| Settings.load(use_env=False) | 忽略环境变量，仅取硬编码与字段默认值 |

优先级由高到低：代码硬编码 > 环境变量与 .env > 字段默认值。
同时存在时以硬编码为准；未被硬编码的字段由环境变量填充。

环境变量辅助项：

| 名称 | 作用 |
| :--- | :--- |
| load_env_overrides(environ=None) | 把 RAG_ 前缀变量组装为分区字典，供 load 分层合并 |
| ENV_PREFIX | 前缀常量 RAG_ |
| ENV_NESTED_DELIMITER | 分区与字段之间的分隔符常量 |

设计要点：

1. 环境变量作为基线，代码硬编码在其上逐字段覆盖，即环境提供默认、显式配置最终生效
2. 每个分区模型 extra 为 forbid，分区内拼错字段会在加载期即报错
3. 根模型 extra 为 ignore：RAG_ 命名空间可能混入无关变量，避免误报
4. 代码硬编码中的分区名由 _validate_sections 校验，写错即抛 ConfigError 并列出可用分区
5. 环境变量支持嵌套覆盖，如 RAG_STORAGE__VECTOR_DIM 覆盖 storage.vector_dim
6. 取值交由 pydantic 转换，JSON 对象与数组按 JSON 解析
7. logging 分区中 json 为保留名，字段名为 json_output 并设置别名 json
8. use_env 为 False 时走 model_validate，绕过环境变量源，结果只取决于显式配置
9. load 的 source 为整体配置字典，与分区参数 overrides 同属代码硬编码
10. 不提供配置文件读取：配置只来自代码与环境变量，避免部署期文件依赖
#### 11.3.2 logging 子包（多框架适配）

职责：以统一接口封装不同日志框架，业务代码只依赖抽象，框架可插拔。

##### base.py

```python
class LoggerAdapter(ABC):
    def bind(self, **fields) -> LoggerAdapter: ...
    def context(self, **fields) -> LogContext: ...
    def debug(self, msg, **fields) -> None: ...
    def info(self, msg, **fields) -> None: ...
    def warning(self, msg, **fields) -> None: ...
    def error(self, msg, **fields) -> None: ...
    def exception(self, msg, **fields) -> None: ...
```

##### stdlib_adapter.py、loguru_adapter.py、structlog_adapter.py

| 适配器 | 后端 | 说明 |
| :--- | :--- | :--- |
| StdlibAdapter | 标准库 logging | 默认回退实现，无额外依赖 |
| LoguruAdapter | loguru | 以 logger.bind 实现字段绑定 |
| StructlogAdapter | structlog | 以结构化事件字段输出，天然支持 JSON |

实现约定：

1. 各适配器惰性导入对应框架，未安装时抛可选依赖缺失异常并在 factory 层降级
2. bind 返回新实例，不修改共享状态，保证并发安全
3. context 使用上下文管理器在块内临时绑定字段
4. 输出字段统一携带 name、level、timestamp 与自定义业务字段

##### factory.py

```python
def configure_logging(settings: Settings) -> LoggerAdapter: ...
def get_logger(name: str) -> LoggerAdapter: ...
```

设计要点：

1. logging.backend 为 auto 时按已安装框架择优选，缺省回退 stdlib
2. 应用启动调用一次 configure_logging，写入全局只读注册表供 get_logger 取用
3. 切换后端仅改配置，业务代码零改动

#### 11.3.3 ingestion 子包

##### parsers.py

- 职责：按扩展名分派解析器，输出带 Markdown 层级的纯文本，自动清理页眉页脚与换行断词
- 接口：def parse_document(path: str, logger=None) -> str
- 依赖：unstructured 或 MinerU；采用惰性导入，缺失时抛 ParserDependencyError
- TODO：PDF 与 Word 的解析尚未实现，文本类文档已支持

##### chunking.py

- 职责：两阶段语义切块，一阶段按版面段落，二阶段按句级切分并保留重叠
- 接口：def build_semantic_chunks(text, max_chars=250, overlap_sents=1, nlp=None, logger=None) -> List[str]
- 设计：nlp 句柄可注入以便测试；纯函数无 IO；超长单句单独成块并通过 LoggerAdapter 记录告警

##### entities.py

- 职责：基于 spaCy NER 抽取实体，用于标量索引与后续过滤
- 接口：def extract_entities(text, nlp=None) -> List[str]
- 设计：去重且保持首次出现顺序，大小写归一

##### pipeline.py

```python
class IngestionPipeline:
    def __init__(self, store, embedder, settings, logger, nlp=None, record_class=MemoryRecord): ...

    def run(self, paths: List[str], user_id: Optional[str] = None) -> int: ...
    def ingest_file(self, path: str, user_id: Optional[str] = None) -> int: ...
    def _make_record(self, chunk: DocumentChunk) -> MemoryRecord: ...

    @staticmethod
    def _content_hash(user_id: str, source_path: str, text: str) -> str: ...
```

设计要点：

1. 编排顺序为 解析 到 切块 到 实体抽取 到 向量化 到 批量 upsert
2. memory 主键由 user_id、source_path 与切块内容做 sha256 生成，重复导入幂等
3. 按 settings.embedding.batch_size 分批写入，降低单次请求压力
4. 使用配置指定的记录类构造记录，使自定义字段获得校验并随记录落库
5. 每个阶段以 bind 绑定 source_path 与 user_id，便于链路定位

#### 11.3.4 embedding 子包

职责：把文本编码为向量，实现由配置选择，内置 openai 与 qwen，第三方可按同样方式接入。

##### base.py（嵌入模型抽象）

```python
class BaseEmbeddingProvider(ABC):
    backend: ClassVar[Optional[str]] = None        # 配置 embedding.provider 的取值
    default_model: ClassVar[str] = ""
    default_dim: ClassVar[int] = 0
    default_base_url: ClassVar[str] = ""
    api_key_env: ClassVar[str] = ""
    max_batch_size: ClassVar[int] = 0              # 0 表示不限制

    def __init_subclass__(cls, **kwargs): ...      # 声明 backend 即自动登记
    def __init__(self, settings, logger): ...

    model: str            # 配置优先，缺省用 default_model
    base_url: str         # 配置优先，缺省用 default_base_url
    dim: int              # 配置优先，缺省用 default_dim
    api_key: str          # 配置优先，其次读 api_key_env 指向的环境变量

    def encode(self, texts: List[str]) -> List[List[float]]: ...
```

##### registry.py（嵌入模型注册表）

职责：以注册表解耦 provider 选择与装配代码，继承即注册，配置只写 provider 名。

```python
BUILTIN_PROVIDERS: Dict[str, str] = {
    "openai": "rag_data.embedding.openai_provider.OpenAIEmbeddingProvider",
    "qwen": "rag_data.embedding.qwen_provider.QwenEmbeddingProvider",
}

def register_embedding(name, ref) -> None: ...
def register_embedding_provider(name): ...        # 装饰器
def available_embedding_providers() -> List[str]: ...
def is_embedding_registered(name) -> bool: ...
def resolve_embedding_provider(name) -> Type[BaseEmbeddingProvider]: ...
def create_embedding_provider(name, settings, logger, **kwargs) -> BaseEmbeddingProvider: ...
```

##### openai_provider.py 与 qwen_provider.py

- openai：backend 为 openai，默认 text-embedding-3-small（1536 维），接口 api.openai.com 的 v1，Key 读 OPENAI_API_KEY
- qwen：backend 为 qwen，默认 text-embedding-v3（1024 维），接口为 DashScope 兼容模式，Key 读 DASHSCOPE_API_KEY
- qwen 直接继承 openai：DashScope 提供 OpenAI 兼容接口，仅默认值、地址、Key 变量与批量上限不同
- 请求走标准库 urllib，不引入额外依赖；HTTP 与网络错误统一包装为 EmbeddingError
- _post_json 为网络接缝，子类与测试可覆盖以隔离真实请求
- _parse 按 index 排序还原顺序，保证向量与输入文本一一对应
- 仅当显式配置 dim 且与默认维度不同时才传 dimensions，兼容不支持该参数的模型

##### embedder.py

- 职责：按配置装配 provider，统一批次与维度校验
- 接口：class Embedder 提供 encode(self, texts: List[str]) -> List[List[float]]
- 设计：provider 由 embedding.provider 经注册表解析；注入的 model 优先且绕过配置
- 设计：批次大小取 embedding.batch_size 与 provider 的 max_batch_size 的较小者
- 设计：输出维度需与 storage.vector_dim 一致，不一致抛 EmbeddingError
- 设计：空输入直接返回，不触发 provider 装配
- TODO：依据 storage.metric 决定是否做 L2 归一化，目前透传 provider 结果


#### 11.3.5 storage 子包

##### schema.py

职责：声明式定义集合字段、索引，并提供记录与存储行之间的编解码。

| 字段 | 类型 | 索引类型 | 说明 |
| :--- | :--- | :--- | :--- |
| id | VARCHAR | Primary Key | 唯一标识，内容哈希 |
| text_payload | VARCHAR | None | 原始记忆文本块 |
| vector | FLOAT_VECTOR | HNSW 或 IVFLAT | 语义向量，维度取自配置 |
| entities | ARRAY(VARCHAR) | Scalar Index | 实体列表，用于过滤 |
| created_at | DOUBLE | Scalar Index | 创建或录入时间戳 |
| metadata | JSON | None | 扩展字段容器 |

已移除字段：memory_id（改名为 id）、user_id、last_accessed_at、recall_count。

导出辅助：

| 名称 | 含义 |
| :--- | :--- |
| PRIMARY_FIELD | 主键字段名，值为 id |
| METADATA_FIELD | 扩展字段容器名，值为 metadata |
| BASE_COLUMNS | metadata 之外的平铺列 |
| SCALAR_INDEX_TYPES | 各字段的标量索引类型 |
| promoted_names(extra_columns) | 返回被提升为独立列的字段名 |
| build_collection_fields(vector_dim, extra_columns) | 生成建表字段定义 |
| build_vector_index_params(index_type, metric) | 生成向量索引参数 |
| build_scalar_index_specs(field_names) | 生成标量索引配置 |

存储行编解码（扩展字段的落库与还原）：

| 函数 | 作用 |
| :--- | :--- |
| to_storage_row(record, promoted) | 基类字段与提升列平铺，其余扩展字段写入 metadata |
| from_storage_row(row, record_class, promoted) | 提升列与 metadata 一并还原为记录 |
| flatten_row(row, promoted) | 存储行摊平为单层字典，供过滤复用 |
| row_matches(row, filters, promoted) | 在存储行上做字段精确匹配 |
| encode_metadata 与 decode_metadata | 扩展字段与 JSON 文本之间的双向转换 |
| is_base_field(name) | 判断字段属于平铺列还是扩展字段 |

设计约定：

1. 扩展字段默认只在 metadata 一处序列化，新增扩展字段无需改表结构
2. 提升为独立列的扩展字段不再重复写入 metadata，避免同一字段存两份
3. 内存实现与 Milvus 实现共用同一套编解码，避免两条落库路径产生行为差异
4. 内存实现保存的是存储行而非 pydantic 对象，使单元测试真实覆盖序列化与还原
5. 过滤在存储行层面进行，内存实现直接匹配字段，Milvus 实现转为 metadata 的 JSON 路径条件

##### registry.py（存储后端注册表）

职责：以注册表解耦后端选择与装配代码，继承即注册，配置只写后端名。

```python
BUILTIN_BACKENDS: Dict[str, str] = {
    "memory": "rag_data.storage.memory_store.InMemoryVectorStore",
    "milvus": "rag_data.storage.milvus_store.MilvusVectorStore",
}
BackendRef = Union[str, type]

def register_store(name: str, ref: BackendRef) -> None: ...   # ref 为类对象或点分路径
def register_backend(name: str): ...                      # 装饰器
def available_backends() -> List[str]: ...
def is_registered(name: str) -> bool: ...
def resolve_store(name: str) -> Type[BaseVectorStore]: ...
def create_store(name, settings, logger, **kwargs) -> BaseVectorStore: ...
```

设计约定：

1. BaseVectorStore.__init_subclass__ 检测子类的 backend 属性并自动注册，无需额外调用
2. 类对象直接入表，避免本地类因 qualname 无法按路径导入；字符串则惰性导入
3. 内置后端以点分路径注册，实现惰性导入，未装 pymilvus 时仍可正常导入 rag_data
4. 未注册的后端名抛 ConfigError，并在信息中列出可用后端
5. 创建实例统一以关键字传入 settings 与 logger，各实现保持一致签名


##### base.py

职责：定义存储层抽象接口，隔离具体向量库实现。

```python
class BaseVectorStore(ABC):
    def ensure_collection(self) -> None: ...
    def upsert(self, records: List[MemoryRecord]) -> int: ...
    def query(self, vector, top_n, filters=None) -> List[QueryHit]: ...
    def close(self) -> None: ...
```

接口契约：

1. upsert 按 id 幂等覆盖，返回实际写入条数；扩展字段一并落库
2. query 的 filters 为字段精确匹配，键可指向基类字段、提升列或扩展字段
3. query 返回按向量相似度降序的候选，score 归一化到 0 到 1，先过滤再排序
4. 入参与出参均为 pydantic 模型，避免裸字典漂移

已移除方法：update_scalar（原为召回反馈回写而设）。

##### models.build_collection_schema（建表入口）

建表权由记录类掌握，MemoryRecord 默认硬编码 MemoryRecord 声明的字段：

```python
class MemoryRecord(BaseModel):
    @classmethod
    def build_collection_schema(cls, pymilvus: MilvusModule, vector_dim: int) -> MilvusCollectionSchema:
        """按 MemoryRecord 声明的字段硬编码创建集合表结构。"""
        ...
        return pymilvus.CollectionSchema(fields=fields, description=...)
```

子类覆盖即可自定义表结构：

```python
class TenantRecord(MemoryRecord):
    user_id: str

    @classmethod
    def build_collection_schema(cls, pymilvus, vector_dim):
        collection_schema = super().build_collection_schema(pymilvus, vector_dim)
        collection_schema.fields.append(
            pymilvus.FieldSchema(name="user_id", dtype=pymilvus.DataType.VARCHAR, max_length=128)
        )
        return collection_schema
```

设计要点：

1. pymilvus 由调用方传入，模型层不依赖可选第三方库
2. 硬编码字段与 schema.BASE_COLUMNS 一一对应，一致性由测试锁定
3. 维度非法时抛 ConfigError，避免生成无效表结构
4. Milvus 侧仅负责调用与追加配置提升列，不再持有字段声明

##### milvus_store.py

```python
class MilvusVectorStore(BaseVectorStore):
    def __init__(self, settings, logger, record_class=None, extra_columns=None, alias="rag_data"): ...

    def column_names(self) -> List[str]: ...
    def promoted_columns(self) -> List[str]: ...
    def vector_index_params(self) -> Dict[str, Any]: ...
    def scalar_index_specs(self) -> List[Dict[str, Any]]: ...

    def ensure_collection(self) -> None: ...
    def upsert(self, records: List[MemoryRecord]) -> int: ...
    def query(self, vector, top_n, filters=None) -> List[QueryHit]: ...
```

建表流程（已实现）：

1. 惰性连接：首次调用时按 settings.storage.milvus_uri 连接并缓存
2. 集合不存在：调用记录类的 build_collection_schema 生成 schema，
   追加配置声明的提升列，再建表、建向量索引与标量索引，最后 load
3. 集合已存在：跳过建表，校验既有向量维度与配置一致，不一致抛 SchemaMismatchError
4. 向量索引：index_type 与 metric 取自配置，HNSW 使用 M 与 efConstruction
5. 标量索引：entities 用 INVERTED，created_at 用 STL_SORT
6. 写入：metadata 由 JSON 文本转为字典后提交，按 batch_size 分批，最后 flush
7. 未安装 pymilvus：抛 OptionalDependencyError 并给出安装指引

列名解析（支持子类覆盖建表）：

| 场景 | column_names 来源 |
| :--- | :--- |
| 未建表 | schema.BASE_COLUMNS 加配置提升列 |
| 已建表 | 集合实际字段（剔除 metadata），可识别子类新增列 |

promoted_columns 返回除基类字段外的平铺列，用于决定哪些扩展字段平铺写入、
哪些进 metadata，以及检索时需要回传的字段。

扩展字段落库策略：

| 场景 | 落库位置 | 说明 |
| :--- | :--- | :--- |
| 普通扩展字段 | metadata JSON 列 | 表结构不变，新增字段无需改表 |
| 子类建表新增的列 | 独立物理列 | 建表后自动识别，写入时平铺 |
| 配置提升的列 | 独立物理列 | 可建标量索引，且不再重复写入 metadata |

TODO：过滤表达式的字符串转义需补强注入防护。

### 11.4 关键调用链

离线导入链路（唯一链路），全链路以 pydantic 模型传递并伴随日志埋点：

```text
parse_document  ->  build_semantic_chunks  ->  extract_entities  ->  Embedder.encode  ->  构造 MemoryRecord 校验  ->  store.upsert
```

横切能力在链路两端生效：

- 入参：Settings 已完成校验，LoggerAdapter 已就绪
- 出参：MemoryRecord 与 QueryHit 均为已校验模型
- 扩展：MemoryRecord 的扩展字段经 extra_fields 透传，由存储层决定落库方式

### 11.5 异常与错误处理

| 异常 | 触发场景 | 处理策略 |
| :--- | :--- | :--- |
| ConfigError | 配置缺失或非法 | pydantic 启动即抛 ValidationError，由装配层转 ConfigError |
| OptionalDependencyError | 可选框架或解析库未安装 | 给出安装指引，日志层降级到 stdlib |
| ParserDependencyError | 解析依赖未安装 | 给出安装指引 |
| ParseError | 文档损坏或格式不支持 | 跳过该文件并累计失败清单 |
| SchemaMismatchError | 向量维度或字段不符 | 拒绝写入或建表并提示重建集合 |
| StoreError | 向量库连接或读写失败 | 有限次重试后抛出 |
| EmbeddingError | 向量化服务失败 | 批次重试，仍失败则中断该批次 |

原则：领域异常统一继承基类 RagDataError，便于上层统一捕获；外部依赖异常在边界层转换。

### 11.6 并发与幂等

- 导入幂等：主键由内容哈希决定，重复导入为覆盖而非新增
- 批量并发：解析与向量化可流水线并行，写入按 batch_size 串行以避免触发限流
- 日志并发：LoggerAdapter 的 bind 返回新实例，不做共享状态改写，天然并发安全
- 建表幂等：ensure_collection 检测集合是否存在，存在则仅校验与 load
- 本计划无在线链路，不涉及反馈计数一致性

### 11.7 日志与可观测性

日志以 LoggerAdapter 统一出口，后端可切换：

| 场景 | 建议后端 | 说明 |
| :--- | :--- | :--- |
| 本地开发 | structlog 或 loguru | 彩色与结构化输出，调试友好 |
| 生产与容器 | structlog 加 JSON | 便于采集与检索 |
| 最小依赖或库内嵌 | stdlib logging | 零额外依赖 |

关键指标：

| 维度 | 指标 | 用途 |
| :--- | :--- | :--- |
| 导入 | 解析文件数、切块数、写入条数、失败数 | 导入质量监控 |
| 向量化 | 编码批次数、平均耗时、维度 | 性能与一致性监控 |
| 建表 | 集合名、字段列表、提升列 | 表结构审计 |

约定：日志命名空间按模块层级组织，如 rag_data.ingestion.pipeline；关键路径 bind memory 主键、source_path、user_id；异常统一用 exception 记录堆栈。

### 11.8 单元测试与代码映射

| 测试文件 | 覆盖对象 | 关键用例 |
| :--- | :--- | :--- |
| test_models.py | pydantic 模型 | 字段约束、取值边界、id 主键、已移除字段缺席、extra 扩展、子类继承校验 |
| test_config.py | Settings | 默认值、环境变量覆盖、代码硬编码优先、分区名校验、Literal 与区间校验 |
| test_logging.py | LoggerAdapter | 三后端参数化输出、bind 与 context、auto 探测与回退 |
| test_chunking.py | build_semantic_chunks | 空文本、单句、恰好等于上限、跨块重叠、超长单句 |
| test_entities.py | extract_entities | 去重、大小写归一、空文本 |
| test_embedder.py | Embedder | 批次切分、维度校验、假模型注入 |
| test_embedding_provider.py | 嵌入模型注册表 | 内置 provider、继承即注册、配置解析、API Key 来源、请求构造、响应解析、错误包装、配置驱动切换、批量上限 |
| test_schema.py | Schema 与编解码 | 字段与索引、主键为 id、metadata 划分、存储行往返、提升列、建表字段生成 |
| test_store_contract.py | BaseVectorStore | 以 InMemoryVectorStore 跑接口契约、按 id 幂等、filters 过滤基类与扩展字段 |
| test_pipeline.py | IngestionPipeline | 端到端导入、幂等、分批切分 |
| test_public_api.py | 公共入口 | 版本、__all__ 可解析、核心符号可从包根获取、端到端 |
| test_record_class.py | 记录类路径解析 | 配置分区、点分路径解析、非法路径报错、配置驱动的写入与读回 |
| test_facade.py | 流程门面 | 各流程装配、一键调用、上下文管理、依赖注入 |
| test_store_registry.py | 存储后端注册表 | 内置后端、继承即注册、重复名覆盖、路径与类对象注册、未知后端报错、配置驱动切换 |
| test_milvus_mapping.py | Milvus 建表与映射 | 表结构生成、维度注入、数组规范化、提升列、索引参数、建表幂等、维度校验、分批写入 |

### 11.9 公共入口（Public API）

对外只暴露一个文件，即包根的 __init__.py，内部子包路径不外泄。

```python
import rag_data
from rag_data import Settings, RagData, InMemoryVectorStore
```

| 分组 | 导出符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、StorageSettings、ModelSettings、ChunkingSettings、EmbeddingSettings、NLPSettings、LoggingSettings、load_env_overrides |
| 数据模型 | DocumentChunk、MemoryRecord、QueryHit、resolve_record_class、DEFAULT_RECORD_CLASS_PATH |
| 导入管道 | IngestionPipeline、parse_document、build_semantic_chunks、extract_entities |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 存储 | BaseVectorStore、InMemoryVectorStore、register_store、register_backend、available_backends、is_registered、resolve_store、create_store |
| 日志适配 | LoggerAdapter、configure_logging、get_logger |
| 流程门面 | RagData、init_collection、ingest、build_settings、build_logger、build_store、build_nlp、build_embedder、build_pipeline、build_record_class、DEFAULT_TOP_N |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、SchemaMismatchError、StoreError、EmbeddingError |

设计约定：

1. __init__.py 只做统一导出，不含业务逻辑
2. __all__ 显式登记公共 API，内部符号不外泄
3. 依赖可选第三方库的实现（如 MilvusVectorStore）不纳入顶层导出，按需从子包引入，确保未安装可选依赖时仍可 import rag_data
4. 公共入口引入的模块仅依赖标准库或核心依赖，保证导入轻量

### 11.10 流程门面（facade.py）

把各流程的调用方法集中封装于一处，调用方无需逐个导入子模块。

```python
from rag_data import RagData

app = RagData.create()      # 装配配置、日志、存储、向量化、管道
app.init_collection()       # 流程：建集合
app.ingest(paths, user_id)  # 流程：批量导入
app.ingest_file(path)       # 流程：导入单文件
app.query(vector, top_n)    # 流程：按向量检索
app.close()                 # 流程：释放资源
```

分流程的装配方法：

| 方法 | 职责 |
| :--- | :--- |
| build_settings(source) | 加载配置，支持整体字典与分区硬编码 |
| build_logger(settings) | 装配日志适配层 |
| build_record_class(settings) | 解析 settings.models.record_class |
| build_store(settings, logger, record_class) | 按 storage.backend 装配向量库，milvus 惰性导入 |
| build_nlp(settings, logger) | 加载 spaCy 句柄，缺失或失败时降级为 None |
| build_embedder(settings, logger, model) | 装配向量化组件，provider 由配置决定 |
| build_pipeline(...) | 装配导入管道 |
| init_collection(source) | 一键建表 |
| ingest(paths, source, user_id) | 一键导入 |

设计约定：

1. 门面只做流程装配与转发，不重复实现业务逻辑，真实逻辑仍在各子包
2. RagData 支持依赖注入，store、embedder、logger、nlp、record_class 均可外部替换，便于测试
3. 支持上下文管理，with 退出时自动 close
4. 可选依赖（pymilvus、spaCy）一律惰性导入并降级，保证门面在最小依赖环境可用
5. user_id 与 filters 由门面透传到管道与存储，支持按扩展字段过滤

## 12. 编码规范与工程约定

1. 各模块首行统一使用 from __future__ import annotations，兼容 Python 3.8 的类型语法
2. 数据模型与配置一律使用 pydantic v2 写法：ConfigDict、field_validator、model_validator
3. 业务代码只依赖 LoggerAdapter 抽象，禁止直接 import loguru、structlog 或标准库 logging
4. 日志轮转与级别由配置驱动，不在业务代码里硬编码 handler
5. 外部依赖 pymilvus、spacy、unstructured、loguru、structlog 惰性导入，缩短冷启动并便于注入测试替身
6. 公共接口均带类型注解，交由 mypy 配合 pydantic 插件检查
7. 使用 ruff 统一风格，固定行宽与导入顺序
8. 不新增模块级可变全局状态，依赖一律通过构造注入
9. 对外 API 一律经包根 __init__.py 导出并在 __all__ 中登记，子包内部符号不作为公共契约
