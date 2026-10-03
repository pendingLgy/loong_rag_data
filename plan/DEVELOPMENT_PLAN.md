# rag-data 代码开发计划（向量化与数据导入）

依据文档：rag_artifact.md（历史文本数据导入与长期记忆构建架构设计）

修订说明：

1. 项目范围收窄，移除在线记忆召回模块，仅保留向量化与离线数据导入
2. 数据模型与配置统一采用 pydantic 做类型校验
3. 新增日志适配层，支持多种日志框架切换
4. 配置改为按模块分区，默认读取环境变量，并支持代码硬编码
5. 新增流程门面，各流程调用集中封装
6. 记录类支持通过配置切换，便于继承扩展字段
7. 存储层整体移除（Milvus 实现、后端注册表、向量库配置与相关测试与文档），当前仓库聚焦向量化；向量落库流程待存储层重新设计后再补全

> **实施状态**：存储层已落地后又被整体移除，M4 与 §11.3.5 等存储相关设计标注为（已移除），仅作历史参考；
> 当前代码聚焦「解析 → 切块 → 实体 → 向量化」，与本文叙述不一致处以后者为准。

## 0. 项目现状盘点

- pyproject.toml：hatchling 构建，源码根为 src 下的 rag_data 包，Python 需 3.8 以上；已声明核心与可选依赖、mypy 的 pydantic 插件、pytest 与 coverage 配置；hatch 环境分 default 与 rag_data_dev 两套，依赖由 uv 处理
- src 下的 rag_data：核心包已生成，含 config、models、exceptions、facade、ingestion、embedding、logging
- ingestion：解析、两阶段切块、实体抽取与导入编排
- embedding：向量化封装，含批次与维度校验
- storage：已整体移除；Milvus 实现、后端注册表与存储配置一并删除，落库流程待重新设计
- logging：日志适配层，含 stdlib、loguru、structlog 三后端
- facade：流程门面，一站式装配与调用
- tests：单元测试覆盖模型、配置、日志、切块、实体、文档解析（含 EPUB）、向量化、导入管道、字符串向量化、公共入口与流程门面
- README.md：项目说明、公共 API、流程门面、数据模型、配置与开发
- plan：本设计与开发计划
- rag_artifact.md：架构设计源文档
- 依赖现状：核心依赖 pydantic 与 pydantic-settings；可选依赖含 spacy、unstructured、loguru、structlog 等
- 待补全：涉及外部服务的复杂逻辑以 TODO 标注，主要集中在文档解析（PDF 与 Word）与真实模型加载
- 当前目录不是 git 仓库

结论：骨架、类型模型、日志适配层、流程门面与向量化链路已就绪，单元测试可运行；文档解析与真实模型加载待人工补全，向量落库流程待重新设计。

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
- 模块二：向量库 Schema（存储层已移除，待重新设计）
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
| 文档解析 | Unstructured、MinerU；EPUB 用 ebooklib 与 BeautifulSoup | 需带 Markdown 层级输出；EPUB 解析已落地 |
| 分句与实体 | spaCy、zh_core_web_sm | 停用 ner 与 parser，启用 sentencizer |
| 向量化 | 外部 Embedding 服务（openai、qwen）或自建 provider | 默认 openai；维度需与 embedding.dim 对齐 |
| 日志框架 | stdlib logging、loguru、structlog | 由适配层统一封装，可切换 |
| 构建与依赖 | hatch 管理构建与运行环境，uv 解析与安装依赖 | 构建后端为 hatchling |
| 测试 | pytest、pytest-cov | 已配置 coverage |
| 质量 | mypy、ruff | 配合 pydantic 插件做类型检查 |

依赖分层：pydantic 与 pydantic-settings 为核心必装；loguru、structlog、sentence-transformers 均为可选依赖，按需安装。

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
│     ├─ registry.py         通用注册表（嵌入 provider 复用）
│     ├─ models              pydantic 数据模型
│     │  ├─ __init__.py      公共导出，保持 rag_data.models 入口不变
│     │  └─ document.py      DocumentChunk
│     ├─ exceptions.py       领域异常基类
│     ├─ facade.py           流程门面，封装各流程调用方法
│     ├─ ingestion           解析、实体抽取与编排
│     │  ├─ entities.py      NER 实体抽取与归一化
│     │  └─ pipeline.py      离线导入编排
│     ├─ parsers             DocumentParser 基类 + 各格式子类
│     │  ├─ markdown.py      .md、.markdown
│     │  ├─ txt.py           .txt
│     │  ├─ epub.py          .epub
│     │  ├─ pdf.py           .pdf 待实现
│     │  └─ word.py          .docx、.doc 待实现
│     ├─ embedding           向量化接口、provider 注册表与 openai、qwen 实现
│     │  └─ embedder.py      Embedding 封装
│     └─ logging             日志适配层
│        ├─ base.py          LoggerAdapter 抽象
│        ├─ stdlib_adapter.py
│        ├─ loguru_adapter.py
│        ├─ structlog_adapter.py
│        └─ factory.py       get_logger 与自动探测
└─ tests
   ├─ conftest.py
   ├─ test_base.py
   └─ test_epub.py
```

说明：日志适配层目录名为 logging，位于 rag_data 包内，采用绝对导入避免与标准库 logging 冲突；如仍担心歧义，可改名为 observability。

## 4. 里程碑与任务分解

### M0 项目骨架与环境（0.5 人天）

任务：

- 创建 src 下的 rag_data 包及 __about__.py，定义版本号供 hatch 读取
- 创建各子包 __init__.py
- 在 pyproject.toml 声明核心依赖 pydantic、pydantic-settings 与可选依赖 loguru、structlog 等
- 声明 dev 依赖 pytest、mypy、ruff 以及 pydantic 的 mypy 插件
- 配置 hatch 环境：default 含 dev 工具，rag_data_dev 叠加全部可选依赖，并定义 test、types、check、format 脚本
- 依赖安装与锁定交由 uv 处理，uv 直接读取 pyproject.toml 的依赖声明
- 创建 README.md

交付：可本地以可编辑方式安装并可导入 rag_data
验收：mypy 对空包无报错

### M1 类型模型与配置（1 人天）

任务：

- models 包：以 pydantic BaseModel 定义 DocumentChunk，切块元数据与向量同体
- 为关键字段加约束：chunk_id 与 text 非空、created_at 非负；向量维度由 Embedder 校验
- config.py：以 pydantic-settings BaseSettings 定义 Settings，按模块分区，含 env_prefix、嵌套分隔符、.env 加载
- 对互相约束的字段使用 model_validator，对单字段使用 field_validator
- 配置来源优先级：初始化参数 大于 环境变量 大于 .env 大于 默认值

交付：models 包、config.py
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

- 在 parsers/base.py 的 DocumentParser 提供句柄加载，各解析器子类自持 parse 与 chunk 规则
- 复用文档示例逻辑，补齐边界处理：空文本、超长单句、overlap 大于已累积句数
- 保持纯函数、无 IO，便于单元测试
- 以 LoggerAdapter 记录超长单句等告警，不直接依赖某个日志框架

交付：parsers/base.py 的句柄加载 + parsers/<format>.py 的子类 parse、chunk
验收：单测覆盖 空文本、单句、恰好等于上限、需 overlap 的跨块、超长单句

### M4 向量库 Schema（1 人天，已移除）

对应文档 模块二，字段按向量化范围裁剪；本里程碑对应的存储层已整体移除，下列任务仅作历史记录。

任务：

- 记录类定义字段：id、text_payload、vector、entities、created_at
- milvus_store.MilvusRecord 声明建表字段并定义索引：主键、标量索引（entities、created_at）、向量索引 HNSW、度量 COSINE
- 记录类提供存储行编解码：记录与行之间的双向映射，扩展字段按提升列平铺
- milvus_store 提供建表生成器：向量索引参数、标量索引配置
- 存储接口（无基类）：ensure_collection、upsert、close
- milvus_store.py 实现，建表前用 Settings.vector_dim 与模型维度做校验

交付：models 包、storage 包
验收：本地 Milvus 可建表并写入读取；接口用内存 Fake 实现做契约测试

### M5 向量化模块与导入管道（2 人天）

任务：

- embedding：provider 注册表与 openai、qwen 实现，批量 embedding，输出维度需与 storage.vector_dim 一致
- ingestion 的 entities.py：spaCy NER 抽取实体列表
- parsers/：PDF、Word、Markdown 转带层级文本，清理页眉页脚与断词
- ingestion 的 pipeline.py：解析 到 切块 到 实体抽取 到 向量化 到 upsert，支持分批与幂等
- 全链路以 pydantic 模型传递数据，构造 MilvusRecord 时自动校验维度与字段
- 记录类可由配置指定，写入与读回共用同一个类

交付：embedder.py、entities.py、parsers/、pipeline.py
验收：样例文档端到端跑通，写入条数正确，重复导入幂等

### M6 公共入口与流程门面（1 人天）

任务：

- 包根 __init__.py 统一导出全部公共 API，并登记 __all__
- facade.py 封装各流程调用：配置、日志、句柄、向量化、管道
- RagData 提供 ingest、ingest_file、vectorize、vectorize_text，支持依赖注入

交付：__init__.py、facade.py
验收：外部仅导入 rag_data 即可完成全部流程；__all__ 全部可解析

### M7 测试与文档（1 人天）

任务：

- 补齐测试与覆盖率、README、本计划
- 通过 mypy 与 ruff

验收：pytest 全绿且覆盖核心模块，mypy 无错误

## 5. 关键接口契约

- config.Settings：pydantic-settings 的 BaseSettings，按模块分区，实例化即完成校验
- models.DocumentChunk：pydantic BaseModel
- logging.factory.configure_logging（settings）返回 LoggerAdapter
- logging.factory.get_logger（name）返回 LoggerAdapter
- logging.base.LoggerAdapter：debug、info、warning、error、exception、bind、context
- parsers.parse_document（path）返回纯文本字符串
- parsers.chunk_document（path，text，max_chars 默认 250，overlap_sents 默认 1，nlp 可选，logger 可选）按扩展名派发到该格式的 chunk
- ingestion.entities.extract_entities（text，nlp 可选）返回实体列表
- embedding.embedder.Embedder.encode（texts）返回向量列表
- ingestion.pipeline.IngestionPipeline：ingest_file（path，user_id 可选）、run（paths，user_id 可选）返回携带向量的切块列表
- facade.RagData：ingest、ingest_file、vectorize、vectorize_text，依赖均可注入
- 一键入口：ingest（paths，source，user_id）、vectorize（texts，source）

## 6. 配置与默认值

配置按模块分区，与 JSON 结构一一对应；模型为 pydantic 嵌套模型，环境变量与 JSON 均可覆盖。

| 模块 | 字段 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| chunking | max_chars | 250 | 切块字符上限，区间 150 至 300 |
| chunking | overlap_sents | 1 | 切块重叠句数 |
| embedding | provider | openai | 嵌入模型注册表中的键 |
| embedding | model | 空字符串 | Embedding 模型标识，留空用 provider 默认模型 |
| embedding | batch_size | 128 | 批量编码大小 |
| embedding | dim | 1024 | 期望输出维度，向量化结果据此校验 |
| embedding | api_key | 空字符串 | 留空时读取 provider 约定的环境变量 |
| embedding | base_url | 空字符串 | 留空时用 provider 默认地址 |
| embedding | timeout | 60.0 | 单次请求超时秒数 |
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
- 端到端：小样例文档经假嵌入模型编码到向量产出断言
- 覆盖率目标：核心模块 85 以上

## 8. 依赖与风险

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| pydantic v1 与 v2 语法差异 | 配置与校验写法不兼容 | 锁定 v2，统一使用 ConfigDict、field_validator、model_validator |
| 日志框架为可选依赖 | 未安装时导入失败 | 适配器惰性导入，缺失时回退 stdlib 并给出提示 |
| 子包名 logging 与标准库同名 | 误导入歧义 | 包内一律绝对导入，必要时改名 observability |
| spaCy 中文模型体积与安装 | 首次环境搭建慢 | 模型下载脚本化与 CI 缓存 |
| Unstructured 与 MinerU 版本差异 | 解析结果不稳定 | 锁版本并加解析回归样例 |
| Embedding 模型与维度不一致 | 下游按错误维度使用 | 维度写入 embedding.dim，Embedder 编码后逐条校验 |
| Python 3.8 运行期类型语法 | list 下标等新语法在 3.8 报错 | 文件首行加 from __future__ import annotations |

## 9. 建议排期（合计约 7.5 人天）

| 里程碑 | 内容 | 人天 |
| :--- | :--- | :--- |
| M0 | 骨架与环境 | 0.5 |
| M1 | 类型模型与配置（pydantic） | 1.0 |
| M2 | 日志适配层 | 1.0 |
| M3 | 两阶段切块 | 1.0 |
| M4 | 向量库 Schema（已移除） | 1.0 |
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
parsers、entities、embedder   领域服务层


config.Settings（pydantic-settings）   横切层：类型安全的只读配置
models（pydantic BaseModel）          横切层：数据契约与校验
logging（LoggerAdapter）              横切层：日志抽象，屏蔽具体框架
```

依赖原则：

1. 门面通过装配获得依赖，领域服务不直接依赖具体实现
2. 具体实现通过构造注入，即依赖反转，便于替换为测试替身
3. 领域服务层只依赖 LoggerAdapter 抽象，不 import loguru 或 structlog
4. config 为不可变且已校验的配置对象，逐层向下传递，禁止模块级全局可变状态
5. 所有跨层数据以 pydantic 模型承载，边界处自动校验

### 11.2 数据模型（pydantic）

```python
class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    entities: List[str] = Field(default_factory=list)
    created_at: float = Field(ge=0.0)
    vector: List[float] = Field(default_factory=list)   # 向量化后填入
```

字段约定：

| 模型 | 字段 |
| :--- | :--- |
| DocumentChunk | chunk_id、user_id、source_path、text、entities、created_at、vector |

设计要点：

1. 切块与向量同体，一个模型贯穿「切块 → 向量化」，不再区分中间态与记录类
2. model_config 为 frozen，实例不可变；向量化用 model_copy(update=...) 回填 vector
3. vector 默认空表示尚未向量化，维度由 Embedder 按 embedding.dim 校验
4. chunk_id 由内容哈希生成，重复导入得到相同 id，便于后续落库保持幂等

### 11.3 各模块职责与接口签名

#### 11.3.1 config.py（pydantic-settings，按模块分区）

职责：以嵌套模型承载各模块配置，实例化即完成校验；默认读取环境变量，也支持代码硬编码。

```python
class ChunkingSettings(BaseModel):
    max_chars: int = Field(default=250, ge=150, le=300)
    overlap_sents: int = Field(default=1, ge=0)


class EmbeddingSettings(BaseModel):
    provider: str = "openai"
    model: str = ""
    batch_size: int = Field(default=128, gt=0)
    dim: int = Field(default=1024, gt=0)


ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        extra="ignore",
    )

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def load(cls, source=None, *, use_env=True, **overrides) -> Settings: ...


# 分区名到分区模型的映射，用于在关闭环境变量时补齐字段默认值
SECTION_MODEL_CLASSES: Dict[str, Type[BaseModel]] = { ... }
```

加载入口与优先级：

| 入口 | 说明 |
| :--- | :--- |
| Settings() | 默认读取环境变量（含 .env），未提供时用字段默认值 |
| Settings.load() | 同上，显式表达按环境变量加载 |
| Settings.load(embedding={...}) | 代码硬编码分区配置，逐字段覆盖环境变量 |
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
| _with_defaults(layered) | 用分区默认值补齐字段，供 use_env=False 屏蔽环境变量 |

设计要点：

1. 环境变量作为基线，代码硬编码在其上逐字段覆盖，即环境提供默认、显式配置最终生效
2. 每个分区模型 extra 为 forbid，分区内拼错字段会在加载期即报错
3. 根模型 extra 为 ignore：RAG_ 命名空间可能混入无关变量，避免误报
4. 代码硬编码中的分区名由 _validate_sections 校验，写错即抛 ConfigError 并列出可用分区
5. 环境变量支持嵌套覆盖，如 RAG_EMBEDDING__DIM 覆盖 embedding.dim
6. 取值交由 pydantic 转换，JSON 对象与数组按 JSON 解析
7. logging 分区中 json 为保留名，字段名为 json_output 并设置别名 json
8. use_env 为 False 时先以分区默认值补齐全部字段再作为构造实参，显式实参优先级最高，环境变量无从渗入，不依赖 model_validate 的版本行为
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

##### parsers/（基类与子类）

目录：__init__.py（派发）、markdown.py、txt.py、epub.py、pdf.py、word.py。

- 职责：按扩展名派发到各 DocumentParser 子类，输出纯文本
- 接口：parse_document(path: str, logger=None) -> str；各子类统一暴露 SUFFIXES 并实现 parse、chunk
- 派发：__init__ 汇总各模块 SUFFIXES，得到「扩展名 → parse」派发表；新增格式只需实现约定并登记
- 后缀分组：MARKDOWN 为 .md 与 .markdown；TXT 为 .txt；EPUB 为 .epub；PDF 为 .pdf；WORD 为 .docx 与 .doc；TEXT 为 MARKDOWN 并 TXT，BINARY 为 PDF 并 WORD
- 依赖：unstructured 或 MinerU（PDF 与 Word）、ebooklib 与 BeautifulSoup（EPUB）；一律惰性导入，缺失时抛 ParserDependencyError
- 已支持：Markdown、TXT 直读；EPUB 按 spine 顺序提取正文，并排除目录页与样式等非正文项
- EPUB 另有 chunk(text, max_chars, overlap_sents, nlp) 与其它格式同签名，规则自持
- TODO：PDF 与 Word 的解析尚未实现；Markdown 目前保留原始标记

##### parsers/base.py 的句柄加载

- 职责：spaCy 句柄的按需加载与缓存，供装配层（facade）与需要严格句柄的格式共用
- 接口：DocumentParser.load_nlp(model_name=DEFAULT_MODEL, logger=None) 与 clear_cache()
- 设计：未装 spaCy 或模型缺失时返回 None 并告警；自动补 sentencizer 以支持句级切分；成功与失败均缓存，clear_cache() 可重建


##### parsers/base.py 与各格式 chunk

- 职责：各 parsers/<format>.py 的子类自持 parse 与 chunk 规则；基类只提供句柄加载
- 接口：各格式 chunk(text, max_chars, overlap_sents, nlp, logger) 与 parse(path, logger)
- 设计：切块规则随格式不同，由各子类自行实现；nlp 由装配层注入，传 None 时按各子类规则处理

##### entities.py

- 职责：基于 spaCy NER 抽取实体，供切块元数据与后续过滤使用
- 接口：def extract_entities(text, nlp=None, *, labels=None) -> List[str]
- 设计：句柄由调用方注入，未注入（None）则不抽取；归一化（去空白、剥包裹引号括号）、过滤纯标点、大小写不敏感去重，并可选按实体类型白名单过滤

##### pipeline.py

```python
class IngestionPipeline:
    def __init__(self, embedder, settings, logger, nlp=None): ...

    def run(self, paths: List[str], user_id: Optional[str] = None) -> List[DocumentChunk]: ...
    def ingest_file(self, path: str, user_id: Optional[str] = None) -> List[DocumentChunk]: ...
    def _embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]: ...

    @staticmethod
    def _content_hash(user_id: str, source_path: str, text: str) -> str: ...
```

设计要点：

1. 编排顺序为 解析 到 切块 到 实体抽取 到 向量化 到 回填向量
2. memory 主键由 user_id、source_path 与切块内容做 sha256 生成，重复导入幂等
3. 按 settings.embedding.batch_size 分批编码，降低单次请求压力
4. 向量经 model_copy 回填到 frozen 的 DocumentChunk.vector
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
- qwen：backend 为 qwen，默认 qwen3.7-text-embedding（1024 维），接口为 DashScope 兼容模式，默认端点可按部署环境覆盖，Key 读 DASHSCOPE_API_KEY
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
- 设计：输出维度需与 embedding.dim 一致，不一致抛 EmbeddingError
- 设计：空输入直接返回，不触发 provider 装配


#### 11.3.5 storage 子包（已移除）

> 存储层已整体移除，本小节及其下各接口签名仅作历史参考。

##### 记录类与 Milvus 索引定义（已移除）

职责：记录类提供存储行编解码，milvus_store 定义索引与建表参数。

| 字段 | 类型 | 索引类型 | 说明 |
| :--- | :--- | :--- | :--- |
| id | VARCHAR | Primary Key | 唯一标识，内容哈希 |
| text_payload | VARCHAR | None | 原始记忆文本块 |
| vector | FLOAT_VECTOR | HNSW 或 IVFLAT | 语义向量，维度取自配置 |
| entities | ARRAY(VARCHAR) | Scalar Index | 实体列表，用于过滤；须声明 max_capacity 与元素级 max_length |
| created_at | DOUBLE | Scalar Index | 创建或录入时间戳 |

已移除字段：memory_id（改名为 id）、user_id、last_accessed_at、recall_count。

导出符号（记录类 / milvus_store）：

| 名称 | 含义 |
| :--- | :--- |
| milvus_store.BASE_FIELD_NAMES | 基础字段名，之外的一律视为扩展字段 |
| milvus_store.SCALAR_INDEX_TYPES | 各字段的标量索引类型；未列出的字段不建标量索引 |
| milvus_store.VECTOR_INDEX_PARAMS | 各索引类型的默认构建参数；未登记的类型由 Milvus 采用自身默认值 |

记录类提供的存储行编解码：

| 函数 | 作用 |
| :--- | :--- |
| record.to_storage_row() | 基础字段平铺，未声明字段不落库 |
| Cls.from_storage_row(row) | 存储行还原为记录 |

设计约定：

1. 扩展字段默认不落库；提升为独立列后才写入，新增字段无需改表结构
2. 提升为独立列的扩展字段以物理列存储，写入时平铺
3. 编解码归属 MilvusRecord 自身，存储实现按同一套行为完成写入与还原
4. 测试替身保存的是存储行而非 pydantic 对象，使单元测试真实覆盖序列化与还原

##### store_registry.py（存储后端注册表）

职责：以注册表解耦后端选择与装配代码，显式登记，配置只写后端名。

```python
BUILTIN_BACKENDS: Dict[str, str] = {
    "milvus": "rag_data.storage.milvus_store.MilvusVectorStore",
}
Ref = Union[str, type]

def register_store(name: str, ref: Ref) -> None: ...      # ref 为类对象或点分路径
def register_backend(name: str): ...                      # 装饰器
def available_backends() -> List[str]: ...
def is_registered(name: str) -> bool: ...
def resolve_store(name: str) -> Type[Any]: ...
def load_store_modules(refs: Sequence[str]) -> None: ...  # 导入用户模块，导入即登记
def create_store(name, settings, logger, **kwargs) -> Any: ...
```

设计约定：

1. 后端以点分路径或类对象显式登记，实现类无需继承基类
2. 类对象直接入表，避免本地类因 qualname 无法按路径导入；字符串则惰性导入
3. 内置后端以点分路径注册，实现惰性导入，未装 pymilvus 时仍可正常导入 rag_data
4. 未注册的后端名抛 ConfigError，并在信息中列出可用后端
5. 创建实例统一以关键字传入 settings 与 logger，各实现保持一致签名
6. 用户可以模块形式扩充后端：storage.store_modules 列出点分路径或 .py 文件，
   模块在导入期调用 register_store 登记，create_store 装配前按需导入且幂等


##### 存储接口（无基类）

存储实现为独立类，无需继承任何基类，只需约定一致的方法签名：

```python
class MilvusVectorStore:          # Milvus 实现

    def ensure_collection(self) -> None: ...
    def upsert(self, records) -> int: ...
    def close(self) -> None: ...
```

接口契约：

1. upsert 按 id 幂等覆盖，返回实际写入条数；扩展字段一并落库
2. 入参与出参均为 pydantic 模型，避免裸字典漂移
3. 新增后端只需实现类并用 register_store 登记，或经 storage.store_modules 提供模块；装配代码不变

已移除：查询（query）能力与对应基类。

##### models.build_collection_schema（建表入口）

建表权由记录类掌握，MilvusRecord 默认硬编码声明的字段：

```python
class MilvusRecord(BaseModel):
    @classmethod
    def build_collection_schema(cls, pymilvus: Any, vector_dim: int) -> Any:
        """按声明的字段硬编码创建集合表结构。"""
        ...
        return pymilvus.CollectionSchema(fields=fields, description=...)
```

子类覆盖即可自定义表结构：

```python
class TenantRecord(MilvusRecord):
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
2. 硬编码字段与 milvus_store.BASE_FIELD_NAMES 一一对应，一致性由测试锁定
3. 维度非法时抛 ConfigError，避免生成无效表结构
4. Milvus 侧仅负责调用记录类建表；索引参数与标量索引表定义在 milvus_store

##### milvus_store.py

```python
class MilvusVectorStore:
    def __init__(self, settings, logger, alias="rag_data"): ...

    def column_names(self) -> List[str]: ...
    def vector_index_params(self) -> Dict[str, Any]: ...
    def scalar_index_specs(self) -> List[Tuple[str, Dict[str, Any]]]: ...

    def ensure_collection(self) -> None: ...
    def upsert(self, records: List[MilvusRecord]) -> int: ...
```

建表流程（已实现）：

1. 惰性连接：首次调用时按 settings.storage.milvus_uri 与 milvus_db 连接并缓存，库名随连接生效
2. 集合不存在：调用 MilvusRecord.build_collection_schema 生成 schema，再建表、建向量索引与标量索引，最后 load
3. 集合已存在：跳过建表，直接 load
4. 向量索引：index_type 与 metric 取自配置，HNSW 使用 M 与 efConstruction
5. 标量索引：entities 用 INVERTED，created_at 用 STL_SORT
6. 写入：记录转存储行后按 batch_size 分批提交，最后 flush
7. 未安装 pymilvus：抛 OptionalDependencyError 并给出安装指引

列名解析：未建表时为 BASE_FIELD_NAMES，已建表时取集合实际字段。


### 11.4 关键调用链

离线导入链路（唯一链路），全链路以 pydantic 模型传递并伴随日志埋点：

```text
parse_document  ->  chunk_document  ->  extract_entities  ->  Embedder.encode（分批）  ->  回填 vector（model_copy）  ->  DocumentChunk
```

横切能力在链路两端生效：

- 入参：Settings 已完成校验，LoggerAdapter 已就绪
- 出参：DocumentChunk 为已校验模型，vector 已按 embedding.dim 校验

### 11.5 异常与错误处理

| 异常 | 触发场景 | 处理策略 |
| :--- | :--- | :--- |
| ConfigError | 配置缺失或非法 | pydantic 启动即抛 ValidationError，由装配层转 ConfigError |
| OptionalDependencyError | 可选框架或解析库未安装 | 给出安装指引，日志层降级到 stdlib |
| ParserDependencyError | 解析依赖未安装 | 给出安装指引 |
| ParseError | 文档损坏或格式不支持 | 跳过该文件并累计失败清单 |
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
| test_models.py | pydantic 模型 | DocumentChunk 约束与冻结、向量默认值与回填 |
| test_config.py | Settings | 默认值、环境变量覆盖、代码硬编码优先、分区名校验、Literal 与区间校验 |
| test_logging.py | LoggerAdapter | 三后端参数化输出、bind 与 context、auto 探测与回退 |
| test_entities.py | extract_entities | 去重、大小写归一、空文本、默认句柄加载、类型过滤、归一化与过滤规则 |
| test_base.py | DocumentParser | 抽象契约与文件读取、spaCy 句柄加载与缓存、失败降级、clear_cache |
| test_epub.py | EPUB 解析 | 后缀、依赖缺失报错、章节顺序、目录页排除、非文档项排除、清洗、损坏容器报错、chunk 切块（分组、重叠、空句、模型缺失）|
| test_embedder.py | Embedder | 批次切分、维度校验、假模型注入 |
| test_embedding_provider.py | 嵌入模型注册表 | 内置 provider、继承即注册、配置解析、API Key 来源、请求构造、响应解析、错误包装、配置驱动切换、批量上限 |
| test_pipeline.py | IngestionPipeline | 端到端向量化、幂等、分批切分、缺失文件 |
| test_public_api.py | 公共入口 | 版本、__all__ 可解析、核心符号可从包根获取、端到端 |
| test_facade.py | 流程门面 | 各流程装配、一键调用、依赖注入、字符串向量化 |

### 11.9 公共入口（Public API）

对外只暴露一个文件，即包根的 __init__.py，内部子包路径不外泄。

```python
import rag_data
from rag_data import Settings, RagData
```

| 分组 | 导出符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、ChunkingSettings、EmbeddingSettings、NLPSettings、LoggingSettings、load_env_overrides、ENV_PREFIX、ENV_NESTED_DELIMITER |
| 数据模型 | DocumentChunk |
| 导入管道 | IngestionPipeline、parse_document、extract_entities |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 日志适配 | LoggerAdapter、configure_logging、get_logger |
| 流程门面 | RagData、ingest、vectorize、build_settings、build_logger、build_nlp、build_embedder、build_pipeline |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError |

设计约定：

1. __init__.py 只做统一导出，不含业务逻辑
2. __all__ 显式登记公共 API，内部符号不外泄
3. 依赖可选第三方库的实现不纳入顶层导出，按需从子包引入，确保未安装可选依赖时仍可 import rag_data
4. 公共入口引入的模块仅依赖标准库或核心依赖，保证导入轻量

### 11.10 流程门面（facade.py）

把各流程的调用方法集中封装于一处，调用方无需逐个导入子模块。

```python
from rag_data import RagData

app = RagData.create()      # 装配配置、日志、向量化、管道
app.ingest(paths, user_id)  # 流程：向量化文件
app.ingest_file(path)       # 流程：向量化单文件
app.vectorize(texts)        # 流程：向量化字符串
```

分流程的装配方法：

| 方法 | 职责 |
| :--- | :--- |
| build_settings(source) | 加载配置，支持整体字典与分区硬编码 |
| build_logger(settings) | 装配日志适配层 |
| build_nlp(settings, logger) | 加载 spaCy 句柄，缺失或失败时降级为 None |
| build_embedder(settings, logger, model) | 装配向量化组件，provider 由配置决定 |
| build_pipeline(...) | 装配导入管道 |
| ingest(paths, source, user_id) | 一键向量化文件 |
| vectorize(texts, source) | 一键向量化字符串 |

设计约定：

1. 门面只做流程装配与转发，不重复实现业务逻辑，真实逻辑仍在各子包
2. RagData 支持依赖注入，embedder、logger、nlp 均可外部替换，便于测试
3. 字符串向量化直接返回向量列表，不经切块元数据
4. 可选依赖（spaCy）一律惰性导入并降级，保证门面在最小依赖环境可用

## 12. 编码规范与工程约定

1. 各模块首行统一使用 from __future__ import annotations，兼容 Python 3.8 的类型语法
2. 数据模型与配置一律使用 pydantic v2 写法：ConfigDict、field_validator、model_validator
3. 业务代码只依赖 LoggerAdapter 抽象，禁止直接 import loguru、structlog 或标准库 logging
4. 日志轮转与级别由配置驱动，不在业务代码里硬编码 handler
5. 外部依赖 spacy、unstructured、loguru、structlog 惰性导入，缩短冷启动并便于注入测试替身
6. 公共接口均带类型注解，交由 mypy 配合 pydantic 插件检查
7. 使用 ruff 统一风格，固定行宽与导入顺序
8. 不新增模块级可变全局状态，依赖一律通过构造注入
9. 对外 API 一律经包根 __init__.py 导出并在 __all__ 中登记，子包内部符号不作为公共契约
