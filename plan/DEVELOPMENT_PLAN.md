# rag-data 代码开发计划（向量化与数据导入）

依据文档：rag_artifact.md（历史文本数据导入与长期记忆构建架构设计）

修订说明：

1. 项目范围收窄，移除在线记忆召回模块，仅保留向量化与离线数据导入
2. 数据模型与配置统一采用 pydantic 做类型校验
3. 新增日志适配层，支持多种日志框架切换
4. 配置改为按模块分区，默认读取环境变量，并支持代码硬编码
5. 新增流程门面，各流程调用集中封装
6. 记录类支持通过配置切换，便于继承扩展字段
7. 进一步收窄：移除 models 与 ingestion 子包，仅保留 parsers 与 embedding；config 分区收窄为 parsing、embedding、logging；facade 去掉管道编排与 ingest，仅保留解析、切块与向量化
8. 存储层整体移除（Milvus 实现、后端注册表、向量库配置与相关测试与文档），当前仓库聚焦向量化；落库流程待重新设计，现由手工集成测试直连 Milvus 验证写入与检索
9. 日志适配层重构：以 LoggerFactory 取代 configure_logging 与 get_logger，后端名改为 stdliblog、structlog、loguru；日志行含完整源文件路径、方法名与行号，时区可配
10. 嵌入实现改用官方 openai SDK，取代标准库 urllib；新增 openai 与 milvus 可选依赖，spacy 升到 3.8 以上
11. 句柄默认模型改为 zh_core_web_md，加载时关闭 tagger、ner 与 attribute_ruler
12. parsing 分区新增 safe_max_chars，max_chars 默认值改为 500，切块长度由解析器构造参数注入

> **实施状态**：本文记录设计意图与演进历史，代码为当前事实，不一致处一律以代码为准。
> 已移除：models、ingestion（entities、pipeline）、storage 三个模块，故 §11.2 与 §11.3.3 的 entities、pipeline 小节仅作历史参考。
> 当前仓库为「解析 → 切块 → 向量化」，公共导出 41 个符号，测试为 130 项单元测试加 7 项手工集成测试。

## 0. 项目现状盘点

- pyproject.toml：hatchling 构建，源码根为 src 下的 rag_data 包，Python 需 3.8 以上；已声明核心与可选依赖、mypy 的 pydantic 插件、pytest 与 coverage 配置；hatch 环境分 default 与 rag_data_dev 两套，依赖由 uv 处理
- src 下的 rag_data：含 config、exceptions、registry、facade、parsers、embedding、logging
- parsers：DocumentParser 基类加各格式子类（markdown、txt、epub、pdf、word），按扩展名构造并派发 parse_document 与 chunk_document
- embedding：向量化封装，provider 注册表与 openai、qwen 实现，含批次与维度校验
- logging：LoggerFactory 全局装配，stdliblog、structlog、loguru 三个适配器，按配置切换
- facade：流程门面，一站式装配与调用
- models、ingestion、storage：均已移除，落库流程待重新设计
- tests：单元测试覆盖配置、日志、解析与切块、向量化、流程门面与解析派发；另有手工集成测试直连 Qwen 与 Milvus
- README.md：项目说明、公共 API、流程门面、嵌入模型注册、配置与开发
- plan：本设计与开发计划
- rag_artifact.md：架构设计源文档
- 依赖现状：核心依赖 pydantic 与 pydantic-settings；可选依赖含 spacy、unstructured、pdfplumber、EbookLib、python-docx、openai、pymilvus、loguru、structlog 等
- 待补全：涉及外部服务的复杂逻辑以 TODO 标注，主要集中在版面级解析（PDF 双栏、Word 表格）与本地模型 provider
- 当前目录是 git 仓库，历史提交可追溯存储层的落地与移除

结论：骨架、类型模型、日志适配层、流程门面与向量化链路已就绪；解析与切块覆盖五种格式，向量落库流程待重新设计。

## 1. 目标与范围

### 1.1 目标

实现离线批量导入与向量化管道（当前已落地范围以括注为准）：

1. 类型模型：以 pydantic 定义并校验配置，边界处拒绝脏数据（models 已移除）
2. 文档解析：将 TXT、Markdown、EPUB、PDF、Word 转为纯文本（已落地五种格式）
3. 语义切块：各格式自持规则，递归切分并按四分之一重叠前缀衔接（已落地）
4. 实体抽取：抽取实体并写入标量字段（已随 ingestion 移除）
5. 向量化：批量将切块文本编码为稠密向量（已落地）
6. 批量写入：将向量与元数据写入向量数据库（待重新设计；现由手工集成测试直连 Milvus 验证）
7. 日志适配：以统一接口适配多种日志框架，业务代码与框架解耦（已落地，LoggerFactory 装配）
8. 流程门面：把各流程调用集中封装，外部调用无需逐个导入子模块（已落地）

### 1.2 范围内

- 模块一：批量切块 Batch Chunking（已落地）
- 向量化模块：Embedding 封装与批量编码（已落地）
- 配置层：pydantic-settings 分区配置（已落地）
- 日志适配层：stdliblog、structlog、loguru 多后端（已落地）
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
| 文档解析 | unstructured 与各格式专用库：EPUB 用 EbookLib 与 BeautifulSoup，PDF 用 pdfplumber，Word 用 python-docx | 五种格式均已落地 |
| 分句 | spaCy、zh_core_web_md | 加载时停用 tagger、ner 与 attribute_ruler，按需补 sentencizer |
| 向量化 | 官方 openai SDK，内置 openai 与 qwen provider 或自建 provider | 默认 openai；维度需与 embedding.dim 对齐 |
| 日志框架 | stdlib logging、structlog、loguru | 由 LoggerFactory 统一装配，可切换 |
| 构建与依赖 | hatch 管理构建与运行环境，uv 解析与安装依赖 | 构建后端为 hatchling |
| 测试 | pytest、pytest-cov | 已配置 coverage |
| 质量 | mypy、ruff | 配合 pydantic 插件做类型检查 |

依赖分层：pydantic 与 pydantic-settings 为核心必装；spacy、解析库、openai SDK、pymilvus、loguru、structlog 均为可选依赖，按需安装（openai 与 qwen 两个内置 provider 依赖 openai extra）。

## 3. 目录结构

与现有 src 布局一致（models、ingestion 已移除）：

```text
rag-data
├─ pyproject.toml
├─ README.md
├─ plan
│  └─ DEVELOPMENT_PLAN.md
├─ .vcl
│  └─ struct.md            项目结构说明
├─ rag_artifact.md
├─ src
│  └─ rag_data
│     ├─ __about__.py        版本号
│     ├─ __init__.py         公共入口，统一导出 API
│     ├─ py.typed            类型标记（PEP 561）
│     ├─ config.py           pydantic-settings 分区配置
│     ├─ exceptions.py       领域异常基类
│     ├─ registry.py         通用注册表（嵌入 provider 复用）
│     ├─ facade.py           流程门面，封装各流程调用
│     ├─ parsers             DocumentParser 基类 + 各格式子类
│     │  ├─ __init__.py      后缀索引与派发：resolve_parser、parse_document、chunk_document
│     │  ├─ base.py          DocumentParser：契约、句柄加载与切块原语
│     │  ├─ markdown.py      .md、.markdown
│     │  ├─ txt.py           .txt
│     │  ├─ epub.py          .epub
│     │  ├─ pdf.py           .pdf
│     │  └─ word.py          .docx
│     ├─ embedding           向量化：抽象、注册表与 openai、qwen 实现
│     │  ├─ base.py          BaseEmbeddingProvider 抽象
│     │  ├─ registry.py      provider 注册表
│     │  ├─ openai_provider.py  OpenAI 实现（官方 openai SDK）
│     │  ├─ qwen_provider.py    通义千问实现（继承 OpenAI）
│     │  └─ embedder.py      批次与维度校验
│     └─ logging             日志适配层
│        ├─ base.py          LoggerAdapter 抽象
│        ├─ factory.py       LoggerFactory：全局装配与 get_logger
│        ├─ stdlib_adapter.py   StdlibLogAdapter 与 StdlibTimezoneFormatter
│        ├─ loguru_adapter.py   loguru 适配
│        └─ structlog_adapter.py structlog 适配
└─ tests
   ├─ conftest.py
   ├─ test_facade.py
   ├─ test_logging.py
   ├─ test_parsers_base.py
   ├─ test_parsers_dispatch.py
   ├─ test_parsers_epub.py
   ├─ test_parsers_markdown.py
   ├─ test_parsers_pdf.py
   ├─ test_parsers_txt.py
   ├─ test_parsers_word.py
   └─ test_integration_qwen_epub_milvus.py
```

说明：日志适配层目录名为 logging，位于 rag_data 包内，采用绝对导入避免与标准库 logging 冲突。

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

### M1 类型模型与配置（1 人天，models 已移除）

任务：

- models 包：以 pydantic BaseModel 定义 DocumentChunk，切块元数据与向量同体
- 为关键字段加约束：chunk_id 与 text 非空、created_at 非负；向量维度由 Embedder 校验
- config.py：以 pydantic-settings BaseSettings 定义 Settings，按模块分区，含 env_prefix、嵌套分隔符、.env 加载
- 对互相约束的字段使用 model_validator，对单字段使用 field_validator
- 配置来源优先级：初始化参数 大于 环境变量 大于 .env 大于 默认值

交付：models 包、config.py
验收：非法输入抛 ValidationError 且信息可读；环境变量可覆盖默认值

### M2 日志适配层（1 人天，已重构为 LoggerFactory）

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

### M5 向量化模块（2 人天，已落地）

任务：

- embedding：provider 注册表与 openai、qwen 实现，批量编码并按 embedding.dim 校验维度
- parsers 子包：五种格式解析为纯文本，各格式自持切块规则
- 门面对外暴露 parse、chunk、vectorize、vectorize_text

交付：embedding 子包、parsers 子包
验收：样例文档解析与切块正确，向量条数与维度符合预期


### M6 公共入口与流程门面（1 人天，ingest 已移除）

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

- config.Settings：pydantic-settings 的 BaseSettings，按模块分区，实例化即完成校验；load(source, use_env=True, **overrides) 分层覆盖
- logging.base.LoggerAdapter：bind、debug、info、warning、error、exception
- logging.factory.LoggerFactory：setup(backend, timezone_name, level, fmt) 全局装配，get_logger(name) 取用适配器
- parsers.resolve_parser(path, max_chars=None, safe_max_chars=None, logger=None) 按扩展名构造解析器实例
- parsers.parse_document(path, logger=None) 返回纯文本字符串
- parsers.chunk_document(path, text, nlp=None, logger=None, max_chars=None, safe_max_chars=None) 按扩展名派发到该格式的 chunk
- parsers.base.DocumentParser：SUFFIXES、parse(path)、chunk(text, nlp)；logger 经构造注入
- embedding.embedder.Embedder.encode(texts) 返回向量列表，批次取配置与 provider 上限的较小者
- facade.RagData：parse、chunk、vectorize、vectorize_text，依赖均可注入
- 一键入口：vectorize(texts, source)

## 6. 配置与默认值

配置按模块分区，与 JSON 结构一一对应；模型为 pydantic 嵌套模型，环境变量与代码硬编码均可覆盖。

| 模块 | 字段 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| parsing | spacy_model | zh_core_web_md | 分句模型 |
| parsing | max_chars | 500 | 单块字符上限，注入各解析器 |
| parsing | safe_max_chars | 2000 | 单块安全上限，极端长句兜底 |
| embedding | provider | openai | 嵌入模型注册表中的键 |
| embedding | model | 空字符串 | 留空用 provider 默认模型 |
| embedding | batch_size | 128 | 批量编码大小 |
| embedding | dim | 1024 | 期望输出维度，向量化结果据此校验 |
| embedding | api_key | 空字符串 | 留空时读取 provider 约定的环境变量 |
| embedding | base_url | 空字符串 | 留空时用 provider 默认地址 |
| embedding | timeout | 60.0 | 单次请求超时秒数 |
| logging | backend | stdliblog | stdliblog、structlog 或 loguru |
| logging | level | INFO | 日志级别 |
| logging | format | 空字符串 | 留空时读 pyproject 的 log_cli_format，再回退内置默认 |
| logging | timezone | 空字符串 | 留空用系统本地时区，可填 UTC 或 IANA 名 |

配置来源优先级：代码硬编码 大于 环境变量与 .env 大于 默认值。

已移除：chunking 与 nlp 分区（合并进 parsing）、half_life_days 等召回参数、storage 分区、logging.json。

## 7. 测试策略

- 配置层：默认值、环境变量覆盖、代码硬编码优先、分区名校验
- 日志层：LoggerFactory 装配、时区格式化、调用点（完整路径）与字段绑定
- 解析与切块：各格式契约、解析依赖缺失报错、按井号分章与重叠前缀、递归切块边界
- 派发层：后缀索引唯一性、按扩展名注入长度配置构造解析器
- 向量化：批次切分、维度校验、假模型注入
- 门面：各流程装配、逐流程调用、一键入口与依赖注入
- 手工集成：真实 EPUB 解析、切块、Qwen 向量化、Milvus 写入与检索（默认跳过）
- 默认以 -m 'not integration' 跳过集成项；覆盖率目标为核心模块 85 以上

## 8. 依赖与风险

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| pydantic v1 与 v2 语法差异 | 配置与校验写法不兼容 | 锁定 v2，统一使用 ConfigDict、field_validator、model_validator |
| 日志框架为可选依赖 | 未安装时导入失败 | 适配器惰性导入，选用时缺失抛 OptionalDependencyError 并给出安装指引 |
| 子包名 logging 与标准库同名 | 误导入歧义 | 包内一律绝对导入，必要时改名 observability |
| spaCy 中文模型体积与安装 | 首次环境搭建慢 | 句柄失败缓存并降级；模型经 parsers extra 一并安装 |
| Unstructured 与 MinerU 版本差异 | 解析结果不稳定 | 锁版本并加解析回归样例 |
| Embedding 模型与维度不一致 | 下游按错误维度使用 | 维度写入 embedding.dim，Embedder 编码后逐条校验 |
| Python 3.8 运行期类型语法 | list 下标等新语法在 3.8 报错 | 文件首行加 from __future__ import annotations |

## 9. 建议排期（合计约 7.5 人天）

| 里程碑 | 内容 | 人天 | 状态 |
| :--- | :--- | :--- | :--- |
| M0 | 骨架与环境 | 0.5 | 已完成 |
| M1 | 类型模型与配置（pydantic） | 1.0 | 配置已完成；models 已移除 |
| M2 | 日志适配层 | 1.0 | 已完成（重构为 LoggerFactory） |
| M3 | 两阶段切块 | 1.0 | 已完成 |
| M5 | 向量化模块 | 2.0 | 已完成 |
| M6 | 公共入口与流程门面 | 1.0 | 已完成 |
| M7 | 测试与文档 | 1.0 | 持续维护 |
| 合计 | | 7.5 | |

## 10. 落地顺序与依赖

M0 → M1 → M2 → M3 → M5 → M6 → M7

说明：M1 类型层与 M2 日志层为横切基础，先行落地；M5 依赖 M3；M6 依赖 M5；M7 收口。本计划不含召回链路，故无召回分支。

## 11. 代码层级设计说明

### 11.1 分层结构与依赖方向

```text
RagData（facade）        表现层：流程门面
   |
parsers、embedding      领域服务层：解析切块与向量化


config.Settings            横切层：类型安全的只读配置
registry、logging          横切层：通用注册表与日志抽象
```

依赖原则：

1. 门面通过装配获得依赖，领域服务不直接依赖具体实现
2. 具体实现通过构造注入，即依赖反转，便于替换为测试替身
3. 领域服务层只依赖 LoggerAdapter 抽象，不 import loguru 或 structlog
4. config 为不可变且已校验的配置对象，逐层向下传递，禁止模块级全局可变状态
5. 嵌入 provider 经注册表解析，新增实现不改动装配代码


### 11.2 数据模型（已移除）

> DocumentChunk 与 models 子包已移除：切块不再产出 pydantic 模型，向量化直接返回向量列表。
> 下文为历史设计，仅作参考。

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
4. chunk_id 由内容哈希生成，重复处理同一文件得到相同 id，便于按内容去重

### 11.3 各模块职责与接口签名

#### 11.3.1 config.py（pydantic-settings，按模块分区）

职责：以嵌套模型承载各模块配置，实例化即完成校验；默认读取环境变量，也支持代码硬编码。

```python
class ParsingSettings(BaseModel):
    spacy_model: str = Field(default="zh_core_web_md")
    max_chars: int = Field(default=500, gt=0)
    safe_max_chars: int = Field(default=2000, gt=0)


class EmbeddingSettings(BaseModel):
    provider: str = "openai"
    model: str = ""
    batch_size: int = Field(default=128, gt=0)
    dim: int = Field(default=1024, gt=0)
    api_key: str = ""
    base_url: str = ""
    timeout: float = Field(default=60.0, gt=0)


class LoggingSettings(BaseModel):
    backend: Literal["stdliblog", "structlog", "loguru"] = "stdliblog"
    level: str = "INFO"
    format: str = ""        # 留空时由工厂读取 pyproject 的 log_cli_format
    timezone: str = ""      # 留空用系统本地时区


ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        extra="ignore",
    )

    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def load(cls, source=None, *, use_env=True, **overrides) -> Settings: ...
```

加载入口与优先级：

| 入口 | 说明 |
| :--- | :--- |
| Settings() | 默认读取环境变量（含 .env），未提供时用字段默认值 |
| Settings.load(embedding={...}) | 代码硬编码分区配置，逐字段覆盖环境变量 |
| Settings.load({...}) | 代码硬编码整体字典，与分区写法等价 |
| Settings.load(use_env=False) | 忽略环境变量，仅取硬编码与字段默认值 |

优先级由高到低：代码硬编码 > 环境变量与 .env > 字段默认值。

设计要点：

1. 环境变量作为基线，代码硬编码在其上逐字段覆盖
2. 每个分区模型 extra 为 forbid，分区内拼错字段会在加载期即报错
3. 根模型 extra 为 ignore：RAG_ 命名空间可能混入无关变量，避免误报
4. 代码硬编码中的分区名由 _validate_sections 校验，写错即抛 ConfigError 并列出可用分区
5. 取值交由 pydantic 转换，JSON 对象与数组按 JSON 解析
6. use_env 为 False 时先以分区默认值补齐全部字段再作为构造实参，环境变量无从渗入
7. 不提供配置文件读取：配置只来自代码与环境变量


#### 11.3.2 logging 子包（多框架适配）

职责：以统一接口封装不同日志框架，业务代码只依赖抽象，框架可插拔。

##### base.py

```python
class LoggerAdapter(ABC):
    def bind(self, **fields) -> LoggerAdapter: ...
    def debug(self, msg, **fields) -> None: ...      # info、warning、error、exception 同
```

##### 三个适配器

| 适配器 | 后端 | 说明 |
| :--- | :--- | :--- |
| StdlibLogAdapter | 标准库 logging | 以 stacklevel 回退栈帧，调用点指向业务代码；配套 StdlibTimezoneFormatter |
| StructlogAdapter | structlog | 以 _stacklevel 与 CallsiteParameterAdder 定位调用点与完整路径 |
| LoguruAdapter | loguru | 以 opt(depth) 回退栈帧，format 回调渲染固定格式 |

##### factory.py

```python
class LoggerFactory:
    @classmethod
    def setup(cls, backend="stdliblog", timezone_name=None, level="INFO", fmt=None) -> None: ...
    @classmethod
    def get_logger(cls, name="rag_data") -> LoggerAdapter: ...
```

设计要点：

1. setup 全局装配后端、级别、格式与时区，get_logger 取用适配器
2. 后端名称为 stdliblog、structlog、loguru，缺省 stdliblog
3. 格式 fmt 优先级：参数 大于 pyproject 的 log_cli_format 大于 内置默认（均含完整路径、方法名与行号）
4. 时区留空用系统本地，可填 UTC 或 IANA 名（需 Python 3.9 及以上）；无法识别时装配报错
5. 切换后端仅改配置，业务代码零改动
6. 门面 build_logger 从 logging 分区取配置后调用 LoggerFactory.setup


#### 11.3.3 parsers 子包（解析与切块；ingestion 已移除）

##### parsers/（基类与子类）

目录：__init__.py（派发）、markdown.py、txt.py、epub.py、pdf.py、word.py。

- 职责：按扩展名派发到各 DocumentParser 子类，输出纯文本
- 接口：resolve_parser(path, max_chars, safe_max_chars, logger) 构造实例；parse_document(path, logger) 返回纯文本；chunk_document(path, text, nlp, logger, max_chars, safe_max_chars) 派发切块
- 派发：__init__ 由 PARSER_CLASSES 汇总后缀，得到「扩展名 → 解析器类」索引；新增格式只需实现约定并登记
- 后缀分组：MARKDOWN 为 .md 与 .markdown；TXT 为 .txt；EPUB 为 .epub；PDF 为 .pdf；WORD 为 .docx；TEXT 为 MARKDOWN 并 TXT，BINARY 为 PDF 并 WORD
- 依赖：unstructured 或 MinerU（PDF 与 Word）、ebooklib 与 BeautifulSoup（EPUB）；一律惰性导入，缺失时抛 ParserDependencyError
- 各格式均已落地：Markdown 与 TXT 直读；EPUB 按 reading order 提取章节并排除目录页；PDF 用 pdfplumber 按物理坐标提取并过滤页眉页脚；Word 用 python-docx 提取段落
- 各格式 chunk(text, nlp) 同签名，规则自持，长度为构造参数
- TODO：版面级解析细化，如 PDF 双栏、Word 表格；Markdown 目前保留原始标记

##### parsers/base.py 的句柄加载

- 职责：spaCy 句柄的按需加载与缓存，供装配层（facade）与需要严格句柄的格式共用
- 接口：DocumentParser.load_nlp(model_name=DEFAULT_MODEL, logger=None) 与 clear_cache()；DEFAULT_MODEL 为 zh_core_web_md
- 设计：未装 spaCy 或模型缺失时返回 None 并告警；加载时停用 tagger、ner 与 attribute_ruler，按需补 sentencizer；成功与失败均缓存，clear_cache() 可重建


##### parsers/base.py 与各格式 chunk

- 职责：各 parsers/<format>.py 的子类自持 parse 与 chunk 规则；基类只提供句柄加载
- 接口：各格式 chunk(text, nlp) 与 parse(path)；max_chars 与 safe_max_chars 为构造参数，logger 亦经构造注入
- 设计：切块规则随格式不同，由各子类自行实现；nlp 由装配层注入，传 None 时按各子类规则处理（缺失句柄时抛 ParserDependencyError）

##### entities.py（已移除）

- 职责：基于 spaCy NER 抽取实体，供切块元数据与后续过滤使用
- 接口：def extract_entities(text, nlp=None, *, labels=None) -> List[str]
- 设计：句柄由调用方注入，未注入（None）则不抽取；归一化（去空白、剥包裹引号括号）、过滤纯标点、大小写不敏感去重，并可选按实体类型白名单过滤

##### pipeline.py（已移除）

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
- 请求走官方 openai SDK，客户端惰性初始化（api_key、base_url、timeout 取自配置）；连接、状态与其它 SDK 错误统一包装为 EmbeddingError
- client 属性为 SDK 接缝，便于注入替身；_build_request_kwargs 为请求体接缝，子类可覆盖
- _parse 按 index 排序还原顺序，保证向量与输入文本一一对应
- 仅当显式配置 dim 且与默认维度不同时才传 dimensions，兼容不支持该参数的模型

##### embedder.py

- 职责：按配置装配 provider，统一批次与维度校验
- 接口：class Embedder 提供 encode(self, texts: List[str]) -> List[List[float]]
- 设计：provider 由 embedding.provider 经注册表解析；注入的 model 优先且绕过配置
- 设计：批次大小取 embedding.batch_size 与 provider 的 max_batch_size 的较小者
- 设计：输出维度需与 embedding.dim 一致，不一致抛 EmbeddingError
- 设计：空输入直接返回，不触发 provider 装配


### 11.4 关键调用链

两条链路，伴随日志埋点：

```text
文件向量化：parse_document  ->  chunk_document  ->  切块列表
字符串向量化：Embedder.encode（分批）  ->  向量列表
```

横切能力在链路两端生效：

- 入参：Settings 已完成校验，LoggerAdapter 已就绪
- 出参：切块为字符串列表（已去空白），向量已按 embedding.dim 校验

### 11.5 异常与错误处理

| 异常 | 触发场景 | 处理策略 |
| :--- | :--- | :--- |
| ConfigError | 配置分区名非法、配置源非字典 | load 阶段即抛，并列出可用分区 |
| OptionalDependencyError | 可选框架或解析库未安装 | 给出安装指引；日志层的 loguru 与 structlog 缺失时抛此错，由调用方选择后端 |
| ParserDependencyError | 解析或句柄依赖缺失 | 给出安装指引；切块缺少句柄时抛出 |
| ParseError | 文件不存在、文档损坏或格式不支持 | 抛出并附带路径与原因 |
| EmbeddingError | 缺少 API Key、向量化服务失败或维度不一致 | 抛出；维度按 embedding.dim 逐条校验 |

原则：领域异常统一继承基类 RagDataError，便于上层统一捕获；外部依赖异常在边界层转换。


### 11.6 并发与幂等

- 批量编码：按 embedding.batch_size 与 provider 上限分批串行，避免触发限流
- 日志并发：LoggerAdapter 的 bind 返回新实例，不做共享状态改写，天然并发安全
- 句柄缓存：spaCy 句柄按模型名缓存，成功与失败均缓存，clear_cache 可重建


### 11.7 日志与可观测性

日志以 LoggerAdapter 统一出口，后端由 LoggerFactory 装配切换：

| 场景 | 建议后端 | 说明 |
| :--- | :--- | :--- |
| 本地开发 | loguru 或 structlog | 彩色与结构化输出，调试友好 |
| 生产与容器 | structlog | 便于采集与检索 |
| 最小依赖或库内嵌 | stdliblog | 零额外依赖，默认回退 |

人类可读行固定为：时间、级别、源文件完整路径与行号、方法名、消息；
时区由 logging.timezone 配置，留空用系统本地。

关键埋点：解析（所属格式与字符数）、切块（块数）、向量化（批次数与维度）。
约定：异常统一用 exception 记录堆栈；业务字段以关键字传入并追加到行尾。


### 11.8 单元测试与代码映射

| 测试文件 | 覆盖对象 | 关键用例 |
| :--- | :--- | :--- |
| test_logging.py | LoggerFactory 与三适配器 | 装配、时区格式化、调用点完整路径、字段绑定 |
| test_parsers_base.py | DocumentParser | 分句、四分之一重叠前缀、递归切块与默认值 |
| test_parsers_dispatch.py | 派发层 | 后缀索引完整性、按扩展名注入长度配置 |
| test_parsers_txt.py | TxtParser | 契约、解析、切块与本机文件 |
| test_parsers_markdown.py | MarkdownParser | 契约、解析、按井号分章切块 |
| test_parsers_epub.py | EpubParser | 契约、章节提取、切块与复杂内容 |
| test_parsers_pdf.py | PdfParser | 契约、解析依赖与异常、切块 |
| test_parsers_word.py | WordParser | 契约、段落提取、切块 |
| test_facade.py | 流程门面 | 各流程装配、逐流程调用、一键入口与依赖注入 |
| test_integration_qwen_epub_milvus.py | 手工集成 | 真实 EPUB 解析、Qwen 向量化、Milvus 写入与检索 |


### 11.9 公共入口（Public API，41 个符号）

对外只暴露一个文件，即包根的 __init__.py，内部子包路径不外泄。

```python
import rag_data
from rag_data import Settings, RagData
```

| 分组 | 导出符号 |
| :--- | :--- |
| 版本 | __version__ |
| 子包 | config、parsers、embedding、logging |
| 配置 | Settings、ParsingSettings、EmbeddingSettings、LoggingSettings、load_env_overrides、ENV_PREFIX、ENV_NESTED_DELIMITER |
| 解析 | parse_document |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 流程门面 | RagData、vectorize、build_settings、build_logger、build_nlp、build_embedder |
| 日志 | LoggerAdapter、LoggerFactory、StdlibLogAdapter、StructlogAdapter、LoguruAdapter |
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

app = RagData.create()              # 装配配置、日志、句柄与向量化
text = app.parse("a.md")             # 流程：解析文件为文本
chunks = app.chunk("a.md", text)     # 流程：按格式切块
app.vectorize(["一段文本"])           # 流程：向量化字符串
```

分流程的装配方法：

| 方法 | 职责 |
| :--- | :--- |
| build_settings(source, **overrides) | 加载配置，支持整体字典与分区硬编码 |
| build_logger(settings) | 按 logging 分区装配日志工厂并取用适配器 |
| build_nlp(settings, logger) | 加载 spaCy 句柄，缺失或失败时降级为 None |
| build_embedder(settings, logger, model) | 装配向量化组件，provider 由配置决定 |
| vectorize(texts, source) | 一键向量化字符串 |

RagData 实例方法：parse(path)、chunk(path, text)、vectorize(texts)、vectorize_text(text)，依赖均可注入。

设计约定：

1. 门面只做流程装配与转发，不重复实现业务逻辑，真实逻辑仍在各子包
2. RagData 支持依赖注入，embedder、logger、nlp 均可外部替换，便于测试
3. chunk 从 parsing 分区取 max_chars 与 safe_max_chars 注入解析器
4. 可选依赖（spaCy）一律惰性导入并降级，保证门面在最小依赖环境可用

## 12. 编码规范与工程约定

1. 各模块首行统一使用 from __future__ import annotations，兼容 Python 3.8 的类型语法
2. 数据模型与配置一律使用 pydantic v2 写法：ConfigDict、field_validator、model_validator
3. 业务代码只依赖 LoggerAdapter 抽象，禁止直接 import loguru、structlog 或标准库 logging
4. 日志轮转与级别由配置驱动，不在业务代码里硬编码 handler
5. 外部依赖 spacy、unstructured、解析库、openai、loguru、structlog 惰性导入，缩短冷启动并便于注入测试替身
6. 公共接口均带类型注解，交由 mypy 配合 pydantic 插件检查
7. 使用 ruff 统一风格，固定行宽与导入顺序
8. 不新增模块级可变全局状态，依赖一律通过构造注入
9. 对外 API 一律经包根 __init__.py 导出并在 __all__ 中登记，子包内部符号不作为公共契约
10. 代码注释与文档随实现同步：模块职责变化时一并更新 README 与 .vcl 与 plan 中的说明
