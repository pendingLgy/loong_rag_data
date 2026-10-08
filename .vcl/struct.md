# rag-data 项目结构

> 版本 0.1.0 · Python >=3.8 · src 布局 · 公共导出 41 个符号

本文档描述仓库的物理结构与模块职责，是阅读代码与二次开发的导航图。
设计取舍与开发计划见 plan/DEVELOPMENT_PLAN.md，使用方式见 README.md。

> 存储层（向量库实现、后端注册表、落库配置）已整体移除，当前聚焦「解析 → 切块 → 向量化」；
> 落库流程待重新设计，现由手工集成测试直连 Milvus 验证写入与检索。

---

## 1. 概览

rag-data 是「历史文本导入与向量化」管道：把文件解析为分块，
编码为稠密向量；也支持字符串直接向量化。
整体分为五层，层间只依赖下层，方向单一。


| 层 | 包 | 职责 |
| :--- | :--- | :--- |
| 配置层 | config.py | 分区配置模型，环境变量与代码硬编码 |
| 契约层 | exceptions.py | 领域异常 |
| 通用设施 | registry.py、logging | 通用注册表、日志适配层 |
| 领域服务 | parsers、embedding | 解析切块与向量化 |
| 门面层 | facade.py | 分流程装配 + 一站式调用 |
| 公共入口 | __init__.py | 统一导出，外部只依赖包根 |

---

## 2. 目录结构

```text
rag-data/
├─ pyproject.toml             构建与依赖声明（hatchling + hatch 环境）
├─ README.md                  使用说明：安装、快速开始、公共 API、各专题
├─ rag_artifact.md            架构设计源文档
├─ .vcl/struct.md             本文件：项目结构说明
├─ plan/
│  └─ DEVELOPMENT_PLAN.md     设计与开发计划
├─ src/
│  └─ rag_data/
│     ├─ __about__.py         版本号（hatch 版本来源）
│     ├─ __init__.py          公共入口，统一导出 41 个符号
│     ├─ py.typed             类型标记（PEP 561）
│     ├─ config.py            分区配置模型与环境变量解析
│     ├─ exceptions.py        领域异常层次
│     ├─ registry.py          通用类型注册表（嵌入 provider 复用）
│     ├─ facade.py            流程门面与一键入口
│     ├─ parsers/            文档解析，一种格式一个 DocumentParser 子类
│     │  ├─ __init__.py      汇总 PARSER_CLASSES，按扩展名构造解析器并派发 parse_document、chunk_document
│     │  ├─ base.py          DocumentParser 基类：SUFFIXES 与 parse、chunk 契约，logger 经构造注入
│     │  ├─ markdown.py      MarkdownParser：.md、.markdown，按井号分章后递归切块
│     │  ├─ txt.py           TxtParser：.txt，整篇递归切块
│     │  ├─ epub.py          EpubParser：.epub，章节提取（排除目录页）与按井号分章递归切块
│     │  ├─ pdf.py           PdfParser：.pdf，按页提取（pdfplumber）与整篇递归切块
│     │  └─ word.py          WordParser：.docx，段落提取（python-docx）与整篇递归切块
│     ├─ embedding/          向量化：抽象、注册表、内置 provider
│     │  ├─ base.py            BaseEmbeddingProvider 抽象
│     │  ├─ registry.py        provider 注册表
│     │  ├─ openai_provider.py OpenAI 实现（官方 openai SDK）
│     │  ├─ qwen_provider.py   通义千问实现（继承 OpenAI）
│     │  └─ embedder.py        按配置装配 provider，统一批次
│     └─ logging/            日志适配层
│        ├─ base.py            LoggerAdapter 抽象
│        ├─ factory.py         LoggerFactory：全局装配与 get_logger
│        ├─ stdlib_adapter.py  标准库适配与 StdlibTimezoneFormatter
│        ├─ loguru_adapter.py  loguru 适配
│        └─ structlog_adapter.py structlog 适配
└─ tests/                   单元测试与手工集成测试（10 个测试文件 + conftest）
```

---

## 3. 分层与依赖方向

依赖自上而下，无环。较低层不反向导入较高层。

```text
__init__（公共入口）
    │
facade（流程门面）
    ├── config           分区配置
    ├── logging          日志适配层
    ├─ parsers           文档解析与切块
    └── embedding         向量化（门面直接持有 Embedder）
              ▲
              │  注册表反向登记：
     embedding/base 延后导入 registry，避免包导入期循环依赖
```

关键约定：

1. 可选依赖（spacy、loguru、structlog、openai）一律惰性导入，未安装时包仍可正常导入
2. 嵌入后端选择走注册表，新增实现不改动装配代码
3. 唯一的外部入口是包根 __init__.py，内部子包路径不作为稳定契约

---
## 4. 基础模块

### 4.1 __about__.py

仅保存 __version__，由 pyproject.toml 的 [tool.hatch.version] 读取。

### 4.2 __init__.py（公共入口）

只做统一导出，不含业务逻辑。外部调用形如：

```python
import rag_data
from rag_data import Settings, RagData
from rag_data import register_embedding, BaseEmbeddingProvider
```

导出 41 个符号，分组见 6.2 节。

### 4.3 config.py

职责：以嵌套模型承载分区配置，实例化即校验；默认读环境变量，支持代码硬编码。

```python
class ParsingSettings(BaseModel): ...     # 解析：spacy_model、max_chars 与 safe_max_chars
class EmbeddingSettings(BaseModel): ...   # 向量化
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
| _with_defaults(layered) | 用分区默认值补齐字段，关闭环境变量时屏蔽环境 |
| _assign_nested(data, parts, value) | 按路径写入嵌套字典 |
| _parse_env_value(raw) | JSON 对象与数组按 JSON 解析 |

优先级：代码硬编码 > 环境变量与 .env > 字段默认值。

### 4.4 exceptions.py

单一根异常，便于上层统一捕获。

```text
RagDataError
├─ ConfigError            配置缺失或非法
├─ DataError              数据校验失败
├─ OptionalDependencyError 可选依赖未安装
│  └─ ParserDependencyError 文档解析依赖未安装
├─ ParseError             文档损坏或格式不支持
└─ EmbeddingError         向量化失败
```

### 4.5 registry.py（通用注册表）

嵌入模型使用的注册表，dict 子类，支持惰性导入与统一实例化。

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

### 4.6 logging/

把三种日志库收敛到同一接口，调用方只依赖 LoggerAdapter。

| 文件 | 内容 |
| :--- | :--- |
| base.py | LoggerAdapter 抽象：bind 与五个级别方法 |
| factory.py | LoggerFactory：setup 全局装配与 get_logger 取用 |
| stdlib_adapter.py | StdlibLogAdapter 与 StdlibTimezoneFormatter（按 tz 格式化时间） |
| loguru_adapter.py | LoguruAdapter 与惰性导入 |
| structlog_adapter.py | StructlogAdapter 与惰性导入 |

LoggerFactory.setup 全局装配后端、级别、格式与时区，get_logger 取用适配器；
后端名称为 stdliblog、structlog、loguru，缺省 stdliblog。

格式 fmt 优先级：参数 > pyproject 的 log_cli_format > 内置默认（均含完整路径与行号）；
时区 timezone 留空用系统本地，可填 UTC 或 IANA 名。


---
## 5. 领域服务层

### 5.1 parsers(解析与切块)

```text
文件 ─ parse_document ─ 文本 ─ chunk_document ─ 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| parsers/ | resolve_parser、parse_document、chunk_document | 按扩展名构造并派发到 DocumentParser 子类；各格式规则自持 |
| parsers/base.py | DocumentParser | 抽象基类：parse(path) 与 chunk(text, nlp) 契约（logger 经构造注入）；句柄加载 load_nlp、clear_cache 与分句 split_sentences、重叠前缀 _extract_overlap_prefix、递归切块 _process_text_recursive |
| parsers/txt.py | TxtParser | .txt，整篇递归切块 |
| parsers/markdown.py | MarkdownParser | .md、.markdown，按行首井号分章后递归切块 |
| parsers/epub.py | EpubParser | .epub，按 spine 提取章节（排除目录页）后按井号分章递归切块 |
| parsers/pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取、过滤页眉页脚后整篇递归切块 |
| parsers/word.py | WordParser | .docx，python-docx 提取段落，整篇递归切块 |

切块由各解析器自持：基类提供递归切分与四分之一重叠前缀；单块上限 max_chars 与安全上限 safe_max_chars 为解析器构造参数，由 parsing 分区注入（默认 500 与 2000）。
句柄模型默认 zh_core_web_md，加载时关闭 tagger、ner 与 attribute_ruler 以省开销。

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
| openai_provider.py | OpenAIEmbeddingProvider，使用官方 openai SDK（openai extra） |
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
| qwen | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY | 10 |

请求细节集中在 openai_provider 的接缝属性与方法，子类与测试可覆盖：

| 方法 | 作用 |
| :--- | :--- |
| client | 惰性初始化的 OpenAI 客户端（api_key、base_url、timeout） |
| _build_request_kwargs(texts) | 构造请求参数字典，仅在维度被覆盖时传 dimensions |
| _parse(data) | 按 index 排序还原顺序，保证与输入一一对应 |

Embedder 的批次策略：取 embedding.batch_size 与 provider 的 max_batch_size 的较小者，
因此 Qwen 的 10 条上限自动生效，无需用户手动调小。编码结果按 embedding.dim 校验。


---

## 6. 门面层与公共 API

### 6.1 facade.py

分流程装配方法，每个方法对应一个可独立调用的流程：

| 方法 | 流程 |
| :--- | :--- |
| build_settings(source, **overrides) | 加载配置 |
| build_logger(settings) | 装配日志 |
| build_nlp(settings, logger) | 加载 spaCy，不可用时降级为 None |
| build_embedder(settings, logger, model=None) | 装配向量化 |

RagData 是一次装配、逐流程调用的门面：

```python
RagData(settings, *, embedder=None, logger=None, nlp=UNSET, model=None)
RagData.create(source=None, *, overrides=None, **kwargs)
```

| 实例方法 | 说明 |
| :--- | :--- |
| parse(path) | 解析单个文件为纯文本 |
| chunk(path, text) | 按文件扩展名切块，规则由对应解析器自持 |
| vectorize(texts) | 批量向量化字符串，返回与输入同序的向量列表 |
| vectorize_text(text) | 单条字符串向量化，返回单个向量 |

全部依赖可注入，故测试可替换 embedder、logger、nlp。
模块级一键入口 vectorize 内部自行装配后调用。

### 6.2 公共 API 导出（41 个符号）

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 子包 | config、parsers、embedding、logging |
| 配置 | Settings、ParsingSettings、EmbeddingSettings、LoggingSettings、load_env_overrides、ENV_PREFIX、ENV_NESTED_DELIMITER |
| 解析 | parse_document |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 流程门面 | RagData、vectorize、build_settings、build_logger、build_nlp、build_embedder |
| 日志 | LoggerAdapter、LoggerFactory、StdlibLogAdapter、StructlogAdapter、LoguruAdapter |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError |

未纳入顶层导出的实现按需从子包引入，如 logging 三个适配器。


---
## 7. 数据流

### 7.1 文件向量化

```text
1. RagData.create(overrides)     装配 settings、logger、nlp、embedder
2. app.parse(path)               解析文件为纯文本
3. app.chunk(path, text)         按格式切块（各 DocumentParser 子类自持规则）
```

### 7.2 字符串向量化

```text
1. app.vectorize(texts)          批量字符串 → 向量列表（与输入同序）
2. app.vectorize_text(text)      单条字符串 → 单个向量
     ├─ 分批：embedding.batch_size 与 provider 上限取较小者
     └─ 校验：逐条按 embedding.dim 校验，不一致抛 EmbeddingError
```

字符串向量化直接返回向量；向量落库流程待存储层重新设计后补全。

---

## 8. 配置项

分区与字段（括号内为默认值）：

| 分区 | 字段 |
| :--- | :--- |
| parsing | spacy_model(zh_core_web_md)、max_chars(500)、safe_max_chars(2000) |
| embedding | provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |
| logging | backend(stdliblog)、level(INFO)、format(空，即读 pyproject)、timezone(空，即系统本地) |

三层来源，优先级由高到低：代码硬编码、环境变量与 .env、字段默认值。
环境变量命名：RAG_ 前缀，分区与字段大写，双下划线分隔，如 RAG_EMBEDDING__DIM。

维度一致性：embedding.dim 声明期望输出维度（默认 1024），Embedder 据此校验每条向量。

---

## 9. 扩展点

嵌入模型登记机制让新增实现无需改动装配代码，继承并声明 backend 即生效。

| 扩展点 | 抽象基类 | 声明方式 | 配置键 | 内置 |
| :--- | :--- | :--- | :--- | :--- |
| 嵌入模型 | BaseEmbeddingProvider | backend 类属性 | embedding.provider | openai、qwen |

新增 provider 的最小骨架：

```python
from rag_data import BaseEmbeddingProvider


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    backend = "local"          # 声明名字，类定义时自动注册
    default_model = "bge-large-zh"
    default_dim = 1024
    default_base_url = "http://localhost:8000/v1"
    api_key_env = "LOCAL_API_KEY"

    def encode(self, texts):
        # 返回 List[List[float]]，顺序与输入一致
        ...
```

---

## 10. 测试

```text
tests/
├─ conftest.py                    路径注入与 settings、logger 夹具
├─ test_facade.py         31 项   facade 装配方法、RagData 逐流程调用与一键入口
├─ test_logging.py        18 项   工厂装配、时区格式化与调用点（完整路径）
├─ test_parsers_base.py   16 项   DocumentParser 分句、四分之一重叠前缀与递归切块
├─ test_parsers_dispatch.py 9 项  后缀索引与按扩展名注入长度配置的解析器构造
├─ test_parsers_markdown.py 12 项  MarkdownParser 契约、解析、按井号分章切块与本机文件
├─ test_parsers_txt.py    12 项   TxtParser 契约、解析、chunk 切块与本机文件
├─ test_parsers_epub.py   10 项   EpubParser 契约、初始化与 chunk 切块（含复杂内容与本机文件）
├─ test_parsers_pdf.py    10 项   PdfParser 契约、解析依赖与异常、chunk 切块与本机文件
├─ test_parsers_word.py   12 项   WordParser 契约、解析依赖与异常与 chunk 切块
└─ test_integration_qwen_epub_milvus.py 7 项 真实 EPUB 解析、Qwen 向量化、Milvus 写入与检索（手工集成）
```

合计 130 项单元测试（含本机冒烟）与 7 项手工集成测试；默认运行以 -m "not integration" 跳过集成项。

---

## 11. 依赖

核心依赖仅两项：pydantic 与 pydantic-settings。

| 组 | 包 |
| :--- | :--- |
| 核心 | pydantic、pydantic-settings |
| parsers | unstructured、spacy(>=3.8.16)、zh-core-web-sm、zh-core-web-md |
| epub | EbookLib、beautifulsoup4 |
| pdf | pdfplumber |
| word | python-docx |
| embedding | sentence-transformers |
| openai | openai SDK（openai extra，内置 openai 与 qwen provider 依赖） |
| milvus | pymilvus（集成测试直连 Milvus 使用） |
| loguru | loguru |
| structlog | structlog |
| dev | pytest、pytest-cov、mypy、ruff |

openai 与 qwen 两个内置 provider 都基于官方 openai SDK，故需安装 openai extra。

---

## 12. 待补全

| 位置 | 事项 |
| :--- | :--- |
| parsers/ | 版面级解析细化，如 PDF 双栏、Word 表格 |
| embedding | 兼容本地模型的 provider，如 sentence-transformers |
| facade.py | 装配流程细化，按需启用句柄与向量化 |
| 数据保存 | 向量落库流程，待存储层重新设计后补全 |
