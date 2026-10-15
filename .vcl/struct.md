# rag-data 项目结构

> 版本 0.1.0 · Python >=3.8 · src 布局 · 包根仅导出当前目录直接模块的 20 个符号

本文档描述仓库的物理结构与模块职责，是阅读代码与二次开发的导航图。
设计取舍与开发计划见 plan/DEVELOPMENT_PLAN.md，使用方式见 README.md，开发、构建与测试说明见 development.md。

> 存储层（向量库实现、后端注册表、落库配置）已整体移除，当前聚焦「解析 → 切块 → 向量化」；
> 落库流程待存储层重新设计，现由手工集成测试直连 Milvus 验证写入与检索。

---

## 1. 概览

rag-data 是「历史文本导入与向量化」管道：把文件解析为分块，编码为稠密向量；也支持字符串直接向量化。
整体分层，层间只依赖下层，方向单一。

| 层 | 包 | 职责 |
| :--- | :--- | :--- |
| 配置层 | config.py | 分区配置模型与三类来源加载（JSON 对象、环境变量、config.yml） |
| 契约层 | exceptions.py | 领域异常 |
| 通用设施 | registry.py、logging | 通用注册表、日志适配层 |
| 领域服务 | parsers、embedding | 解析切块与向量化 |
| 门面层 | facade.py | 一站式装配 RagData 与私有装配方法 |
| 公共入口 | __init__.py | 只导出当前目录（直接模块）的公共符号，不含子包 |

---

## 2. 目录结构

```text
rag-data/
├─ pyproject.toml          构建与依赖声明（hatchling + hatch 环境）
├─ README.md               使用说明：特性、快速开始、包根导出、配置、解析、向量化、日志、门面
├─ development.md          开发指南：环境与依赖管理、构建与测试命令
├─ rag_artifact.md         架构设计源文档
├─ .vcl/struct.md          本文件：项目结构说明
├─ plan/
│  └─ DEVELOPMENT_PLAN.md   设计与开发计划
├─ src/
│  └─ rag_data/
│     ├─ __about__.py      版本号（hatch 版本来源）
│     ├─ __init__.py       包根，仅导出当前目录直接模块的 20 个公共符号
│     ├─ py.typed          类型标记（PEP 561）
│     ├─ config.py         分区配置模型与三类来源加载
│     ├─ exceptions.py     领域异常层次
│     ├─ registry.py       通用注册表（Registry、Ref）
│     ├─ facade.py         流程门面 RagData 与私有装配方法 _build_*
│     ├─ parsers/
│     │  ├─ __init__.py    汇总导出，默认适配器的绑定方法作模块级函数
│     │  ├─ base.py        DocumentParser 基类：SUFFIXES 与 parse、chunk 契约，句柄加载 load_nlp
│     │  ├─ parser_adapter.py  ParserAdapter：按扩展名派发解析与切块
│     │  ├─ markdown.py    MarkdownParser：.md、.markdown，按井号分章后递归切块
│     │  ├─ txt.py         TxtParser：.txt，整篇递归切块
│     │  ├─ epub.py        EpubParser：.epub，按 spine 提取章节后递归切块
│     │  ├─ pdf.py         PdfParser：.pdf，pdfplumber 按物理坐标提取
│     │  └─ word.py        WordParser：.docx，python-docx 提取段落
│     ├─ embedding/
│     │  ├─ __init__.py    汇总导出 provider 与注册函数
│     │  ├─ base.py        BaseEmbeddingProvider 抽象：声明 backend 即自动注册
│     │  ├─ registry.py    provider 注册表（BUILTIN_PROVIDERS 与 register、resolve、create）
│     │  ├─ openai_provider.py OpenAI 实现（官方 openai SDK）
│     │  ├─ qwen_provider.py   通义千问实现（继承 OpenAI）
│     │  └─ embedder.py    按配置装配 provider，统一批次与维度校验
│     └─ logging/
│        ├─ __init__.py    汇总导出适配器与工厂
│        ├─ base.py        LoggerAdapter 抽象
│        ├─ factory.py     LoggerFactory：全局装配与 get_logger
│        ├─ stdlib_adapter.py  标准库适配与 StdlibTimezoneFormatter
│        ├─ loguru_adapter.py  loguru 适配（惰性导入）
│        └─ structlog_adapter.py structlog 适配（惰性导入）
└─ tests/                   单元测试与手工集成测试
```

> 说明：src/rag_data 下另有 llm、memory、storage、util 四个空目录（仅残留 __pycache__，源码已移除），
> 未纳入上面的结构图，可按需清理。

---

## 3. 分层与依赖方向

依赖自上而下，无环。较低层不反向导入较高层。

```text
__init__（包根，仅导出直接模块符号）
     ▲
facade（流程门面 RagData）
     ├── config           分区配置与三类来源加载
     ├── logging          日志适配层
     ├── parsers          文档解析与切块（ParserAdapter）
     └── embedding        向量化（门面持有 Embedder）
```

关键约定：

1. 可选依赖（spacy、loguru、structlog、openai、PyYAML）一律惰性导入，未安装时包仍可正常导入
2. 嵌入后端选择走注册表，新增实现不改动装配代码
3. 包根 __init__.py 只导出当前目录直接模块的公共符号，不跨模块、不列子包；子包按模块路径取用
4. config 的 YAML 来源由 _import_yaml 惰性导入 PyYAML，缺失时抛 OptionalDependencyError

---

## 4. 基础模块

### 4.1 __about__.py

仅保存 __version__，由 pyproject.toml 的 [tool.hatch.version] 读取。

### 4.2 __init__.py（包根）

只导出当前目录直接模块的公共符号，共 20 个，不含子包，也不做跨模块重导出。外部按具体模块取用：

```python
from rag_data.config import Settings, ParsingSettings
from rag_data.facade import RagData
from rag_data.exceptions import RagDataError
```

分组见 6.2 节。

### 4.3 config.py

职责：以嵌套模型承载分区配置，实例化即校验；提供三类来源的加载方法与一组私有工具函数。

```python
class ParsingSettings(BaseModel): ...     # 解析：spacy_model、max_chars 与 safe_max_chars
class EmbeddingSettings(BaseModel): ...   # 向量化
class LoggingSettings(BaseModel): ...     # 日志

ENV_PREFIX = "RAG_"
ENV_NESTED_DELIMITER = "__"
SECTION_PARSING / SECTION_EMBEDDING / SECTION_LOGGING
MODULE_SECTIONS = [...]                   # 已知分区名列表
SECTION_MODEL_CLASSES = {...}             # 分区名到分区模型的映射

class Settings(BaseSettings):
    @classmethod
    def load_mapping(cls, source=None, **overrides): ...   # 来源一：JSON 对象（分区字典）
    @classmethod
    def load_env(cls, environ=None, **overrides): ...      # 来源二：环境变量
    @classmethod
    def load_yaml(cls, path, **overrides): ...             # 来源三：config.yml
```

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

三类来源的私有解析器挂在 Settings 上：_read_mapping、_read_env、_read_yaml。
优先与合并：overrides > 来源数据 > 分区默认值。

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

嵌入模型使用的注册表，dict 子类，支持惰性导入与统一实例化；Ref 为注册值类型别名。

```python
class Registry(dict):
    def __init__(self, kind: str, builtins: Mapping[str, Ref]): ...
    def register(self, name, ref) -> None: ...      # ref 为类对象或点分路径
    def decorator(self, name): ...                  # 装饰器形式
    def names(self) -> List[str]: ...
    def resolve(self, name) -> Type[Any]: ...       # 路径登记则此时导入
    def create(self, name, **kwargs) -> Any: ...
```

登记类对象而非路径，可让函数内定义的本地类也能注册。

### 4.6 logging/

把三种日志库收敛到同一接口，调用方只依赖 LoggerAdapter。

| 文件 | 内容 |
| :--- | :--- |
| base.py | LoggerAdapter 抽象：bind 与五个级别方法 |
| factory.py | LoggerFactory：setup 全局装配与 get_logger 取用 |
| stdlib_adapter.py | StdlibLogAdapter 与 StdlibTimezoneFormatter（按 tz 格式化时间） |
| loguru_adapter.py | LoguruAdapter 与惰性导入 |
| structlog_adapter.py | StructlogAdapter 与惰性导入 |

后端名称为 stdliblog、structlog、loguru，缺省 stdliblog；
格式 fmt 优先级：参数 > pyproject 的 log_cli_format > 内置默认（均含完整路径与行号）。

---

## 5. 领域服务层

### 5.1 parsers（解析与切块）

```text
文件 → parse_document → 文本 → chunk_document → 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| parser_adapter.py | ParserAdapter、PARSER_CLASSES | 按扩展名派发；构造时可替换解析器类集合，索引在构造时建立 |
| __init__.py | DocumentParser、ParserAdapter、PARSER_CLASSES | 汇总导出；另把默认适配器的绑定方法暴露为 resolve_parser、parse_document、chunk_document |
| base.py | DocumentParser、load_nlp | 抽象基类：SUFFIXES 与 parse(path)、chunk(text, nlp) 契约；句柄加载与缓存 |
| txt.py | TxtParser | .txt，整篇递归切块 |
| markdown.py | MarkdownParser | .md、.markdown，按行首井号分章后递归切块 |
| epub.py | EpubParser | .epub，按 spine 提取章节（排除目录页）后按井号分章递归切块 |
| pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取、过滤页眉页脚后整篇递归切块 |
| word.py | WordParser | .docx，python-docx 提取段落，整篇递归切块 |

ParserAdapter 的实例方法：

| 方法 | 作用 |
| :--- | :--- |
| resolve_parser(path, max_chars=None, safe_max_chars=None, logger=None) | 按扩展名构造该格式的解析器实例 |
| parse_document(path, logger=None) | 解析单个文档并返回纯文本 |
| chunk_document(path, text, nlp=None, logger=None, max_chars=None, safe_max_chars=None) | 按扩展名派发到该格式的 chunk |
| parser_classes（属性） | 本次派发使用的解析器类集合 |

私有：_build_index 展开后缀建索引（含唯一性断言）、_resolve 按扩展名取解析器类。
切块由各解析器自持：基类提供递归切分与四分之一重叠前缀；单块上限 max_chars 与安全上限 safe_max_chars 为构造参数，由 parsing 分区注入（默认 500 与 2000）。
句柄模型默认 zh_core_web_md，加载时关闭 tagger、ner 与 attribute_ruler 以省开销。

### 5.2 embedding/（向量化）

实现由配置 embedding.provider 选择，内置 openai 与 qwen。

```text
Embedder → create_embedding_provider → Registry → OpenAIEmbeddingProvider
                                              └ QwenEmbeddingProvider
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseEmbeddingProvider 抽象：声明 backend 即自动登记 |
| registry.py | provider 注册表（BUILTIN_PROVIDERS）与 register、resolve、create 系列函数 |
| openai_provider.py | OpenAIEmbeddingProvider，使用官方 openai SDK（openai extra） |
| qwen_provider.py | QwenEmbeddingProvider，继承 OpenAI（DashScope 兼容模式） |
| embedder.py | Embedder，按配置装配 provider，统一批次与维度校验 |

抽象类可声明的类属性：backend、default_model、default_dim、default_base_url、api_key_env、max_batch_size。

| provider | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY | 10 |

Embedder(settings: EmbeddingSettings, logger, model=None)：model 注入优先且绕过配置。
批次策略：取 embedding.batch_size 与 provider 的 max_batch_size 的较小者；编码结果按 embedding.dim 校验。

---

## 6. 门面层与公共 API

### 6.1 facade.py

模块级私有装配方法，各对应一个分区：

| 方法 | 流程 |
| :--- | :--- |
| _build_logger(logging_settings) | 装配日志适配层 |
| _build_nlp(parsing_settings, logger) | 加载 spaCy，不可用时降级为 None |
| _build_embedder(embedding_settings, logger, model=None) | 装配向量化组件 |

RagData 是一次装配、逐流程调用的门面：

```python
RagData(settings, parser=None)
```

| 成员 | 说明 |
| :--- | :--- |
| settings、parser、logger、nlp、embedder | 构造后挂载的实例变量，parser 缺省为 ParserAdapter 实例 |
| parser_adapter（属性） | 解析派发入口，返回 ParserAdapter 实例 |
| parse_document(path) | 委托 parser 解析单文件为纯文本 |
| chunk_document(path, text) | 委托 parser 切块，长度取自 parsing 分区 |

facade 的 __all__ 仅含 RagData。

### 6.2 包根导出（20 个符号）

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、ParsingSettings、EmbeddingSettings、LoggingSettings、ENV_PREFIX、ENV_NESTED_DELIMITER、SECTION_PARSING、SECTION_EMBEDDING、SECTION_LOGGING |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError |
| 通用设施 | Registry、Ref |
| 流程门面 | RagData |

未纳入包根导出的实现按需从子包引入，例如 parsers 的 ParserAdapter、embedding 的 Embedder、logging 三个适配器。

---

## 7. 数据流

### 7.1 文件向量化

```text
1. RagData(settings)                装配 settings、logger、nlp、embedder
2. app.parse_document(path)         解析文件为纯文本
3. app.chunk_document(path, text)   按格式切块（ParserAdapter 派发）
4. app.embedder.encode(chunks)      分块编码为稠密向量
```

### 7.2 字符串向量化

```text
app.embedder.encode(texts)          批量字符串 → 向量列表（与输入同序）
    ├─ 分批：embedding.batch_size 与 provider 上限取较小者
    └─ 校验：逐条按 embedding.dim 校验，不一致抛 EmbeddingError
```

字符串向量化直接返回向量；向量落库流程待存储层重新设计后补全。

---

## 8. 配置项

分区与字段（括号内为默认值）：

| 分区 | 字段 |
| :--- | :--- |
| parsing | spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |
| logging | backend(stdliblog)、level(INFO)、format(空，即读 pyproject)、timezone(空，即系统本地) |

三类来源（配置见 config.py）：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| Settings.load_mapping | JSON 对象（分区字典） | 否 |
| Settings.load_env | 环境变量 | 是（可显式传 environ） |
| Settings.load_yaml | config.yml 文件 | 否（惰性导入 PyYAML） |

环境变量命名：RAG_ 前缀，分区与字段大写，双下划线分隔，如 RAG_EMBEDDING__DIM；
取值按字段类型转换，对象与数组按 JSON 书写，分区名非法抛 ConfigError。

维度一致性：embedding.dim 声明期望输出维度（默认 1024），Embedder 据此校验每条向量。

---

## 9. 扩展点

嵌入模型登记机制让新增实现无需改动装配代码，继承并声明 backend 即生效。

| 扩展点 | 抽象基类 | 声明方式 | 配置键 | 内置 |
| :--- | :--- | :--- | :--- | :--- |
| 嵌入模型 | BaseEmbeddingProvider | backend 类属性 | embedding.provider | openai、qwen |

新增 provider 的最小骨架：

```python
from rag_data.embedding import BaseEmbeddingProvider


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
├─ conftest.py                      路径注入与 settings、logger 夹具
├─ test_config.py         18 项   三类来源（load_mapping、load_env、load_yaml）与私有工具
├─ test_facade.py         19 项   _build_* 装配方法、RagData 构造与 parse_document、chunk_document
├─ test_logging.py        17 项   工厂装配、时区格式化与调用点（完整路径）
├─ test_parsers_base.py   16 项   DocumentParser 分句、四分之一重叠前缀与递归切块
├─ test_parsers_dispatch.py 7 项  前缀索引与按扩展名注入长度配置的解析器构造
├─ test_parsers_markdown.py 12 项  MarkdownParser 契约、解析与按井号分章切块
├─ test_parsers_txt.py    12 项   TxtParser 契约、解析与 chunk 切块
├─ test_parsers_epub.py   10 项   EpubParser 契约、初始化与 chunk 切块
├─ test_parsers_pdf.py    10 项   PdfParser 契约、解析依赖与异常、chunk 切块
├─ test_parsers_word.py   12 项   WordParser 契约、解析依赖与异常、chunk 切块
└─ test_integration_qwen_epub_milvus.py 7 项 真实 EPUB 解析、Qwen 向量化、Milvus 写入与检索（手工集成）
```

合计 133 项单元测试与 7 项手工集成测试；默认运行以 -m "not integration" 跳过集成项。

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

注：PyYAML 用于 Settings.load_yaml，当前未在 pyproject 声明，生产使用前建议补充依赖或新增 yaml extra。

---

## 12. 待补全

| 位置 | 事项 |
| :--- | :--- |
| parsers/ | 版面级解析细化，如 PDF 双栏、Word 表格 |
| embedding | 兼容本地模型的 provider，如 sentence-transformers |
| facade.py | 装配流程细化，按需启用句柄与向量化 |
| 数据保存 | 向量落库流程，待存储层重新设计后补全 |
| 包根与子包 | 其余 __init__.py 的重导出策略尚未统一，可按需收敛为薄出口 |
| 遗留目录 | llm、memory、storage、util 为空目录（仅 __pycache__），待清理 |

