# rag-data

历史文本导入与向量化管道。范围聚焦离线批量导入链路：文档解析、切块、向量化；向量落库流程待后续重新设计，不含在线召回模块。

开发、构建与测试说明见 [development.md](development.md)。

## 特性

- 五类文档解析：TXT、Markdown、EPUB、PDF、Word，一种格式一个 DocumentParser 子类
- 按格式切块：切块规则由各解析器自持（递归切分 + 四分之一重叠前缀）；spaCy 句柄由门面加载并缓存，逐级注入切块
- pydantic v2 分区配置 + pydantic-settings，实例化即校验，环境变量以 RAG_ 前缀命名
- 日志适配层：stdliblog、loguru、structlog 三后端可切换
- 向量化：Embedder 封装 provider（openai、qwen），字符串直接编码为稠密向量
- 包根只导出当前目录（直接模块）的公共符号；子包按模块路径取用

## 快速开始

```python
from rag_data.config import Settings
from rag_data.facade import RagData

app = RagData(Settings.load_mapping())      # 一次装配：配置、日志、句柄、向量化
text = app.parse_document("a.md")         # 文件 → 纯文本
chunks = app.chunk_document("a.md", text)  # 文本 → 切块
vectors = app.embedder.encode(["一段文本"]) # 字符串 → 稠密向量
```

## 包根导出

rag_data 的包根 __init__.py 只导出当前目录直接模块的公共符号（共 20 个），不含子包与子目录内容：

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、ParsingSettings、EmbeddingSettings、LoggingSettings、ENV_PREFIX、ENV_NESTED_DELIMITER、SECTION_PARSING、SECTION_EMBEDDING、SECTION_LOGGING |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError |
| 通用设施 | Registry、Ref |
| 流程门面 | RagData |

子包（parsers、embedding、logging）不参与包根导出，按模块路径取用：

```python
from rag_data.parsers import ParserAdapter, parse_document
from rag_data.embedding import Embedder
from rag_data.logging import LoggerFactory
```

## 配置

config.py 按分区（parsing、embedding、logging）定义模型，提供三类来源的加载方法。

### 三类来源

| 方法 | 来源 | 说明 |
| :--- | :--- | :--- |
| Settings.load_mapping(source=None, **overrides) | JSON 对象（分区字典） | 代码硬编码，不读取环境变量 |
| Settings.load_env(environ=None, **overrides) | 环境变量 | 抽取 RAG_ 前缀，键如 RAG_EMBEDDING__DIM |
| Settings.load_yaml(path, **overrides) | config.yml 文件 | 惰性导入 PyYAML，内容须为分区映射 |

```python
from rag_data.config import Settings

settings = Settings()                            # 读 RAG_ 环境变量与 .env
settings = Settings.load_env()                   # 显式按环境变量加载
settings = Settings.load_mapping({"embedding": {"provider": "qwen"}})
settings = Settings.load_mapping(embedding={"dim": 1024}, parsing={"max_chars": 180})
settings = Settings.load_yaml("config.yml")
```

三类方法均以分区默认值补齐字段，overrides 以分区为单位逐字段覆盖。

### 环境变量

分区与字段之间用双下划线分隔，名称大写：

```bash
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__DIM=1024
RAG_LOGGING__LEVEL=DEBUG
RAG_LOGGING__TIMEZONE=Asia/Shanghai
```

取值按字段类型自动转换；对象与数组按 JSON 书写。分区名写错会抛 ConfigError，并列出可用分区。

### 分区与字段（括号内为默认值）

| 分区 | 字段 |
| :--- | :--- |
| parsing | spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |
| logging | backend(stdliblog)、level(INFO)、format(空，读 pyproject)、timezone(空，系统本地) |

维度一致性：embedding.dim 声明期望输出维度（默认 1024），Embedder 据此校验每条向量，不一致抛 EmbeddingError。

### 加载依赖说明

load_yaml 依赖 PyYAML，未安装时抛 OptionalDependencyError（惰性导入，不影响其余功能）。

## 文档解析与切块（parsers）

parsers 子包按扩展名把解析与切块派发到对应 DocumentParser 子类；派发逻辑集中在 parser_adapter.py 的 ParserAdapter。

```text
文件 → parse_document → 文本 → chunk_document → 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| parser_adapter.py | ParserAdapter、PARSER_CLASSES | 按扩展名派发；构造时可替换解析器类集合 |
| base.py | DocumentParser、load_nlp | 抽象基类（SUFFIXES 与 parse、chunk 契约）与 spaCy 句柄加载 |
| txt.py | TxtParser | .txt，整篇递归切块 |
| markdown.py | MarkdownParser | .md、.markdown，按井号分章后递归切块 |
| epub.py | EpubParser | .epub，按 spine 提取章节（排除目录页）后递归切块 |
| pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取、过滤页眉页脚 |
| word.py | WordParser | .docx，python-docx 提取段落 |

ParserAdapter 提供三个方法：

| 方法 | 作用 |
| :--- | :--- |
| resolve_parser(path, max_chars=None, safe_max_chars=None, logger=None) | 按扩展名构造该格式的解析器实例 |
| parse_document(path, logger=None) | 解析单个文档并返回纯文本 |
| chunk_document(path, text, nlp=None, logger=None, max_chars=None, safe_max_chars=None) | 按扩展名派发到该格式的 chunk |

单块上限 max_chars 与安全上限 safe_max_chars 由 parsing 分区注入（默认 500 与 2000）；句柄模型默认 zh_core_web_md。

子包根还提供默认适配器实例的绑定方法，便于直接调用：

```python
from rag_data.parsers import parse_document, chunk_document, resolve_parser
```

## 向量化（embedding）

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseEmbeddingProvider 抽象：声明 backend 即自动注册 |
| registry.py | provider 注册表与 register、resolve、create 系列函数 |
| openai_provider.py | OpenAIEmbeddingProvider（官方 openai SDK） |
| qwen_provider.py | QwenEmbeddingProvider（继承 OpenAI，DashScope 兼容模式） |
| embedder.py | Embedder：按配置装配 provider，统一批次与维度校验 |

### 内置 provider

| provider | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY | 10 |

Embedder 的批次策略：取 embedding.batch_size 与 provider 的 max_batch_size 的较小者，因此 qwen 的 10 条上限自动生效。

### 自定义 provider：继承即注册

```python
from rag_data.embedding import BaseEmbeddingProvider

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    backend = "local"                # 声明名字，类定义时自动注册
    default_model = "bge-large-zh"
    default_dim = 1024
    default_base_url = "http://localhost:8000/v1"
    api_key_env = "LOCAL_API_KEY"

    def encode(self, texts):
        # 返回 List[List[float]]，顺序与输入一致
        ...
```

Embedder 也支持注入实现 encode(texts) 的对象以绕过配置：

```python
from rag_data.embedding import Embedder

embedder = Embedder(settings.embedding, logger, model=my_model)
```

## 日志（logging）

logging 子包把三种日志库收敛到同一接口，调用方只依赖 LoggerAdapter。

| 文件 | 内容 |
| :--- | :--- |
| base.py | LoggerAdapter 抽象：bind 与五个级别方法 |
| factory.py | LoggerFactory：setup 全局装配与 get_logger 取用 |
| stdlib_adapter.py | StdlibLogAdapter 与 StdlibTimezoneFormatter（按 tz 格式化时间） |
| loguru_adapter.py | LoguruAdapter（惰性导入） |
| structlog_adapter.py | StructlogAdapter（惰性导入） |

```python
from rag_data.logging import LoggerFactory

LoggerFactory.setup(backend="structlog", level="INFO", timezone_name="Asia/Shanghai")
logger = LoggerFactory.get_logger()
```

后端名称：stdliblog（默认）、structlog、loguru。format 留空时优先读 pyproject 的 log_cli_format，再回退内置默认（均含完整路径与行号）。

时区 timezone 留空用系统本地，可填 UTC 或 IANA 名。

## 流程门面（facade）

facade.py 提供 RagData：一次装配 settings、logger、nlp 与 embedder，随后按流程调用。

```python
from rag_data.config import Settings
from rag_data.facade import RagData

app = RagData(Settings.load_mapping())   # 只传 settings，parser 默认 ParserAdapter
```

| 成员 | 说明 |
| :--- | :--- |
| settings、parser、logger、nlp、embedder | 构造后挂载的实例变量 |
| parser_adapter | 解析派发入口，返回 ParserAdapter 实例 |
| parse_document(path) | 委托 parser 解析单文件为纯文本 |
| chunk_document(path, text) | 委托 parser 切块，长度取自 parsing 分区 |

装配方法为模块级私有函数，构造时按分区调用：

| 方法 | 说明 |
| :--- | :--- |
| _build_logger(logging_settings) | 装配日志适配层 |
| _build_nlp(parsing_settings, logger) | 加载 spaCy，不可用则降级为 None |
| _build_embedder(embedding_settings, logger, model=None) | 装配向量化组件 |

facade 的 __all__ 仅含 RagData。

## 目录结构

```text
src/rag_data
- __init__.py        包根，仅导出当前目录直接模块的公共符号
- __about__.py       版本号（hatch 版本来源）
- config.py          分区配置模型与三类来源加载（load_mapping、load_env、load_yaml）
- exceptions.py      领域异常
- registry.py        通用注册表（Registry、Ref）
- facade.py          流程门面，RagData 与私有装配方法 _build_*
- parsers            文档解析：base、各格式子类与 parser_adapter
- embedding          向量化：base、registry、openai/qwen provider、embedder
- logging            日志适配层：base、factory 与三个适配器
```

## 待实现（TODO）

- 解析细化：PDF 双栏、Word 表格等版面级处理
- 向量化：兼容本地模型（如 sentence-transformers）的 provider
- 数据保存：向量落库流程，待存储层重新设计后补全

