# rag-data 项目结构

> 版本 0.1.0，Python 3.8 及以上，src 布局，包根仅导出当前目录直接模块的 23 个符号。

本文档描述仓库的物理结构与模块职责，是阅读代码与二次开发的导航图。设计取舍与开发计划见 plan 目录的 DEVELOPMENT_PLAN.md，记忆层设计见 MEMORY_DESIGN.md，使用方式见 README.md，开发、构建与测试说明见 development.md。

> 存储层（关系库与向量库）与记忆层（memory）均已实现并接入门面开关；向量落库与批量导入编排仍由调用方自行组合 parser、embedder 与 vector。

---

## 1. 概览

rag-data 同时提供两条能力线：**解析向量化**（把文件解析为分块并编码为稠密向量，也支持字符串直接向量化）与**对话记忆**（把消息抽取为事实条目并做增删改查）。两条线共用下层能力，但由两个独立门面分别装配：`RagData` 负责解析，`MemoryServer` 负责记忆，二者配置也彼此独立。整体分层，层间只依赖下层，方向单一。

| 层 | 包 | 职责 |
| :--- | :--- | :--- |
| 配置层 | config.py | 解析侧 `RagSetting`（三分区）与独立 `MemorySettings`，三类来源加载，各分区带 enable 开关 |
| 契约层 | exceptions.py | 领域异常 |
| 通用设施 | registry.py、logging、nlp | 通用注册表、日志适配层、spaCy 句柄加载与缓存 |
| 领域服务 | parsers、embedding、llm、memory | 解析切块、向量化、大模型调用、记忆四流程 |
| 存储层 | storage/db、storage/vector_store | 关系库（历史与消息）与向量库（稠密与关键词检索） |
| 门面层 | facade.py | 两个门面 `RagData`、`MemoryServer` 与私有装配方法 `_build_*` |
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
│  ├─ DEVELOPMENT_PLAN.md   分阶段开发计划
│  └─ MEMORY_DESIGN.md      记忆层（memory）架构设计
├─ src/
│  └─ rag_data/
│     ├─ __about__.py      版本号（hatch 版本来源）
│     ├─ __init__.py       包根，仅导出当前目录直接模块的 25 个公共符号
│     ├─ py.typed          类型标记（PEP 561）
│     ├─ config.py         RagSetting 三分区 + 独立 MemorySettings 与三类来源加载
│     ├─ exceptions.py     领域异常层次
│     ├─ registry.py       通用注册表（Registry、Ref）
│     ├─ facade.py         门面 RagData、MemoryServer 与私有装配方法 _build_*
│     ├─ nlp/
│     │  ├─ __init__.py          汇总导出 DEFAULT_MODEL、load_nlp、clear_cache、ensure_sentence_boundaries
│     │  └─ spacy.py             spaCy 句柄加载：惰性导入、按模型名缓存（失败也缓存）与清空
│     ├─ parsers/
│     │  ├─ __init__.py          汇总导出、默认处理器的绑定方法与句柄加载再导出
│     │  ├─ base.py              DocumentParser 基类与递归切块原语（句柄加载已抽出）
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
│     ├─ memory/
│     │  ├─ __init__.py          汇总导出 MemoryBase、MemoryHandler 与数据对象
│     │  ├─ base.py              MemoryBase 抽象：add、search、update、delete 四条契约
│     │  ├─ handler.py           MemoryHandler：流程编排、结果组装与 DEFAULT_* 常量
│     │  ├─ stores.py            MemoryStore：批量写入、历史审计与失败降级
│     │  ├─ scope.py             作用域构造与校验（纯函数，无外部依赖）
│     │  ├─ prompts.py           抽取提示词常量与渲染函数
│     │  └─ models.py            MemoryItem、AddResult、SearchResult 等数据对象
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
│     └─ util/               空目录（仅 __pycache__），待清理
└─ tests/
├─ conftest.py                     路径注入、settings 夹具、--integration 选项、集成配置解析（JSON 拆为各 Settings 对象）
   ├─ test_config.py        RagSetting 与 MemorySettings 的分区、三类来源与 enable
   ├─ test_facade.py        模块级装配方法与 facade 公开契约
   ├─ test_facade_ragdata.py RagData 构造、解析与切块
   ├─ test_facade_memoryserver.py MemoryServer 装配与流程代理
   ├─ test_parsers_*.py     各格式契约、解析与切块、派发与注册表
   ├─ test_embedding.py     向量化注册表、内置 embedder 与 EmbedderHandler
   ├─ test_llm.py           大模型注册表、内置 provider 与 LLMHandler
   ├─ test_db.py            关系库注册表、表结构与 SQLite 端到端、MySQL 与 PG 集成
   ├─ test_vector_store.py  向量库注册表、DDL、引擎，以及真实服务集成
   ├─ test_integration_qwen_epub_milvus.py  真实 EPUB 到 Milvus 端到端
   ├─ test_integration_facade_ragdata.py    RagData 集成：真实解析与切块
   └─ test_integration_facade_memoryserver.py  MemoryServer 集成：四条流程
```

memory 子包无独立测试文件：四条流程由 `test_facade_memoryserver.py` 中的替身用例覆盖装配与代理，纯函数（`scope.py`、`stores.py`、`prompts.py`）的用例尚未补齐，详见第 13 节。

---

## 3. 分层与依赖方向

依赖自上而下，无环；较低层不反向导入较高层。

```text
__init__（包根，仅导出直接模块符号）
     ▲
facade（两个门面：RagData 解析、MemoryServer 记忆）
     ├── config            RagSetting 与 MemorySettings 及三类来源加载
     ├── nlp               spaCy 句柄加载与缓存（parsers 与 facade 共用）
     ├── parsers           文档解析与切块（ParserHandler）
     ├── embedding         向量化（EmbedderHandler）
     ├── llm               大模型（LLMHandler）
     ├── memory            记忆四流程（MemoryHandler → MemoryStore）
     ├── storage/db        关系库（build_database）
     └── storage/vector_store  向量库（build_vector_engine）
```

关键约定：

1. 可选依赖（spacy、openai、PyYAML、pymilvus、psycopg、PyMySQL、SQLAlchemy、EbookLib、pdfplumber、python-docx）一律惰性使用，未安装时包仍可正常导入
2. 实现选择走注册表：声明 backend 即自动登记，新增实现不改动装配代码
3. 注册表只登记实现类，实例化在各自的 handler 或 engine 中进行
4. 包根只导出当前目录直接模块的公共符号，不跨模块、不列子包
5. config 的 YAML 来源由 _import_yaml 惰性导入 PyYAML，缺失时抛 OptionalDependencyError
6. memory 只依赖抽象基类与 engine（BaseDB、VectorEngine、EmbedderHandler、LLMHandler），不 import 具体方言实现，也不自建连接
7. nlp 不反向依赖 parsers 或 facade：只提供句柄，不含业务逻辑

---

## 4. 基础模块

### 4.1 __about__.py

仅保存 __version__，由 pyproject.toml 的 [tool.hatch.version] 读取。

### 4.2 __init__.py（包根）

只导出当前目录直接模块的公共符号，共 23 个，不含子包，也不做跨模块重导出：

```python
from rag_data.config import (
    ENV_NESTED_DELIMITER, ENV_PREFIX, MEMORY_ENV_PREFIX,
    SECTION_EMBEDDING, SECTION_PARSING,
    EmbeddingSettings, LLMSettings,
    MemorySettings, ParsingSettings, RagSetting,
)
from rag_data.exceptions import ConfigError, RagDataError
from rag_data.facade import MemoryServer, RagData
from rag_data.registry import Ref, Registry
```

未纳入导出的符号（`SECTION_DB`、`SECTION_VECTOR`、`DBSettings`、`VectorSettings`、`StorageError`）按模块路径取用，分组见 7.2 节。

> 与旧版差异：`Settings` 已改名为 `RagSetting`，并新增 `MemorySettings`、`MEMORY_ENV_PREFIX`、`MemoryServer`。

### 4.3 config.py

职责：以嵌套模型承载配置，实例化即校验；提供三类来源的加载方法与一组私有工具函数。解析侧与记忆侧是**两套独立配置**：`RagSetting` 只含 parsing、embedding，`MemorySettings` 自带 embedding、llm、db、vector 并使用独立前缀。

```python
class ParsingSettings(BaseModel): ...      # enable、spacy_model、max_chars、safe_max_chars
class EmbeddingSettings(BaseModel): ...    # enable、provider、model、batch_size、dim、api_key、base_url、timeout
class LLMSettings(BaseModel): ...          # enable、provider、model、temperature、max_tokens、api_key、base_url、timeout
class SQLiteSettings / MySQLSettings / PGSettings
class MilvusSettings / PGVectorSettings
class VectorSettings(BaseModel): ...       # enable、backend、collection、vector_size、top_k、metric_type、milvus、pg
class DBSettings(BaseModel): ...           # enable、backend、sqlite、mysql、pg
class MemorySettings(BaseSettings): ...    # enable、embedding、llm、db、vector（前缀 RAG_MEMORY_）
class RagSetting(BaseSettings): ...        # parsing、embedding
```

常量：`SECTION_PARSING`、`SECTION_EMBEDDING`、`MODULE_SECTIONS`、`SECTION_MODEL_CLASSES`、`ENV_PREFIX`（`RAG_`）、`MEMORY_ENV_PREFIX`（`RAG_MEMORY_`）、`ENV_NESTED_DELIMITER`（`__`）。

`SECTION_LLM`、`SECTION_DB`、`SECTION_VECTOR`、`SECTION_MEMORY`、`SECTION_LOGGING` 均已删除：前三个分区改由 `MemorySettings` 直接承载，不再作为 `RagSetting` 的分区键；`memory` 也不再是分区（它是独立的设置类）。因此 `MODULE_SECTIONS` 与 `SECTION_MODEL_CLASSES` 都只含 parsing、embedding，把 `memory` 作为 `RagSetting` 的分区键传入会抛 `ConfigError`。

两个设置类的类方法（语义对等）：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| `RagSetting.load_mapping(source=None, **overrides)` | JSON 对象（分区字典） | 否 |
| `RagSetting.load_env(environ=None, **overrides)` | 环境变量（`RAG_` 前缀） | 是 |
| `RagSetting.load_yaml(path, **overrides)` | YAML 文件 | 否 |
| `MemorySettings.load_mapping(source=None, **overrides)` | JSON 对象（字段字典） | 否 |
| `MemorySettings.load_env(environ=None, **overrides)` | 环境变量（`RAG_MEMORY_` 前缀） | 是 |
| `MemorySettings.load_yaml(path, **overrides)` | YAML 文件 | 否 |

`MemorySettings` 是 `BaseSettings`，直接实例化会读环境变量；`load_mapping` 与 `load_yaml` 先用 `_with_model_defaults` 按 `model_fields` 默认值补齐全字段再叠加来源数据，借此屏蔽环境读取。直接实例化时由 `_MemoryEnvSource` 读取 `RAG_MEMORY_` 前缀，与 `load_env` 共用同一套解析（前缀以单下划线结尾、嵌套用双下划线，原生前缀解析会切错键），故两种下划线写法行为一致；记忆侧不读 `.env`，需要文件来源时用 `load_yaml`。`extra=forbid` 使未知字段（含已移除的 `nlp`、`threshold`）直接抛 `ValidationError`。

模块级函数：

| 函数 | 作用 |
| :--- | :--- |
| _load_env_overrides(environ=None) | 把 `RAG_` 前缀环境变量组装为分区字典 |
| _filter_sections(data) | 只保留已知分区，忽略无关的 `RAG_` 变量 |
| _load_memory_env_overrides(environ=None) | 剥离 `RAG_MEMORY_` 前缀组装字段字典，兼容双下划线与单下划线两种写法 |
| _validate_sections(*groups) | 校验代码硬编码中的分区名合法 |
| _merge_sections(base, overrides) | 按分区逐字段合并，实现覆盖 |
| _with_defaults(layered) | 用分区模型默认值补齐字段，屏蔽环境变量 |
| _with_model_defaults(model_cls, layered) | 用 `model_fields` 默认值补齐顶层键，供 `MemorySettings` 屏蔽环境 |
| _as_mapping(source, loader) | 校验配置源为字典映射 |
| _as_yaml_mapping(path) | 读取 YAML 并校验为字典映射 |
| _assign_nested(data, parts, value) | 按路径写入嵌套字典 |
| _parse_env_value(raw) | JSON 对象与数组按 JSON 解析 |
| _import_yaml() | 惰性导入 PyYAML，缺失时抛 OptionalDependencyError |

优先与合并：overrides 高于来源数据，来源数据高于字段默认值。

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

memory 不新增异常类型，沿用上表：构造期缺组件抛 `ConfigError`，入口参数非法抛 `DataError`。

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

### 4.6 日志（stdlib logging）

不提供日志配置：库内只使用标准库 logging，不调用 basicConfig，也没有日志分区、适配层与工厂。

| 约定 | 说明 |
| :--- | :--- |
| 模块级 logger | 各模块用 logging.getLogger(__name__) 自取，不接收也不透传 logger |
| 门面构造 | (settings) 不接收 logger |
| 级别与格式 | 由调用方自行配置；库内日志把字段拼为 key=value 后交给 logger 输出 |

### 4.7 nlp/（spaCy 句柄）

从解析器基类抽出，供解析器与两个门面共用；只做句柄加载与缓存，不含切块逻辑。

| 符号 | 说明 |
| :--- | :--- |
| `DEFAULT_MODEL` | 默认模型名 `zh_core_web_md` |
| `load_nlp(model_name=DEFAULT_MODEL)` | 加载句柄；不可用时返回 `None` 并告警 |
| `clear_cache()` | 清空缓存（含失败记录），供测试或运行时重建 |
| `ensure_sentence_boundaries(nlp)` | 缺 parser 与 sentencizer 时补一个 sentencizer |

行为要点：

1. 惰性 `import spacy`，未安装则缓存失败哨兵并返回 `None`；
2. 加载时 `disable=[tagger, ner, attribute_ruler]`，完成后补句边界；
3. 成功与失败都写入缓存，避免逐次重试拖慢批量导入；
4. 文件与第三方包同名，但 Python 3 只做绝对导入，函数内 `import spacy` 仍解析到第三方包。

解析器基类已移除句柄加载实现与 `load_nlp`、`clear_cache` 类方法，仅按旧路径再导出这三个符号以兼容既有引用；parsers 子包也额外导出 `DEFAULT_MODEL`、`load_nlp`、`clear_cache`。

---

## 5. 领域服务层

### 5.1 parsers（解析与切块）

```text
文件 → parse_document → 文本 → chunk_document → 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| registry_parser.py | BUILTIN_PARSERS、register_parser、resolve_parser_class、available_parsers | 纯注册表，只登记不实例化 |
| parser_handler.py | ParserHandler、PARSER_CLASSES | 构造时预实例化全部解析器；按扩展名派发 |
| base.py | DocumentParser | 抽象基类：SUFFIXES、backend 与 parse、chunk 契约；递归切块与重叠前缀 |
| txt.py | TxtParser | .txt，整篇递归切块 |
| markdown.py | MarkdownParser | .md、.markdown，按行首井号分章后递归切块 |
| epub.py | EpubParser | .epub，按 spine 提取章节（排除目录页）后按井号分章递归切块 |
| pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取、过滤页眉页脚后整篇递归切块 |
| word.py | WordParser | .docx，python-docx 提取段落，整篇递归切块 |

ParserHandler：构造时可替换解析器类集合与默认长度；索引在构造时建立并做后缀唯一性断言。

| 方法 | 作用 |
| :--- | :--- |
| resolve_parser(path, max_chars=None, safe_max_chars=None) | 取该格式实例；长度与缓存一致时复用，不一致则临时新建 |
| parse_document(path) | 解析单个文档并返回纯文本 |
| chunk_document(path, text, nlp=None, max_chars=None, safe_max_chars=None) | 按扩展名派发到该格式的 chunk |
| get_instance(path) | 取缓存实例，不新建 |
| parser_classes、instances（属性） | 解析器类集合、已预实例化的实例副本 |

切块由各解析器自持：基类提供递归切分与四分之一重叠前缀；单块上限 max_chars 与安全上限 safe_max_chars 为构造参数，由 parsing 分区注入（默认 500 与 2000）。句柄由装配层通过 `nlp.spacy` 加载后逐级注入，解析器只消费注入的句柄。

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

EmbedderHandler(settings, model=None)：注入的 model 优先且绕过配置；provider 首次访问时实例化并缓存。批次策略：取 embedding.batch_size 与 embedder 的 max_batch_size 的较小者；编码结果按 embedding.dim 校验，不一致抛 EmbeddingError。

memory 复用该组件：`MemorySettings` 自带 `embedding` 分区，`MemoryServer` 据此装配独立的嵌入器，与解析侧互不影响。

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

LLMHandler(settings, model=None) 提供 provider（属性）、chat(messages, **overrides) 与 complete(prompt, system, **overrides)；注入的 model 优先且绕过配置。

memory 复用该组件做事实抽取：`MemorySettings.llm` 是独立的一份 `LLMSettings`，与解析侧的 llm 分区互不影响。

### 5.4 memory（记忆四流程）

```text
MemoryHandler → MemoryStore → VectorEngine（向量读写）
      │                    └ BaseDB（历史与消息）
      ├ LLMHandler（事实抽取）
      └ EmbedderHandler（文本向量化）
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| base.py | MemoryBase | 抽象契约：add、search、update、delete |
| handler.py | MemoryHandler、DEFAULT_* | 流程编排与结果组装；四条流程均同步阻塞 |
| stores.py | MemoryStore、build_payload、split_payload、memory_hash、keyword_text、now_iso、to_vector_id、to_vector_id_int | 向量库与关系库的薄封装，统一 id 转换、payload 布局与失败降级 |
| scope.py | validate_entity_id、strip_identity_keys、build_scope、session_scope、validate_search_params、validate_query、escape_scope_value、ENTITY_PARAMS、IDENTITY_KEYS | 作用域构造与校验，纯函数无外部依赖 |
| prompts.py | ADDITIVE_EXTRACTION_PROMPT、AGENT_CONTEXT_SUFFIX、PAST_MESSAGE_TRUNCATION_LIMIT、generate_additive_extraction_prompt | 抽取提示词与渲染函数 |
| models.py | MemoryItem、AddResult、SearchResult、UpdateResult、DeleteResult | pydantic 数据对象（extra 为 ignore） |

MemoryHandler(settings, embedder=None, llm=None, db=None, vector=None)：四个组件（embedder、llm、db、vector）均可注入，便于测试；构造期调用 `_validate()` 校验四项硬依赖（vector、embedder、llm、db），缺一即抛 `ConfigError`；随后建立 `MemoryStore`，不发起任何网络或数据库请求。

| 公共方法 | 职责 |
| :--- | :--- |
| add(messages, **scope) | 归一消息 → 召回既有记忆去重 → 大模型抽取事实 → 嵌入并写向量库、历史与消息 |
| search(query, **filters) | 语义与关键词双路召回，融合打分后按阈值过滤、条数截断并格式化 |
| update(memory_id, text=None, metadata=None) | 合并 payload；正文变化时才重算向量；写向量库并记 UPDATE 历史 |
| delete(memory_id) | 向量库物理删除，历史表软删（`is_deleted=1`）留审计 |

检索与写入的默认值不放入配置，由 `handler.py` 的模块常量承接，避免配置项与代码默认值双份漂移：

| 常量 | 值 | 用途 |
| :--- | :--- | :--- |
| DEFAULT_THRESHOLD | 0.1 | 语义召回的相似度下限 |
| DEFAULT_TOP_K | 20 | 检索默认返回条数 |
| DEFAULT_CONTEXT_TOP_K | 10 | 抽取前召回的既有记忆条数 |
| DEFAULT_KEYWORD_WEIGHT | 0.5 | BM25 分在融合时的权重 |
| DEFAULT_HISTORY_MESSAGES | 10 | 写路径给模型的历史消息条数 |

关键约定：

1. 主键沿用 `storage.db` 的雪花 ID（BIGINT），向量库 id 为 `str(memory_id)`，转换集中在 `stores.py`；
2. payload 固定核心键（正文、词形文本、hash、时间戳、作用域），其余键原样保留为业务元数据；`hash` 取正文 MD5 用于去重；
3. 写入顺序为先向量库后历史：向量库是检索的事实来源，历史表是审计；
4. 降级边界：写历史失败、存消息失败、关键词检索不支持、抽取响应非 JSON 均只告警不中断；嵌入与向量写入失败则抛 `EmbeddingError` 或 `StorageError`；
5. 身份键（user_id、agent_id、run_id、actor_id）只从显式参数进入，metadata 中的同名字段一律剥离。

---

## 6. 存储层

### 6.1 storage/db（关系库）

```text
build_database → SQLiteDB / MySQLDB / PGDB → SQLDatabase（SQLAlchemy Core）
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseDB 抽象接口：声明 backend 即自动登记，无状态、无构造参数 |
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

build_database(settings) 按 settings.backend 惰性导入并构造方言实现：sqlite 要求 settings.sqlite.path 非空，否则抛 ConfigError。

数据对象：HistoryRecord 与 Message 为 pydantic 模型（extra 为 ignore），支持传入字典构造。

memory 通过 `MemoryStore` 复用这些方法（批量写历史失败回退逐条、消息按会话归组），不新开连接。

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

build_vector_engine(settings) 按 settings.backend 组装参数：集合名、维度、度量类型固定注入；连接参数按后端从 milvus 或 pg 分区取，留空字段不下传（交由实现自身默认值）。

memory 通过 `MemoryStore` 复用 `insert`、`update`、`delete`、`get`、`search`、`keyword_search`；后端未实现关键词检索时 `MemoryStore.supports_keyword` 为假，融合权重自动降为 0。

---

## 7. 门面层与公共 API

### 7.1 facade.py

模块级装配方法：`_build_*` 为私有实现，各对应一个分区或设置类，且通常受 enable 门控（为 False 时返回 None）：

| 方法 | 流程 | 使用者 |
| :--- | :--- | :--- |
| _build_nlp(parsing_settings) | 加载 spaCy；parsing.enable 为 False 或不可用时返回 None | RagData |
| _build_embedder(embedding_settings, model=None) | 装配向量化组件 | RagData |
| _build_llm(llm_settings, model=None) | 装配大模型入口 | MemoryServer |
| _build_db(db_settings) | 按 db 分区装配关系库 | MemoryServer |
| _build_vector(vector_settings) | 按 vector 分区装配向量引擎 | MemoryServer |
| _build_memory(memory_settings, embedder, llm, db, vector) | 装配 MemoryHandler | MemoryServer |

两个门面分担不同职责：`RagData` 只装配 parsing、embedding；`MemoryServer` 只接收独立的 `MemorySettings`，内部装配 memory 下的 embedding、llm、db、vector，最后构造 `MemoryHandler`。两个门面的构造签名都是 `(settings)`：组件日志由各组件按模块名自取，门面既不接收也不透传 logger。

RagData 是一次装配、逐流程调用的解析门面：

```python
RagData(settings: RagSetting)
```

| 成员 | 说明 |
| :--- | :--- |
| settings、embedder、db、vector、llm | 构造后挂载的实例变量；对应分区关闭时为 None |
| parser_handler（属性） | 解析派发入口，返回 ParserHandler 实例 |
| parse_document(path) | 委托 parser 解析单文件为纯文本 |
| chunk_document(path, text) | 委托 parser 切块，长度取自 parsing 分区 |

MemoryServer 是记忆服务门面，代理四条流程：

```python
MemoryServer(settings: MemorySettings)
```

| 成员 | 说明 |
| :--- | :--- |
| settings、logger、embedder、db、vector、llm | 构造后挂载的实例变量；未注入且对应分区关闭时为 None |
| handler（属性 memory） | MemoryHandler 实例；memory.enable 为 False 时为 None |
| add / search / update / delete | 四条流程代理，未启用时经 _require() 抛 ConfigError |

解析器不做门控：parser 始终装配，保证 parse_document 与 chunk_document 在任何配置下可用。

facade 的 __all__ 含 RagData 与 MemoryServer。

### 7.2 包根导出（23 个符号）

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | RagSetting、ParsingSettings、EmbeddingSettings、LLMSettings、MemorySettings、ENV_PREFIX、ENV_NESTED_DELIMITER、MEMORY_ENV_PREFIX、SECTION_PARSING、SECTION_EMBEDDING |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError、LLMError |
| 通用设施 | Registry、Ref |
| 流程门面 | RagData、MemoryServer |

未纳入包根导出的实现按需从子包引入，例如 parsers 的 ParserHandler、nlp 的 load_nlp、embedding 的 EmbedderHandler、llm 的 LLMHandler、memory 的 MemoryHandler、storage 的 build_database 与 build_vector_engine、storage 的 StorageError。

---

## 8. 数据流

### 8.1 文件向量化

```text
1. RagData(settings)                装配 settings、logger、nlp、embedder（按 enable 门控）
2. app.parse_document(path)         解析文件为纯文本
3. app.chunk_document(path, text)   按格式切块（ParserHandler 派发）
4. app.embedder.encode(chunks)      分块编码为稠密向量
5. engine.write(...)                写入与检索（调用方自行编排）
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

### 8.4 记忆四流程

```text
MemoryServer(settings)                          按各分区 enable 装配 embedding、llm、db、vector
    ├─ add(messages, user_id=...)             抽取事实并写入向量库、历史与消息
    ├─ search(query, user_id=...)             双路召回并融合排序
    ├─ update(memory_id, text=...)            改正文或元数据，必要时重算向量
    └─ delete(memory_id)                      向量库物理删除、历史表软删
```

`MemoryServer` 与 `RagData` 可共用同一个 `EmbedderHandler` 实例，也可各自独立创建；两者配置不同源，互不覆盖。

---

## 9. 配置项

解析侧 `RagSetting` 的分区与字段（括号内为默认值）：

| 分区 | 字段 |
| :--- | :--- |
| parsing | enable(false)、spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | enable(false)、provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |

记忆侧 `MemorySettings`（独立类，非 `RagSetting` 分区，前缀 `RAG_MEMORY_`）：

| 字段 | 说明 |
| :--- | :--- |
| enable(false) | 记忆分区开关；关闭时 `MemoryServer.memory` 为 None |
| embedding | EmbeddingSettings：enable(false)、provider、model、batch_size、dim、api_key、base_url、timeout |
| llm | LLMSettings：enable(false)、provider、model、temperature、max_tokens、api_key、base_url、timeout |
| db | DBSettings：enable(false)、backend(sqlite)、sqlite/mysql/pg 连接参数 |
| vector | VectorSettings：enable(false)、backend(milvus)、collection(rag_data)、vector_size(1024)、top_k(5)、metric_type(COSINE)、milvus/pg 连接参数 |

`MemorySettings` 自带 embedding 分区：记忆侧的嵌入器独立于解析侧。

两个设置类不持有 `logger`：各组件用 `logging.getLogger(__name__)` 自取，级别与格式由调用方自行配置。检索与写入的数值参数（threshold、top_k、context_top_k、keyword_weight、history_messages）也不在配置中，由 `memory` 包的模块常量承接，见 5.4 节。

三类来源：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| RagSetting.load_mapping | JSON 对象（分区字典） | 否 |
| RagSetting.load_env | 环境变量 | 是（可显式传 environ） |
| RagSetting.load_yaml | 配置文件 | 否（惰性导入 PyYAML） |
| MemorySettings.load_mapping | JSON 对象（字段字典） | 否 |
| MemorySettings.load_env | 环境变量 | 是（可显式传 environ） |
| MemorySettings.load_yaml | 配置文件 | 否（惰性导入 PyYAML） |

环境变量命名：解析侧前缀 `RAG_`，记忆侧前缀 `RAG_MEMORY_`；分区与字段大写、双下划线分隔。

```bash
RAG_EMBEDDING__DIM=1024

RAG_MEMORY__ENABLE=true
RAG_MEMORY__DB__ENABLE=true
RAG_MEMORY__VECTOR__BACKEND=pg
```

记忆侧兼容 `RAG_MEMORY__X` 与 `RAG_MEMORY_X` 两种写法（剥离前缀时可选消费一个下划线）。取值按字段类型转换，对象与数组按 JSON 书写；解析侧分区名非法抛 `ConfigError`，记忆侧未知字段因 `extra=forbid` 抛 `ValidationError`。

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

记忆层不在上表：`MemoryHandler` 是单一实现，无 backend 概念，扩展方式是继承 `MemoryBase` 后替换 `MemoryServer` 持有的 handler。

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

下表为各文件的 `def test_` 静态计数（不含 pytest 参数化展开，也未运行 pytest 核对）：

```text
tests（静态统计 402 项 def test_；其中模块级标记 integration 的五个文件合计 165 项）
├─ conftest.py                     路径注入、settings 夹具、--integration 选项、集成配置解析
├─ test_config.py           56 项  RagSetting 与 MemorySettings 的分区、三类来源与 enable
├─ test_facade.py          16 项  模块级装配方法、门控与 facade 公开契约
├─ test_facade_ragdata.py  14 项  RagData 构造、解析派发与切块
├─ test_facade_memoryserver.py 5 项  MemoryServer 装配、流程代理与门控
├─ test_parsers_base.py     16 项  分句、重叠前缀与递归切块
├─ test_parsers_dispatch.py 19 项  派发、注册表与预实例化复用
├─ test_parsers_txt.py      11 项  TxtParser 契约、解析与切块
├─ test_parsers_markdown.py 11 项  MarkdownParser
├─ test_parsers_epub.py     10 项  EpubParser
├─ test_parsers_pdf.py      10 项  PdfParser
├─ test_parsers_word.py     12 项  WordParser
├─ test_embedding.py        21 项  注册表、内置 embedder 与 EmbedderHandler
├─ test_llm.py              36 项  注册表、内置 provider 与 LLMHandler
├─ test_db.py               63 项  注册表、表结构、SQLite 端到端、MySQL 与 PG 集成
├─ test_vector_store.py     73 项  注册表、DDL、引擎、真实 Milvus 与 PG 集成
├─ test_integration_qwen_epub_milvus.py  6 项  真实 EPUB 到 Milvus 端到端
├─ test_integration_facade_ragdata.py  12 项  RagData 集成：真实解析与切块
└─ test_integration_facade_memoryserver.py  11 项  MemoryServer 集成：四条流程
```

集成用例分布在 test_db.py、test_vector_store.py、test_integration_qwen_epub_milvus.py、test_integration_facade_ragdata.py、test_integration_facade_memoryserver.py：这五个文件在模块级声明 pytestmark 为 integration，并检查 --integration 参数，未提供时以 allow_module_level 整体跳过并打印 [SKIP INFO]；默认 addopts 另加 -m not integration 反选。

跳过信息由各测试模块内的局部辅助函数输出，包含依赖名、目标地址与原始异常，配合 -rs 可查看跳过清单。集成用例的配置统一写在测试目录的 conftest.py（顶部的各分区 JSON 变量）：各分区实例化为 Settings 对象，其余从对应 *_CONFIG 字典读取，测试不再读取环境变量；数据库名与集合名统一以 autotest_ 开头，删除动作只在该前缀下执行。

memory 尚无独立测试文件：装配与四条流程代理由 test_facade_memoryserver.py 的替身用例覆盖，`scope.py`、`stores.py`、`prompts.py` 的纯函数用例待补。

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
| dev | pytest、pytest-cov、mypy、ruff |

环境装配：pyproject.toml 定义 default 与 rag_data_dev 两套 hatch 环境，按 features 自动安装上述 extras；rag_data_dev 额外包含 sqlalchemy、pg、mysql、milvus、word、pdf、openai。

memory 未引入新依赖：它复用 embedding、llm 与 storage 三层的现成包；spaCy 句柄位于 nlp 子包，仍属 parsers extra 覆盖范围。

注意：PyYAML 用于两个设置类的 `load_yaml`，当前未在 pyproject.toml 声明，生产使用前建议补充依赖或新增 yaml extra。

---

## 13. 待补全

| 位置 | 事项 |
| :--- | :--- |
| parsers | 版面级解析细化，如 PDF 双栏、Word 表格 |
| embedding | 兼容本地模型的 embedder（如 sentence-transformers 后端） |
| memory | 四条流程的异步版本、delete_all、reset 均不在本期；实体图谱、时序与衰减策略明确排除 |
| memory | 检索与写入参数已下沉为模块常量；若确需运行时调优，再考虑回配置并同步测试 |
| memory | 无独立测试文件，scope、stores、prompts 的纯函数用例待补 |
| 数据保存 | 向量落库流程与批量导入编排，需调用方自行组合 parser、embedder 与 vector |
| facade | 是否把记忆的检索与写入纳入更多流程方法，而非仅由 MemoryServer 代理 |
| config | pyproject.toml 未声明 PyYAML |
| 包导入 | `import rag_data` 经 facade 间接依赖 memory、storage 的顶层导入，未装 SQLAlchemy 时会抛 ModuleNotFoundError，与可选依赖的惰性原则不符 |
| 子包导出 | 各子包 __init__.py 的重导出策略尚未完全统一 |
| 遗留目录 | src 下的 rag_data/util 为空目录（仅 __pycache__），待清理 |


