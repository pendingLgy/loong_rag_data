# rag-data

历史文本导入与向量化管道：文档解析、切块、向量化，并提供关系库（历史记录与消息）、向量库（稠密检索与关键词检索）与日志的统一装配入口；不含在线召回与业务编排。

开发、构建与测试说明见 [development.md](development.md)。

## 特性

- 五类文档解析：TXT、Markdown、EPUB、PDF、Word，一种格式一个 DocumentParser 子类；继承基类并声明 backend 即自动注册
- 按格式切块：规则由各解析器自持（递归切分加四分之一重叠前缀）；spaCy 句柄由门面加载并缓存，逐级注入切块
- pydantic v2 分区配置 + pydantic-settings，实例化即校验；每个分区带 enable 开关，未启用则不初始化对应组件
- 日志：直接使用标准库 logging（无适配层、无工厂、无日志配置项），各组件用 logging.getLogger(__name__) 自取，门面不接收 logger
- 向量化：EmbedderHandler 按配置装配 embedder（openai、qwen），支持注入模型绕过配置
- 大模型：LLMHandler 按配置装配 provider（openai、qwen、deepseek），提供 chat 与 complete
- 关系库：SQLite、MySQL、PostgreSQL（基于 SQLAlchemy Core），雪花 ID 主键，启动时检查并按需建表
- 向量库：Milvus（单一 BM25 schema）与 PostgreSQL（pgvector，HNSW 或 DiskANN 加 GIN 索引）；VectorEngine 统一读写调用
- 五个注册表均为纯注册表：只登记实现类，实例化在各自的 handler 或 engine 中进行
- 包根只导出当前目录直接模块的公共符号（共 24 个）；子包按模块路径取用

## 快速开始

```python
from rag_data.config import RagSetting
from rag_data.facade import RagData

# 各分区默认关闭：未启用的组件为 None，不做初始化
app = RagData(RagSetting.load_mapping(
    parsing={'enable': True},
    embedding={'enable': True},
))
text = app.parse_document('a.md')          # 文件到纯文本
chunks = app.chunk_document('a.md', text)   # 文本到切块
vectors = app.embedder.encode(chunks)         # 文本到稠密向量
```

## 包根导出

包根 __init__.py 只导出当前目录（直接模块）的公共符号，共 23 个，不含子包与子目录内容：

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | RagSetting、ParsingSettings、EmbeddingSettings、LLMSettings、MemorySettings、ENV_PREFIX、MEMORY_ENV_PREFIX、ENV_NESTED_DELIMITER、SECTION_PARSING、SECTION_EMBEDDING |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError、LLMError |
| 通用设施 | Registry、Ref |
| 流程门面 | RagData、MemoryServer |

未纳入包根导出的符号按模块路径取用：DBSettings、VectorSettings（config），StorageError（exceptions）。

子包不参与包根导出，按模块路径取用：

```python
from rag_data.parsers import ParserHandler, register_parser
from rag_data.embedding import EmbedderHandler, BaseEmbedder
from rag_data.llm import LLMHandler, BaseLLMProvider
from rag_data.memory import MemoryHandler
from rag_data.storage.db import build_database, HistoryRecord, Message
from rag_data.storage.vector_store import VectorEngine, build_vector_store, VectorStoreResult
```

## 配置

config.py 按分区定义嵌套模型，提供三类来源的加载方法。

### 三类来源

| 方法 | 来源 | 是否读环境变量 |
| :--- | :--- | :--- |
| RagSetting.load_mapping(source=None, **overrides) | JSON 对象（分区字典） | 否 |
| RagSetting.load_env(environ=None, **overrides) | 环境变量 | 是（可显式传 environ） |
| RagSetting.load_yaml(path, **overrides) | config.yml 文件 | 否（惰性导入 PyYAML） |
| MemorySettings.load_mapping(source=None, **overrides) | JSON 对象（memory 字段字典） | 否 |
| MemorySettings.load_env(environ=None, **overrides) | 环境变量（RAG_MEMORY_ 前缀） | 是（可显式传 environ） |
| MemorySettings.load_yaml(path, **overrides) | config.yml 文件 | 否（惰性导入 PyYAML） |

优先级：overrides 高于来源数据，来源数据高于字段默认值；三类方法均以字段为单位合并。MemorySettings 的 load_mapping／load_yaml 会先补齐字段默认值，因此不读取环境变量；只有 load_env 与直接构造会读 RAG_MEMORY_ 变量，两者共用同一套解析，兼容 RAG_MEMORY__DB__BACKEND 与 RAG_MEMORY_DB__BACKEND 两种写法；记忆侧不读 .env，需要文件来源时用 load_yaml。

```python
from rag_data.config import RagSetting

settings = RagSetting()                            # 读 RAG_ 环境变量与 .env
settings = RagSetting.load_env()                   # 显式按环境变量加载
settings = RagSetting.load_mapping({'embedding': {'provider': 'qwen'}})
settings = RagSetting.load_mapping(embedding={'dim': 1024}, parsing={'max_chars': 180})
settings = RagSetting.load_yaml('config.yml')
```

```python
from rag_data.config import MemorySettings

memory = MemorySettings.load_mapping({'db': {'backend': 'pg'}})
memory = MemorySettings.load_env()                 # 读 RAG_MEMORY_ 前缀
memory = MemorySettings.load_yaml('memory.yml')     # 内容为 memory 字段的映射
```

### 环境变量

分区与字段之间用双下划线分隔，名称大写；取值按字段类型自动转换，对象与数组按 JSON 书写。

```bash
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__DIM=1024

# 记忆分区独立装配，前缀固定为 RAG_MEMORY_
RAG_MEMORY__ENABLE=true
RAG_MEMORY__DB__ENABLE=true
RAG_MEMORY__VECTOR__BACKEND=pg
RAG_MEMORY__VECTOR__PG__DATABASE=rag_data
```

分区名写错会抛 ConfigError，并列出可用分区。

### 分区与字段（括号内为默认值）

| 分区 | 字段 |
| :--- | :--- |
| parsing | enable(false)、spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | enable(false)、provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |
| memory（独立） | enable(false)、embedding、llm、db、vector；各子分区沿用自身字段与默认值 |


RagSetting 只含 parsing、embedding 两个分区；MemorySettings 独立构造（前缀 RAG_MEMORY_），自带 embedding、llm、db、vector，不再出现在 RagSetting 中。
enable 语义：分区开关；各分区默认关闭。关闭时门面不初始化对应组件（实例为 None），可避免不必要的依赖与连接。

维度一致性：embedding.dim 声明期望输出维度（默认 1024），EmbedderHandler 据此校验每条向量，不一致抛 EmbeddingError。

### 加载依赖说明

load_yaml 依赖 PyYAML，未安装时抛 OptionalDependencyError（惰性导入，不影响其余功能）。当前 pyproject.toml 未声明该依赖，生产使用前请补充。

## 文档解析与切块（parsers）

parsers 子包按扩展名把解析与切块派发到对应 DocumentParser 子类；派发逻辑在 parser_handler.py 的 ParserHandler，实现类登记在 registry_parser.py。

```text
文件 → parse_document → 纯文本 → chunk_document → 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| registry_parser.py | BUILTIN_PARSERS、register_parser、resolve_parser_class | 纯注册表，只登记实现类，不实例化 |
| parser_handler.py | ParserHandler、PARSER_CLASSES | 按扩展名派发；构造时预实例化全部解析器并复用 |
| base.py | DocumentParser | 抽象基类：SUFFIXES、backend 与 parse、chunk 契约，以及递归切块与重叠前缀 |
| nlp/spacy.py | DEFAULT_MODEL、load_nlp、clear_cache、ensure_sentence_boundaries | spaCy 句柄加载：惰性导入、按模型名缓存、失败也缓存 |
| nlp/__init__.py | 汇总导出上述四个符号 | 供 parsers 与门面共用 |
| txt.py | TxtParser | .txt，整篇递归切块 |
| markdown.py | MarkdownParser | .md、.markdown，按行首井号分章后递归切块 |
| epub.py | EpubParser | .epub，按 spine 提取章节（排除目录页）后按井号分章递归切块 |
| pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取并过滤页眉页脚后整篇递归切块 |
| word.py | WordParser | .docx，python-docx 提取段落，整篇递归切块 |

ParserHandler 的实例方法：

| 方法 | 作用 |
| :--- | :--- |
| resolve_parser(path, max_chars=None, safe_max_chars=None) | 取该格式解析器实例；配置与缓存一致时复用，否则临时新建 |
| parse_document(path) | 解析单个文档并返回纯文本 |
| chunk_document(path, text, nlp=None, max_chars=None, safe_max_chars=None) | 按扩展名派发到该格式的 chunk |
| parser_classes（属性） | 本次派发使用的解析器类集合 |
| instances（属性） | 已预实例化的解析器副本 |
| get_instance(path) | 按扩展名取缓存实例，不新建 |

单块上限 max_chars 与安全上限 safe_max_chars 由 parsing 分区注入（默认 500 与 2000）；句柄模型默认 zh_core_web_md，加载时关闭 tagger、ner 与 attribute_ruler 以省开销。

子包根提供默认处理器实例的绑定方法，便于直接调用：

```python
from rag_data.parsers import parse_document, chunk_document, resolve_parser
from rag_data.parsers import register_parser, available_parsers

```

自定义解析器：继承 DocumentParser 并声明 backend 即自动登记（也可显式 register_parser）。

```python
from rag_data.parsers import DocumentParser

class MyParser(DocumentParser):
    backend = 'my'                        # 声明名字，类定义时自动注册
    SUFFIXES = frozenset({'.my'})

    def parse(self, path): ...             # 文件返回纯文本
    def chunk(self, text, nlp=None): ...   # 文本返回切块列表
```

## 向量化（embedding）

实现由配置 embedding.provider 选择，内置 openai 与 qwen；实例化在 EmbedderHandler 调用时进行。

```text
EmbedderHandler → resolve_embedder → Registry → OpenAIEmbedder
                                     └ QwenEmbedder
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseEmbedder 抽象：声明 backend 即自动登记 |
| registry_embedder.py | 纯注册表：register_embedder、resolve_embedder 等（只注册不实例化） |
| openai_embedder.py | OpenAIEmbedder（官方 openai SDK） |
| qwen_embedder.py | QwenEmbedder（继承 OpenAI，DashScope 兼容模式） |
| embedder_handler.py | EmbedderHandler：按配置解析并实例化 embedder，统一批次与维度校验 |

抽象类可声明的类属性：backend、default_model、default_dim、default_base_url、api_key_env、max_batch_size。

### 内置 embedder

| embedder | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY | 10 |

EmbedderHandler 的批次策略：取 embedding.batch_size 与 embedder 的 max_batch_size 的较小者，因此 qwen 的 10 条上限自动生效；编码结果按 embedding.dim 校验。

### 自定义 embedder：继承即注册

```python
from rag_data.embedding import BaseEmbedder

class LocalEmbedder(BaseEmbedder):
    backend = 'local'                # 声明名字，类定义时自动注册
    default_model = 'bge-large-zh'
    default_dim = 1024
    default_base_url = 'http://localhost:8000/v1'
    api_key_env = 'LOCAL_API_KEY'

    def encode(self, texts):
        # 返回 List[List[float]]，顺序与输入一致
        ...
```

EmbedderHandler 也支持注入实现 encode(texts) 的对象以绕过配置：

```python
from rag_data.embedding import EmbedderHandler

embedder = EmbedderHandler(settings.embedding, model=my_model)
```

## 大模型（llm）

实现由配置 llm.provider 选择，内置 openai、qwen、deepseek；实例化在 LLMHandler 调用时进行。

```text
LLMHandler → resolve_llm_provider → Registry → OpenAILLMProvider
                                          ├ QwenLLMProvider
                                          └ DeepSeekLLMProvider
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseLLMProvider 抽象：声明 backend 即自动登记 |
| registry_llm.py | 纯注册表：register_llm、resolve_llm_provider 等（只注册不实例化） |
| openai_llm.py | OpenAILLMProvider（官方 openai SDK） |
| qwen_llm.py | QwenLLMProvider（继承 OpenAI，DashScope 兼容模式） |
| deepseek_llm.py | DeepSeekLLMProvider（继承 OpenAI） |
| llm_handler.py | LLMHandler：按配置解析并实例化 provider，统一调用入口 |

抽象类可声明的类属性：backend、default_model、default_base_url、api_key_env。

### 内置 provider

| provider | 默认模型 | 默认地址 | Key 环境变量 |
| :--- | :--- | :--- | :--- |
| openai | gpt-4o-mini | api.openai.com 的 v1 | OPENAI_API_KEY |
| qwen | qwen-plus | DashScope compatible-mode | DASHSCOPE_API_KEY |
| deepseek | deepseek-chat | api.deepseek.com 的 v1 | DEEPSEEK_API_KEY |

### LLMHandler

| 成员 | 说明 |
| :--- | :--- |
| provider（属性） | 注入的 model 优先，否则按配置解析并实例化并缓存 |
| chat(messages, **overrides) | 单轮对话，messages 为 role 与 content 组成的字典列表 |
| complete(prompt, system=", **overrides) | 单条用户消息完成一次生成，可选系统提示 |

### 自定义 provider：继承即注册

```python
from rag_data.llm import BaseLLMProvider

class LocalLLMProvider(BaseLLMProvider):
    backend = 'local'
    default_model = 'my-model'
    default_base_url = 'http://localhost:8000/v1'
    api_key_env = 'LOCAL_API_KEY'

    def chat(self, messages, **overrides):
        # 返回回复文本
        ...
```

LLMHandler 同样支持注入实现 chat(messages, **overrides) 的对象以绕过配置：

```python
from rag_data.llm import LLMHandler

llm = LLMHandler(settings.llm, model=my_model)
reply = llm.complete('用一句话概括', system='你是一名编辑')
```

## 关系库（storage 下的 db）

承载历史记录与消息；基于 SQLAlchemy Core，后端由配置 db.backend 选择。

```text
build_database → SQLiteDB / MySQLDB / PGDB → SQLDatabase（SQLAlchemy Core）
```

| 文件 | 内容 |
| :--- | :--- |
| registry_db.py | 纯注册表：register_db、resolve_db、BUILTIN_DBS（sqlite、mysql、pg） |
| base.py | BaseDB 抽象接口与自动登记（声明 backend 即注册） |
| db_engine.py | SQLDatabase 通用实现（SQLAlchemy Core）与 build_database 装配函数 |
| schema.py | 表结构（METADATA、HISTORY_TABLE、MESSAGES_TABLE）、建表 SQL 与数据对象 HistoryRecord、Message |
| snowflake.py | 雪花 ID 生成器（SnowflakeGenerator 与模块级 next_id） |
| sqlite_db.py | SQLiteDB：路径必填，数据一律落盘 |
| pg_db.py | PGDB：psycopg3 优先、psycopg2 回退 |
| mysql_db.py | MySQLDB：PyMySQL 优先、mysql-connector-python 回退 |

### 表结构

| 表 | 主键 | 关键列 |
| :--- | :--- | :--- |
| history | id BIGINT（雪花 ID） | memory_id BIGINT（可空，有索引）、old_memory、new_memory、event、created_at、updated_at、is_deleted、actor_id、role |
| messages | id BIGINT（雪花 ID） | session_scope（索引，与 created_at 组成复合索引）、role、content、name、created_at |

启动时检查表是否存在，缺失的才创建（已存在不动）；建表 SQL 同时以字符串常量形式提供，便于人工建表或核对：

```python
from rag_data.storage.db import CREATE_SQL, create_table_sql, HistoryRecord, Message

print(create_table_sql('sqlite'))   # 取该方言的建表 SQL（文本）
```

### 接口与装配

| 方法 | 说明 |
| :--- | :--- |
| add_history(record) / batch_add_history(records) | 写入历史记录，返回补齐 id 的对象或对象列表 |
| get_history(memory_id) | 按 memory_id 取历史，按时间升序 |
| save_messages(messages, keep_last=10) | 按 session_scope 归组写入，仅保留每会话最近 keep_last 条 |
| get_last_messages(session_scope, limit=10) | 取会话最近消息 |
| reset() / close() | 清空重建 / 释放引擎；支持 with 上下文管理 |

```python
from rag_data.config import DBSettings
from rag_data.storage.db import build_database, HistoryRecord

store = build_database(DBSettings(enable=True, backend='sqlite', sqlite={'path': 'data\rag.db'}))
store.add_history(HistoryRecord(memory_id=1001, new_memory='v1', event='ADD'))
store.close()
```

自定义后端：继承 BaseDB 并声明 backend，即被注册表收录。

```python
from rag_data.storage.db import BaseDB

class MyDB(BaseDB):
    backend = 'mydb'
    # 实现 add_history、batch_add_history、get_history、save_messages、get_last_messages、reset、close
```

## 向量库（storage 下的 vector_store）

承载稠密向量与元数据，支持关键词检索；后端由配置 vector.backend 选择。

```text
build_vector_engine → VectorEngine → VectorStoreBase
                              ├ MilvusVectorStore（单一 BM25 schema）
                              └ PGVectorStore（pgvector 表与索引）
```

| 文件 | 内容 |
| :--- | :--- |
| registry_vector.py | 纯注册表：register_vector_store、resolve_vector_store、BUILTIN_STORES（milvus、pg） |
| base.py | VectorStoreBase 抽象、VectorStoreResult 结果对象、MetricType 度量类型 |
| vector_engine.py | build_vector_store、VectorEngine 与 build_vector_engine 装配函数 |
| milvus_store.py | MilvusVectorStore：集合与索引，单一 BM25 schema |
| pg_store.py | PGVectorStore：建表与索引，DDL 辅助函数 |

### 集合与 schema

| 后端 | 建集合或建表 | 索引 |
| :--- | :--- | :--- |
| milvus | 单一 schema：id（主键，VARCHAR 512）、vectors（FLOAT_VECTOR）、metadata（JSON）、text（VARCHAR，开启分析器）、sparse（稀疏向量，由 BM25 函数按 text 自动生成） | vector_index（AUTOINDEX）；sparse_index（SPARSE_INVERTED_INDEX，BM25） |
| pg | CREATE TABLE IF NOT EXISTS：id（TEXT 主键）、vector vector(维度)、payload JSONB | HNSW（余弦，默认启用）或 DiskANN（需 vectorscale 扩展，缺失自动回退 HNSW）；全文 GIN 索引（表达式与 keyword_search 一致） |

维度较大（大于 2000）时跳过 DiskANN；建表语句以函数形式提供，便于核对：

```python
from rag_data.storage.vector_store.pg_store import table_ddl, hnsw_index_ddl, text_index_ddl

print(table_ddl('autotest_vector', 8))
print(hnsw_index_ddl('autotest_vector'))
print(text_index_ddl('autotest_vector'))
```

### VectorEngine

| 分组 | 成员 |
| :--- | :--- |
| 属性 | store、backend、collection_name、vector_size |
| 写入 | insert(vectors, payloads=None, ids=None) 返回写入条数、update(vector_id, vector=None, payload=None)、delete(vector_id) |
| 检索 | search(query, vectors, top_k=None, filters=None)、keyword_search(query, top_k=None, filters=None)、get(vector_id)、list(filters=None, top_k=None) |
| 维护 | list_cols()、col_info()、reset()、delete_col()、close()，支持 with 上下文管理 |

引擎在写入与检索前校验维度与批量长度（payloads 与 ids 需与 vectors 等长），不一致抛 StorageError；默认检索条数来自 vector.top_k。

```python
from rag_data.config import VectorSettings
from rag_data.storage.vector_store import build_vector_engine

engine = build_vector_engine(VectorSettings(
    enable=True, backend='pg', collection='docs', vector_size=8, top_k=3,
    pg={'host': 'localhost', 'database': 'rag_data'},
))
engine.insert([[0.1]*8], payloads=[{'text_lemmatized': 'hello'}])
hits = engine.search('hello', [0.1]*8)
engine.close()
```

自定义后端：继承 VectorStoreBase 并声明 backend，即被注册表收录。

## 日志（logging）

不做任何日志配置：库内只使用标准库 logging，不调用 basicConfig，也不提供日志相关的配置分区与入口。

| 约定 | 说明 |
| :--- | :--- |
| 模块级 logger | 各模块用 logging.getLogger(__name__) 自取，组件与门面都不接收、不透传 logger |
| 门面构造 | RagData(settings)、MemoryServer(settings)，均不接收 logger |
| 级别与格式 | 由调用方自行配置 basicConfig 或自建 handler，库不介入 |

```python
import logging

from rag_data import RagData
from rag_data.config import RagSetting

logging.basicConfig(level=logging.INFO)

app = RagData(RagSetting.load_mapping(parsing={'enable': True}))
app.parse_document('a.md')
```

## 流程门面（facade）

facade.py 提供 RagData：一次装配各分区组件，随后按流程调用。

```python
from rag_data.config import RagSetting
from rag_data.facade import RagData

# 仅启用解析与向量化；未启用的分区实例为 None
app = RagData(RagSetting.load_mapping(
    parsing={'enable': True},
    embedding={'enable': True},
))
app.parser        # 解析器始终装配，固定为 ParserHandler 实例
app.embedder      # 已装配；记忆侧的 embedding、llm、db、vector 由 MemoryServer 自行装配
text = app.parse_document('a.md')
chunks = app.chunk_document('a.md', text)
```

| 成员 | 说明 |
| :--- | :--- |
| settings、parser、nlp、embedder | 构造后挂载的实例变量；未启用的分区为 None |
| parser_handler（属性） | 解析派发入口，返回 ParserHandler 实例 |
| parse_document(path) | 委托 parser 解析单文件为纯文本 |
| chunk_document(path, text) | 委托 parser 切块，长度取自 parsing 分区 |

装配方法为模块级函数：_build_* 为私有实现，构造时按分区调用，均受对应分区的 enable 门控：

| 方法 | 说明 |
| :--- | :--- |
| _build_nlp(parsing_settings) | 加载 spaCy；parsing.enable 为 False 或不可用时返回 None |
| _build_embedder(embedding_settings, model=None) | 装配向量化组件；embedding.enable 为 False 时返回 None |
| _build_db(db_settings) | 按 db.backend 装配关系库；db.enable 为 False 时返回 None |
| _build_vector(vector_settings) | 按 vector.backend 装配向量引擎；vector.enable 为 False 时返回 None |
| _build_llm(llm_settings, model=None) | 装配大模型入口；llm.enable 为 False 时返回 None |
| _build_memory(memory_settings, embedder, llm, db, vector) | 装配 MemoryHandler；memory.enable 为 False 时返回 None |

RagData 只装配 parsing、embedding；MemoryServer 只接收独立的 MemorySettings，内部装配 memory 下的 embedding、llm、db、vector，并持有 MemoryHandler。两者的构造签名都是 (settings)：组件日志由各组件按模块名自取，门面既不接收也不透传。facade 的 __all__ 含 RagData 与 MemoryServer。


## 目录结构

```text
src/rag_data
- __init__.py        包根，仅导出当前目录直接模块的公共符号
- __about__.py       版本号（hatch 版本来源）
- config.py          分区配置模型与三类来源加载（load_mapping、load_env、load_yaml）
- exceptions.py      领域异常（含 StorageError、LLMError）
- registry.py        通用注册表（Registry、Ref）
- facade.py          流程门面 RagData、MemoryServer，私有装配方法 _build_*
- parsers            文档解析：base、各格式子类、parser_handler、registry_parser
- embedding          向量化：base、registry_embedder、openai 与 qwen embedder、embedder_handler
- llm                大模型：base、registry_llm、openai、qwen、deepseek 实现、llm_handler
- storage            存储子包，见下

src/rag_data/storage/db
- base.py            BaseDB 抽象接口
- registry_db.py     纯注册表（BUILTIN_DBS）
- db_engine.py       SQLDatabase 与 build_database
- schema.py          表结构、建表 SQL、HistoryRecord 与 Message
- snowflake.py       雪花 ID 生成器
- sqlite_db.py       SQLiteDB
- pg_db.py           PGDB
- mysql_db.py        MySQLDB

src/rag_data/storage/vector_store
- base.py            VectorStoreBase、VectorStoreResult、MetricType
- registry_vector.py 纯注册表（BUILTIN_STORES）
- vector_engine.py   build_vector_store、VectorEngine、build_vector_engine
- milvus_store.py    MilvusVectorStore（单一 BM25 schema）
- pg_store.py        PGVectorStore（建表与索引）
```

## 测试

```text
tests
- conftest.py                            路径注入、公共夹器与 --integration 选项
- test_config.py                         配置三类来源与各分区
- test_facade.py                         装配方法与 facade 公开契约
- test_facade_ragdata.py                 RagData 构造、解析与切块
- test_facade_memoryserver.py            MemoryServer 装配与流程代理
- test_parsers_base.py                   切分、重叠前缀与递归切块
- test_parsers_dispatch.py               派发与解析器注册表
- test_parsers_txt.py 等                各格式解析与切块
- test_embedding.py  test_llm.py         向量化与大模型注册表、handler
- test_db.py                             关系库注册表、表结构、SQLite 端到端
- test_vector_store.py                   向量库注册表、DDL、引擎，以及真实服务集成
- test_integration_qwen_epub_milvus.py  真实 EPUB 到 Milvus 的端到端链路
- test_integration_facade_ragdata.py  RagData 集成：真实解析与切块
- test_integration_facade_memoryserver.py  MemoryServer 集成：四条流程
```

规模：402 项（单元 237、集成 165）。集成用例分布在 test_db.py、test_vector_store.py、test_integration_qwen_epub_milvus.py、test_integration_facade_ragdata.py、test_integration_facade_memoryserver.py；默认由 addopts 的 -m not integration 反选，且这五个文件在模块级检查 --integration 参数，未提供时整体跳过并打印 [SKIP INFO]；测试用库名、集合名与表名统一以 autotest_ 开头，删除动作只在该前缀下执行。

```bash
pytest -rs                       # 只跑单元用例，末尾附跳过清单
pytest --integration -m integration -v -s   # 跑集成用例（需真实服务与凭据）
```

跳过信息由各测试模块内的局部辅助函数输出，包含依赖名、目标地址与原始异常。集成测试的连接信息、密钥、后端选择与样例路径统一写在测试目录下的 conftest.py（顶部的各分区 JSON 变量），用例不再读取环境变量；各分区 JSON 会实例化为 EmbeddingSettings、LLMSettings、ParsingSettings 与各存储 Settings 对象，供用例直接读属性。
## 待实现（TODO）

- 解析细化：PDF 双栏、Word 表格等版面级处理
- 向量化：兼容本地模型（如 sentence-transformers）的 embedder（已有 embedding extra，尚未实现后端）
- 数据保存：向量落库流程与 facade 对 vector、db、llm 的装配编排
- 依赖声明：RagSetting.load_yaml 需要 PyYAML，pyproject.toml 尚未声明，建议补充依赖或新增 yaml extra
- openai extra 当前锁定为 openai==3.26.0，建议核对版本是否存在并按需放宽

