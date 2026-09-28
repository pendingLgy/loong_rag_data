# rag-data

历史文本数据导入与向量化管道。范围聚焦离线批量导入链路：文档解析、两阶段语义切块、实体抽取、向量化、批量写入向量库；不含在线召回模块。

## 特性

- 两阶段语义切块：版面结构切分后进行句级切分并保留重叠
- pydantic v2 数据模型与 pydantic-settings 配置，边界即校验
- 日志适配层：标准库 logging、loguru、structlog 三后端可切换
- 向量库抽象：Milvus 实现与内存实现，接口可插拔
- 嵌入模型抽象：OpenAI 与通义千问内置实现，配置切换，自带 HTTP 调用无额外依赖

## 环境与依赖管理

本项目使用 hatch 管理构建与运行环境，使用 uv 解析与安装依赖。

### 环境要求

- Python 3.8 及以上
- uv（依赖管理）与 hatch（构建环境管理）
- 核心依赖：pydantic、pydantic-settings

### 安装 uv 与 hatch

```bash
# uv：依赖解析、虚拟环境与锁定
pip install uv
# macOS/Linux 亦可使用官方脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# hatch：构建与多环境管理
pip install hatch
```

### 使用 uv 管理依赖

```bash
# 创建虚拟环境，默认位于 .venv
uv venv

# 安装核心依赖，并以可编辑模式挂载当前项目
uv pip install -e .

# 按需安装可选依赖
uv pip install -e ".[milvus,loguru]"

# 安装开发依赖
uv pip install -e ".[dev]"

# 或一次性安装全部可选与开发依赖
uv pip install -e ".[parsers,embedding,milvus,qdrant,loguru,structlog,dev]"
```

激活虚拟环境：

```bash
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

依赖发生变更后重新安装，或解析为锁定文件以获得可复现环境：

```bash
# 增量安装新增依赖
uv pip install -e ".[dev]"

# 将 pyproject.toml 解析为锁定文件
uv pip compile pyproject.toml --all-extras -o requirements.lock

# 按锁定文件精确同步环境（会移除多余包）
uv pip sync requirements.lock
```

说明：uv 直接读取 pyproject.toml 的 dependencies 与 optional-dependencies，无需额外配置文件。

### 使用 hatch 管理环境与构建

hatch 环境直接读取 pyproject.toml 的依赖声明，已定义两套：

| 环境 | 名称 | 包含依赖 |
| :--- | :--- | :--- |
| 默认环境 | default | 核心依赖 + dev（pytest、mypy、ruff）|
| 开发环境 | rag_data_dev | parsers、embedding、milvus、qdrant、loguru、structlog、dev |

```bash
# 创建环境
hatch env create
hatch env create rag_data_dev

# 查看已有环境
hatch env show

# 在默认环境中执行命令
hatch run pytest

# 通过脚本执行，脚本定义见 pyproject.toml 的 [tool.hatch.envs.*.scripts]
hatch run test
hatch run types
hatch run check
hatch run format

# 在开发环境中执行命令（前缀为环境名）
hatch run rag_data_dev:test
hatch run rag_data_dev:types
hatch run rag_data_dev:check

# 进入交互式 shell
hatch shell
hatch shell rag_data_dev

# 构建 sdist 与 wheel，产物在 dist/
hatch build

# 发布到 PyPI
hatch publish
```

### 常用组合

| 目标 | 命令 |
| :--- | :--- |
| 克隆后快速开始 | uv venv，再 uv pip install -e ".[dev]" |
| 运行测试 | uv run pytest 或 hatch run pytest |
| 类型检查 | hatch run types 或 hatch run rag_data_dev:types |
| 代码风格 | hatch run check 或 hatch run rag_data_dev:check |
| 构建发行包 | hatch build |

## 快速开始

```python
from rag_data import RagData

app = RagData.create()                # 一次装配全部流程
app.init_collection()                 # 流程：建集合
written = app.ingest(["README.md"])   # 流程：批量导入
print(written)
```

## 公共 API

本包对外只暴露一个入口文件，即包根 __init__.py；外部调用无需关心内部子包路径：

```python
import rag_data
from rag_data import Settings, IngestionPipeline, InMemoryVectorStore
```

导出内容按用途分组：

| 分组 | 导出符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、StorageSettings、ChunkingSettings、EmbeddingSettings、NLPSettings、LoggingSettings、load_env_overrides、ENV_PREFIX、ENV_NESTED_DELIMITER |
| 数据模型 | DocumentChunk、MemoryRecord、MilvusRecord、QueryHit、resolve_record_class、DEFAULT_RECORD_CLASS_PATH |
| 导入管道 | IngestionPipeline、parse_document、build_semantic_chunks、extract_entities |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 存储 | BaseVectorStore、InMemoryVectorStore、register_store、register_backend、available_backends、is_registered、resolve_store、create_store |
| 流程门面 | RagData、init_collection、ingest、build_settings、build_logger、build_store、build_nlp、build_embedder、build_pipeline、build_record_class、DEFAULT_TOP_N |
| 日志适配 | LoggerAdapter、configure_logging、get_logger |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、SchemaMismatchError、StoreError、EmbeddingError |

Milvus 实现因依赖 pymilvus，未纳入顶层导出，按需从子包引入：

```python
from rag_data.storage.milvus_store import MilvusVectorStore
```


## 流程门面

facade.py 把各流程的调用方法集中封装，调用方不必再逐个导入子模块：

```python
from rag_data import RagData

app = RagData.create()                # 装配配置、日志、存储、向量化、管道
app.init_collection()                 # 流程：建集合
app.ingest(["a.md", "b.md"], user_id="tenant-a")  # 流程：批量导入
app.ingest_file("c.md")               # 流程：导入单文件
app.query(vector, top_n=5)                           # 流程：按向量检索
app.query(vector, filters={"user_id": "tenant-a"})  # 流程：按扩展字段过滤
app.close()                           # 流程：释放资源
```

也可以按需只装配单个流程：

| 方法 | 流程 |
| :--- | :--- |
| build_settings(source) | 加载配置（可指定 JSON 文件或目录）|
| build_logger(settings) | 装配日志适配层 |
| build_store(settings, logger) | 按 storage.backend 装配向量库 |
| build_nlp(settings, logger) | 加载 spaCy 句柄，不可用则降级为 None |
| build_embedder(settings, logger, model) | 装配向量化组件 |
| build_pipeline(...) | 装配导入管道 |
| init_collection(source) | 一键建表 |
| ingest(paths, source, user_id) | 一键导入 |
| app.query(vector, top_n, user_id, filters) | 检索；filters 支持基类与扩展字段 |

```python
from rag_data import build_settings, build_logger, build_store

settings = build_settings("config.json")
logger = build_logger(settings)
store = build_store(settings, logger)
```

支持上下文管理，退出时自动释放资源：

```python
from rag_data import RagData

with RagData.create() as app:
    app.ingest(["doc.md"])
```


## 存储后端注册与切换

存储后端采用**注册表**机制：继承 BaseVectorStore 并声明 backend 名称即自动注册，
用户只需在配置里写 storage.backend 就能切换，无需改动装配代码。

### 内置后端

| 后端名 | 实现 | 说明 |
| :--- | :--- | :--- |
| memory | InMemoryVectorStore | 内存实现，零依赖，适合测试与本地验证 |
| milvus | MilvusVectorStore | Milvus 实现，惰性导入 pymilvus |

```python
from rag_data import available_backends

available_backends()   # ["memory", "milvus"]
```

### 切换后端：只改配置

```json
{ "storage": { "backend": "milvus", "milvus_uri": "localhost:19530", "milvus_db": "default" } }
```

```python
from rag_data import RagData

app = RagData.create("config.json")   # store 按 storage.backend 自动装配
```

环境变量亦可：

```bash
RAG_STORAGE__BACKEND=milvus
```

### 自定义后端：继承即注册

```python
from rag_data import BaseVectorStore


class SqliteVectorStore(BaseVectorStore):
    backend = "sqlite"          # 声明后端名，类定义时自动注册

    def __init__(self, settings=None, logger=None, record_class=None):
        ...

    def ensure_collection(self) -> None: ...
    def upsert(self, records): ...
    def query(self, vector, top_n, filters=None): ...
```

注册后即可在配置中直接使用：

```json
{ "storage": { "backend": "sqlite" } }
```

### 手动注册（可选）

类不在导入路径中时，可显式注册：

```python
from rag_data import register_store, register_backend

register_store("custom", "myapp.stores.CustomStore")      # 点分路径，惰性导入

@register_backend("another")                        # 装饰器，直接持有类对象
class AnotherStore(BaseVectorStore):
    ...
```

### 构造函数约定

注册表统一以关键字传入 settings 与 logger，自定义实现请保持一致签名：

```python
def __init__(self, settings=None, logger=None, record_class=None): ...
```

| 参数 | 含义 |
| :--- | :--- |
| settings | 已校验的配置对象 |
| logger | LoggerAdapter 实例 |
| record_class | 记录类，由 models.record_class 解析 |

未注册的后端名会抛 ConfigError，并在信息中列出当前可用后端。


## 嵌入模型注册与切换

嵌入模型同样采用注册表：继承 BaseEmbeddingProvider 并声明 backend 名称即自动注册，
通过配置 embedding.provider 切换，无需改动装配代码。

### 内置 provider

| provider | 实现 | 默认模型 | 默认维度 | API Key 环境变量 |
| :--- | :--- | :--- | :--- | :--- |
| openai | OpenAIEmbeddingProvider | text-embedding-3-small | 1536 | OPENAI_API_KEY |
| qwen | QwenEmbeddingProvider | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY |

qwen 复用 OpenAI 的请求格式（DashScope 兼容模式），差异仅在默认模型、接口地址、
API Key 环境变量与单次批量上限（10 条）。

```python
from rag_data import available_embedding_providers

available_embedding_providers()   # ["openai", "qwen"]
```

### 切换 provider：只改配置

```json
{ "embedding": { "provider": "qwen", "model": "qwen3.7-text-embedding" } }
```

```python
from rag_data import RagData

app = RagData.create(overrides={
    "storage": { "vector_dim": 1024 },
    "embedding": { "provider": "qwen" },
})
```

环境变量亦可：

```bash
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__MODEL=qwen3.7-text-embedding
DASHSCOPE_API_KEY=sk-xxx
RAG_STORAGE__VECTOR_DIM=1024
```

### embedding 配置项

| 字段 | 默认值 | 说明 |
| :--- | :--- | :--- |
| provider | openai | 注册表中的 provider 名 |
| model | 空 | 模型名，留空用 provider 默认模型 |
| dim | 0 | 期望维度，0 表示用 provider 默认维度 |
| batch_size | 128 | 每批文本条数，与 provider 上限取较小者 |
| api_key | 空 | 留空时读取 provider 约定的环境变量 |
| base_url | 空 | 留空时用 provider 默认地址，可指向自建网关 |
| timeout | 60.0 | 单次请求超时秒数 |

输出维度需与 storage.vector_dim 一致，不一致会抛 EmbeddingError。

### 自定义 provider：继承即注册

```python
from rag_data import BaseEmbeddingProvider


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

注册后即可在配置中直接使用：

```json
{ "embedding": { "provider": "local" } }
```

### 手动注册与注入

```python
from rag_data import register_embedding, register_embedding_provider

register_embedding("custom", "myapp.embedding.CustomProvider")   # 点分路径，惰性导入

@register_embedding_provider("another")                     # 装饰器，直接持有类对象
class AnotherProvider(BaseEmbeddingProvider):
    ...
```

测试或接入已有模型时，可直接注入实现了 encode(texts) 的对象，绕过配置：

```python
from rag_data import Embedder

embedder = Embedder(settings, logger, model=my_model)
```

注入对象可选提供 max_batch_size 以限制单次条数。


## Milvus 建表

建表权由**记录类**掌握：Milvus 后端使用 MilvusRecord，它默认硬编码基础字段，子类覆盖即可自定义表结构。
Milvus 的建表调用委托给记录类的 build_collection_schema；MemoryRecord 只描述数据，不含建表逻辑。

### 默认表结构（MilvusRecord 硬编码）

| 字段 | 类型 | 参数 | 索引 |
| :--- | :--- | :--- | :--- |
| id | VARCHAR | max_length=64, is_primary | Primary Key |
| text_payload | VARCHAR | max_length=65535 | - |
| vector | FLOAT_VECTOR | dim 取自 storage.vector_dim | HNSW + COSINE |
| entities | ARRAY | element_type=VARCHAR, max_capacity=64, max_length=256 | INVERTED |
| created_at | DOUBLE | - | STL_SORT |
| metadata | JSON | - | -（扩展字段容器）|

### 覆盖建表：子类自定义表结构

继承 MilvusRecord 并覆盖 build_collection_schema，即可追加或替换字段：

```python
from rag_data import MilvusRecord


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

配置里指向该类即可生效：

```json
{ "models": { "record_class": "myapp.models.TenantRecord" } }
```

建表完成后，集合实际字段会被自动识别：新增列既参与写入平铺，也参与按字段过滤。

### 配置驱动提升列

不改代码也可把扩展字段提升为独立列：

```json
{
  "models": {
    "record_class": "myapp.models.TenantRecord",
    "promoted_fields": [
      { "name": "user_id", "type": "VARCHAR", "max_length": 128 }
    ]
  }
}
```

### 建表调用与行为

```python
from rag_data import RagData

app = RagData.create("config.json")
app.init_collection()      # 建表、建索引并 load；重复调用幂等
```

| 场景 | 行为 |
| :--- | :--- |
| 集合不存在 | 委托记录类生成 schema，建表建索引后 load |
| 集合已存在 | 跳过建表，校验向量维度，一致则直接 load |
| 维度不一致 | 抛 SchemaMismatchError，提示重建集合 |
| 维度非法 | 记录类在生成 schema 时抛 ConfigError |
| pymilvus 未安装 | 抛 OptionalDependencyError，给出安装指引 |

### 扩展字段落库策略

| 场景 | 落库位置 |
| :--- | :--- |
| 普通扩展字段 | metadata JSON 列，表结构不变 |
| 子类建表新增的列 | 独立物理列，写入时平铺 |
| 配置提升的列 | 独立物理列，写入时平铺 |

已平铺的字段不再重复写入 metadata，避免同一字段存两份。

## 数据模型与扩展

MemoryRecord 是基础数据模型，供内存实现与通用场景使用；MilvusRecord 继承它并额外掌握 Milvus 建表，
是配置的默认记录类。两者字段一致，都支持继承与运行时扩展。

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| id | str | 主键，由内容哈希生成，幂等写入依据 |
| text_payload | str | 原始记忆文本块 |
| vector | List[float] | 语义向量 |
| entities | List[str] | 实体列表 |
| created_at | float | 创建或录入时间戳 |

扩展方式一：子类声明类型化字段，获得校验与 IDE 提示。

定义好子类后，只需在配置里填写类路径，写入与读回都会使用该类。

```python
from typing import List

from rag_data import MilvusRecord


class TenantRecord(MilvusRecord):
    user_id: str
    tags: List[str] = []


record = TenantRecord(
    id="m1", text_payload="t", vector=[0.1], created_at=1.0,
    user_id="u1", tags=["重要"],
)
```

在配置中指定（JSON）：

```json
{
  "models": { "record_class": "myapp.models.TenantRecord" }
}
```

```bash
# 或使用环境变量
RAG_MODELS__RECORD_CLASS=myapp.models.TenantRecord
```

未配置时默认使用 rag_data.models.MilvusRecord；它继承 MemoryRecord，内存后端同样适用。

扩展方式二：直接传入未声明字段，无需定义子类。

```python
from rag_data import MemoryRecord


record = MemoryRecord(
    id="m1", text_payload="t", vector=[0.1], created_at=1.0,
    user_id="u1",
)
record.extra_fields   # {"user_id": "u1"}
record.to_row()       # 含全部字段的扁平字典，供存储层落库
```

扩展字段如何落库：

记录写入前会先转成存储行 —— 基类字段平铺成列，扩展字段序列化进 metadata JSON 列。
存储行编解码由记录类自身提供（MemoryRecord 定义，MilvusRecord 继承），两个后端因此共享同一套落库行为。

```python
from rag_data import MemoryRecord
from rag_data.storage.memory_store import InMemoryVectorStore

store = InMemoryVectorStore()
store.upsert([MemoryRecord(
    id="m1", text_payload="t", vector=[0.1, 0.2], created_at=1.0,
    user_id="u1", tags=["重要"],   # 扩展字段
)])

store.get_row("m1")
# {"id": "m1", "text_payload": "t", "vector": [0.1, 0.2], "entities": [], "created_at": 1.0,
#  "metadata": '{"user_id": "u1", "tags": ["重要"]}'}

store.get_record("m1").extra_fields
# {"user_id": "u1", "tags": ["重要"]}   扩展字段完整还原
```

检索时按扩展字段过滤，内存实现直接匹配字段，Milvus 实现转为 metadata 的 JSON 路径条件：

```python
store.query(vector, top_n=5, filters={"user_id": "u1"})
# Milvus 布尔表达式：metadata[\"user_id\"] == \"u1\"
```

编解码 API（记录类方法，MemoryRecord 定义，MilvusRecord 继承）：

| 方法 | 作用 |
| :--- | :--- |
| record.to_storage_row(promoted) | 记录到存储行，扩展字段进 metadata |
| Cls.from_storage_row(row, promoted) | 存储行还原为记录，扩展字段回填 |
| Cls.flatten_row(row, promoted) | 存储行摊平为单层字典 |
| Cls.row_matches(row, filters, promoted) | 按字段匹配，涵盖基类与扩展字段 |
| Cls.storage_fields() | 存储行平铺列名，含 metadata |


## 配置

配置按模块分区，默认读取环境变量，也可在代码中硬编码，模型定义在 config.py。

优先级由高到低：

| 优先级 | 来源 | 示例 |
| :--- | :--- | :--- |
| 1 | 代码硬编码 | Settings.load(storage={"backend": "milvus"}) |
| 2 | 环境变量与 .env | RAG_STORAGE__BACKEND=milvus |
| 3 | 字段默认值 | 各分区模型内声明 |

两者同时存在时以代码硬编码为准，未硬编码的字段由环境变量填充。

分区与主要字段：

| 分区 | 主要字段 |
| :--- | :--- |
| storage | backend、milvus_uri、milvus_db、collection_name、vector_dim、metric、index_type |
| models | record_class、promoted_fields |
| chunking | max_chars、overlap_sents |
| embedding | model、batch_size |
| nlp | spacy_model |
| logging | backend、level、json |

### 加载方式

```python
from rag_data.config import Settings

settings = Settings()                     # 默认读取环境变量（含 .env）
settings = Settings.load()                # 同上，显式表达按环境变量加载
settings = Settings.load(use_env=False)   # 忽略环境变量：仅用字段默认值，硬编码仍生效
```

### 代码硬编码

适合测试、脚本与嵌入式场景，优先级最高，可只覆盖部分字段：

```python
settings = Settings.load(
    storage={"backend": "milvus", "vector_dim": 1024},
    chunking={"max_chars": 180},
)
settings = Settings.load({"storage": {"backend": "memory"}})   # 也可直接传字典
```

门面与一键入口同样支持：

```python
from rag_data import RagData, build_settings

settings = build_settings(storage={"backend": "milvus"})
app = RagData.create(overrides={"storage": {"backend": "milvus"}})
```

### 环境变量

分区与字段之间用双下划线分隔，名称大写：

```bash
RAG_STORAGE__BACKEND=milvus
RAG_STORAGE__VECTOR_DIM=512
RAG_LOGGING__LEVEL=DEBUG
RAG_LOGGING__JSON=true
RAG_MODELS__RECORD_CLASS=myapp.models.TenantRecord
RAG_MODELS__PROMOTED_FIELDS=[{"name": "user_id", "type": "VARCHAR"}]
```

取值按字段类型自动转换，对象与数组按 JSON 书写。
分区名或字段名写错会抛 ConfigError，并列出可用值。

## 目录结构

```text
src/rag_data
- __init__.py        公共入口，统一导出全部 API
- config.py          pydantic-settings 配置
- models.py          pydantic 数据模型
- exceptions.py      领域异常
- facade.py          流程门面，一站式封装各流程调用
- ingestion          解析、切块、实体抽取、导入编排
- embedding          向量化接口、provider 注册表与 openai、qwen 实现
- storage            向量库接口与实现
- logging            日志适配层
```

## 待实现（TODO）

以下复杂逻辑以 TODO 标注，待人工补全：

- 文档解析：PDF 与 Word，基于 unstructured 或 MinerU
- 向量化：按 metric 决定是否做 L2 归一化
- 向量化：兼容 embedding 接口的本地模型（如 sentence-transformers）provider
- 实体抽取：默认加载 spaCy 模型
- Milvus 存储：连接、建表、写入与检索

## 开发

推荐使用 hatch 环境，无需手动激活虚拟环境：

```bash
# 默认环境（含 dev 工具）
hatch run test      # pytest
hatch run types     # mypy src tests
hatch run check     # ruff check src tests
hatch run format    # ruff format src tests

# 开发环境（叠加全部可选依赖）
hatch run rag_data_dev:test
hatch run rag_data_dev:check
```

若使用 uv 直接管理依赖：

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

### 集成测试

单元测试不访问外部服务；调用真实嵌入模型与向量库的用例集中在 tests/test_integration_qwen_milvus.py，
标记为 integration，默认不执行，需人工准备凭据与服务后显式触发。

覆盖链路：Qwen 嵌入、Milvus 建表、写入、检索、过滤、幂等。

#### 前置条件

| 依赖 | 说明 |
| :--- | :--- |
| pymilvus | 向量库客户端，随可选依赖安装 |
| API Key | Qwen（DashScope）的 API Key |
| Milvus 服务 | 可访问的实例，本地容器或云端均可 |

#### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| DASHSCOPE_API_KEY | 是 | 无 | Qwen 的 API Key |
| RAG_IT_API_KEY | 否 | 无 | 覆盖上一项，便于切换多套凭据 |
| RAG_IT_MILVUS_URI | 否 | http://127.0.0.1:19530 | Milvus 地址 |
| RAG_IT_COLLECTION | 否 | rag_data_it_qwen | 集合名 |
| RAG_IT_MODEL | 否 | qwen3.7-text-embedding | 嵌入模型 |
| RAG_IT_KEEP | 否 | 空 | 置 1 则跑完保留集合，默认删除 |
| RAG_IT_DIM | 否 | 1024 | 向量维度，需与模型输出一致 |

凭据只在运行时通过环境变量提供，代码与仓库中不落任何密钥。

#### 执行命令

Windows PowerShell：

```powershell
uv pip install -e ".[dev,milvus]"

$env:DASHSCOPE_API_KEY = "sk-xxx"
$env:RAG_IT_MILVUS_URI = "http://127.0.0.1:19530"

uv run pytest -m integration -v -s
```

macOS 或 Linux：

```bash
uv pip install -e ".[dev,milvus]"

export DASHSCOPE_API_KEY=sk-xxx
export RAG_IT_MILVUS_URI=http://127.0.0.1:19530

uv run pytest -m integration -v -s
```

使用 hatch：

```bash
# 默认环境（需已装 pymilvus）
hatch run test -m integration -v -s

# 开发环境（叠加全部可选依赖，含 pymilvus）
hatch run rag_data_dev:test -m integration -v -s
```

只跑单条用例：

```bash
uv run pytest -m integration -v -s tests/test_integration_qwen_milvus.py::test_query_returns_most_relevant_first
```

不加 -m integration 时这些用例会被自动排除，见 pyproject.toml 中 addopts 的 -m not integration；
因此日常执行 pytest 不会触发任何外部调用。

#### 用例清单

| 用例 | 验证内容 |
| :--- | :--- |
| test_provider_is_qwen | provider 为 qwen，模型与维度取自配置 |
| test_encode_returns_configured_dimension | 编码结果维度等于 vector_dim |
| test_related_text_is_more_similar | 语义相近文本得分高于无关文本 |
| test_ingest_writes_chunks | 导入写入条数大于 0 |
| test_query_returns_most_relevant_first | 结果降序、分数落在 0 到 1、首条命中关键词 |
| test_filter_isolates_tenant | 按 user_id 过滤只取本租户数据，走 metadata 的 JSON 路径 |
| test_hit_fields_are_complete | 命中字段完整且类型正确 |
| test_reingest_is_idempotent | 同文档重复导入不产生重复记录 |

#### 集合清理

用例结束后自动删除本次使用的集合，但只在同时满足以下条件时执行：

1. 集合名以 rag_data_it 开头
2. 未设置 RAG_IT_KEEP=1

因此不会误删他人数据。若想跑完后保留集合以检查数据，设置 RAG_IT_KEEP=1。

详细设计见 plan/DEVELOPMENT_PLAN.md。
