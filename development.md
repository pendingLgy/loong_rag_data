# rag-data 开发指南

开发、构建与测试相关的说明；功能特性与 API 用法见 README.md。

## 环境与依赖管理

本项目使用 hatch 管理构建与运行环境，按其 features 声明自动装配依赖。

### 环境要求

- Python 3.8 及以上
- hatch（构建与环境管理）
- 核心依赖：pydantic、pydantic-settings、PyYAML（load_yaml 使用）

### 命令分类索引

下表按用途归类本文命令，便于按需查找；每个分类对应下方一个小节。

| 分类 | 用途 | 小节 |
| :--- | :--- | :--- |
| 工具安装 | 安装 hatch | 安装 hatch |
| 环境管理 | 创建、查看、清理 hatch 环境，进入交互式 shell | 环境管理（hatch） |
| 测试 | 全量、单文件、单用例与跳过清单 | 测试 |
| 检查与格式化 | 类型检查、静态检查、格式化 | 检查与格式化 |
| 构建与发布 | 构建发行包、发布到 PyPI | 构建与发布 |
| 集成测试 | 修改 conftest 的分区 JSON 后运行需外部服务的用例 | 集成测试 |
| 目录与用例清单 | 源码目录职责、测试文件与用例规模 | 目录结构与测试清单 |
| 待实现 | 已知未完成项与依赖遗留 | 待实现（TODO） |

### 安装 hatch

```bash
# hatch：构建与多环境管理
pip install hatch
```

### 环境管理（hatch）

hatch 环境直接读取 pyproject.toml 的依赖声明，已定义两套；创建环境时按 features 安装对应依赖，无需单独的依赖安装步骤：

| 环境 | 名称 | 包含依赖 |
| :--- | :--- | :--- |
| 默认环境 | default | parsers、embedding、epub、dev |
| 开发环境 | rag_data_dev | parsers、embedding、sqlalchemy、pg、mysql、milvus、epub、pdf、word、openai、dev |

```bash
# 创建环境：默认环境与开发环境
hatch env create
hatch env create rag_data_dev

# 查看已有环境
hatch env show

# 清理环境：不带参数清理全部，-e 或 --env 指定环境名（如 dev、parsers）
hatch env prune
hatch env prune -e parsers

# 查看已安装依赖
hatch run rag_data_dev:pip list

# 进入交互式 shell
hatch shell
hatch shell rag_data_dev
```

### 常用组合

| 目标 | 命令 |
| :--- | :--- |
| 克隆后快速开始 | hatch env create rag_data_dev |
| 运行测试 | hatch run pytest |
| 类型检查 | hatch run types 或 hatch run rag_data_dev:types |
| 代码风格 | hatch run check 或 hatch run rag_data_dev:check |
| 构建发行包 | hatch build |

## 开发

推荐使用 hatch 环境，命令会按环境 features 自动装配依赖，无需手动安装或激活：

脚本定义见 pyproject.toml 的 [tool.hatch.envs.*.scripts]；下面按分类列出常用命令。

### 测试

```bash
# 默认环境（含 dev 工具）
hatch run test      # pytest
hatch run pytest

# 开发环境（叠加全部可选依赖）
hatch run rag_data_dev:test

# 单文件与单用例
hatch run rag_data_dev:pytest -- tests/test_db.py -rs
hatch run rag_data_dev:pytest tests/test_db.py::test_builtin_dbs_declared -rs

# 跳过清单与覆盖率
hatch run rag_data_dev:pytest -rs
hatch run rag_data_dev:pytest -rs --cov=rag_data --cov-report=term-missing
```

### 检查与格式化

```bash
# 默认环境
hatch run types     # mypy src tests
hatch run check     # ruff check src tests
hatch run format    # ruff format src tests

# 开发环境
hatch run rag_data_dev:types
hatch run rag_data_dev:check
```

### 跳过与覆盖率可见性

用例在缺依赖、缺外部服务或缺本机示例文件时跳过；跳过信息由各测试模块内的局部辅助函数输出，包含依赖名、目标地址与原始异常。addopts 已启用 -rs，每次运行都会打印跳过清单：

```bash
hatch run test                    # 末尾附 SKIPPED 摘要（含原因与位置）
pytest -rs tests                  # 与 --cov 叠加即可定位覆盖盲区
pytest -rs -m integration tests   # 只看集成用例的跳过原因
```

| 跳过场景 | 触发条件 | 信息内容 |
| :--- | :--- | :--- |
| 缺可选依赖 | 未装 spaCy 模型、pdfplumber、python-docx、EbookLib、pymilvus、psycopg | 依赖名、模块名、安装命令 |
| 缺 BM25 能力 | pymilvus 低于 2.5（无 Function 与稀疏向量） | 所需的最低版本 |
| 缺外部服务 | MySQL、PG、Milvus 驱动缺失或服务不可达 | 服务名、目标地址、异常类型与原文 |
| 缺本机示例 | 未在 conftest 的 samples 中填写路径 | 需填写的字段名 |
| 仅适用某后端 | 用例限定 PG 后端 | 当前实际后端名 |

与本机示例相关的配置：测试目录下 conftest.py 的 samples 分组（txt、md、pdf、docx、epub 五个字段）。

### 集成测试

集成用例需真实外部服务，默认由 addopts 的 -m not integration 反选，且各文件在模块级检查 --integration（或 --eval）参数，未提供时整体跳过，人工显式指定后执行：

| 文件 | 依赖 |
| :--- | :--- |
| tests/test_integration_qwen_epub_milvus.py | DashScope API Key、Milvus 服务 |
| tests/test_vector_store.py（integration 段） | 真实 Milvus 或 PostgreSQL 向量库 |
| tests/test_db.py（integration 段） | 真实 MySQL 或 PostgreSQL |
| tests/test_integration_facade_ragdata.py、tests/test_integration_facade_memoryserver.py | 真实解析依赖与存储服务 |
| tests/test_milvus_spacy_nlp.py | Milvus 服务、spaCy 中文模型 |

```bash
# 安装 dev 依赖后按文件或用例筛选
hatch run rag_data_dev:pytest -m integration --integration -v -s

# 只跑向量库后端或用例名匹配的部分
hatch run rag_data_dev:pytest -m integration --integration -v -s -k vector_store
hatch run rag_data_dev:pytest -m integration --integration -v -s -k chunk_returns_non_empty_blocks

# 评分用例（eval 标记，另需 --eval 参数）
hatch run rag_data_dev:pytest tests/test_rag_eval.py --eval -v -s
hatch run rag_data_dev:pytest tests/test_epub_to_milvus_and_pg.py --eval -v -s
```

连接信息、后端选择与样例路径统一写在测试目录下的 conftest.py：只改顶部的各分区 JSON 变量（MILVUS、PG、MYSQL、SQLITE、VECTOR、EMBEDDING、LLM、PARSING、SAMPLES、CASES），无需再导出环境变量。唯一的例外是嵌入密钥：conftest 会加载项目根目录的 .env_ai_key，并读取其中的 embedding_api_key 作为 embedding.api_key，密钥不写进仓库。

安全约定：

- 数据库名、集合名与表名统一以 autotest_ 开头；删表与删集合前都要过 storage 的 guard 模块，非 autotest_ 前缀直接抛 StorageError
- vector.prefix 决定用例允许删除的前缀，vector.keep 为 true 时保留集合便于排查
- 单条链路用例 test_epub_to_milvus_and_pg.py 自带独立 JSON，不读 conftest；其余用例只读 conftest 的 *_CONFIG 字典与各 Settings 对象

集成用例的配置项与安全约定见各测试文件头部注释，仓库与代码中不落任何密钥。

## 目录结构与测试清单

```text
src/rag_data
- __init__.py        包根，仅导出当前目录直接模块的公共符号（23 个）
- __about__.py       版本号（hatch 版本来源）
- config.py          分区配置模型与三类来源加载（load_mapping、load_env、load_yaml）
- exceptions.py      领域异常（含 StorageError、LLMError）
- registry.py        通用注册表（Registry、Ref）
- facade.py          流程门面 RagData、MemoryServer，私有装配方法 _build_*
- nlp                spaCy 句柄（spacy.py）与 BM25 词形处理（lemmatization.py）
- parsers            文档解析：base、各格式子类、parser_handler、registry_parser
- embedding          向量化：base、registry_embedder、openai 与 qwen embedder、embedder_handler
- llm                大模型：base、registry_llm、openai、qwen、deepseek、llm_handler
- memory             记忆层：base、handler、stores、scope、prompts、models
- eval               评测：rag 下的 dataset、metrics、evaluator
- storage            存储：guard.py 提供前缀守卫，db 与 vector_store 两个子包
```

```text
src/rag_data/storage/db
- base.py            BaseDB 抽象接口（含 drop_tables）
- registry_db.py     纯注册表（BUILTIN_DBS）
- db_engine.py       SQLDatabase 与 build_database
- schema.py          表结构、建表 SQL、HistoryRecord 与 Message
- snowflake.py       雪花 ID 生成器
- sqlite_db.py       SQLiteDB
- pg_db.py           PGDB
- mysql_db.py        MySQLDB

src/rag_data/storage/vector_store
- base.py            VectorStoreBase、VectorStoreResult、MetricType、drop_table 与 _drop_col
- registry_vector.py 纯注册表（BUILTIN_STORES）
- vector_engine.py   build_vector_store、VectorEngine、build_vector_engine
- milvus_store.py    MilvusVectorStore（单一 BM25 schema，text 配 chinese 分析器）
- pg_store.py        PGVectorStore（建表与索引）
```

测试文件与用例规模（def test_ 静态计数，共 447 项）：

| 文件 | 用例数 | 覆盖范围 |
| :--- | :--- | :--- |
| conftest.py | 无 | 路径注入、分区 JSON 与 Settings 对象、--integration 与 --eval 选项 |
| test_config.py | 56 | 两个设置类的分区、三类来源与 enable |
| test_facade.py | 16 | 模块级 _build_* 门控与参数透传 |
| test_facade_ragdata.py | 14 | RagData 构造、解析与切块 |
| test_facade_memoryserver.py | 5 | MemoryServer 装配、流程代理与门控 |
| test_parsers_base.py | 16 | 分句、重叠前缀与递归切块 |
| test_parsers_dispatch.py | 19 | 派发、注册表与预实例化复用 |
| test_parsers_txt.py | 11 | TxtParser 契约、解析与切块 |
| test_parsers_markdown.py | 11 | MarkdownParser |
| test_parsers_epub.py | 10 | EpubParser |
| test_parsers_pdf.py | 10 | PdfParser |
| test_parsers_word.py | 12 | WordParser |
| test_embedding.py | 21 | 注册表、内置 embedder 与 EmbedderHandler |
| test_llm.py | 36 | 注册表、内置 provider 与 LLMHandler |
| test_db.py | 63 | 注册表、表结构、SQLite 端到端、MySQL 与 PG 集成 |
| test_vector_store.py | 73 | 注册表、DDL、引擎与真实服务集成 |
| test_rag_eval.py | 27 | 评测指标与执行器单测，以及真实检索与生成评分 |
| test_epub_to_milvus_and_pg.py | 2 | EPUB 解析向量化后同时写入 Milvus 与 PG |
| test_integration_qwen_epub_milvus.py | 6 | 真实 EPUB 到 Milvus 端到端 |
| test_integration_facade_ragdata.py | 12 | RagData 集成：真实解析与切块 |
| test_integration_facade_memoryserver.py | 11 | MemoryServer 集成：四条流程 |
| test_spacy.py | 13 | spaCy 分词与词形探索性用例 |
| test_milvus_spacy_nlp.py | 3 | Milvus BM25 与 NLP 分析器匹配度 |

集成与评分用例在模块级声明 pytestmark，并检查对应的命令行选项，未提供时整体跳过并打印 [SKIP INFO]。

## 待实现（TODO）

- 解析细化：PDF 双栏、Word 表格等版面级处理
- 向量化：兼容本地模型（如 sentence-transformers）的 embedder（已有 embedding extra，尚未实现后端）
- 记忆层：异步版本、delete_all 与 reset；实体图谱、时序与衰减策略明确排除
- 记忆层：update 路径的词形文本仍用 keyword_text，与写入侧的 lemmatize_zh_for_bm25 不一致，待统一
- 数据保存：解析、向量化与落库的批量导入编排仍由调用方组合
- 包导入：import rag_data 经 facade 间接依赖 memory 与 storage 的顶层导入，缺少 SQLAlchemy 时会抛 ModuleNotFoundError，与可选依赖的惰性原则不符
- openai extra 当前锁定为 openai==3.26.0，建议核对版本并按需放宽

## 构建与发布

```bash
# 构建 sdist 与 wheel，产物在 dist/
hatch build

# 发布到 PyPI
hatch publish
```

说明：构建产物位于 dist 目录；发布前需配置 PyPI 凭据。

