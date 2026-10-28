# rag-data 开发指南

本文覆盖开发、构建与测试；功能特性与 API 用法见 [README.md](README.md)，模块职责见 [.vcl/struct.md](.vcl/struct.md)。

## 环境与依赖管理

项目使用 hatch 管理构建与运行环境，按 features 声明自动装配依赖。

### 环境要求

- Python 3.8 及以上
- hatch（构建与环境管理）
- 核心依赖：pydantic、pydantic-settings、PyYAML

### hatch 环境

两套环境在 pyproject.toml 中定义，创建时按 features 安装对应依赖，无需单独安装步骤：

| 环境 | features | 适用场景 |
| :--- | :--- | :--- |
| default | parsers、openai、dev | 日常开发与测试 |
| rag_data_dev | parsers、milvus、dev、word、pdf、epub、ragas-eval | 需要 PDF、Word、EPUB、Milvus 与评测依赖时 |

注意：rag_data_dev 未显式声明 openai extra，但 ragas-eval 依赖 langchain-openai，会间接带入 openai SDK。

```bash
# 安装 hatch
pip install hatch

# 创建环境
hatch env create
hatch env create rag_data_dev

# 查看与清理
hatch env show
hatch env prune                 # 清理全部环境
hatch env remove rag_data_dev   # 清理指定环境

# 在环境内增删依赖
hatch run rag_data_dev:pip install pandas python-dotenv datasets
hatch run rag_data_dev:pip uninstall <package_name>
hatch run rag_data_dev:pip list

# 依赖树与包信息
hatch run rag_data_dev:pip install pipdeptree
hatch run rag_data_dev:pipdeptree
hatch run rag_data_dev:pip show ragas
hatch run rag_data_dev:pipdeptree -p ragas

# 交互式 shell
hatch shell
hatch shell rag_data_dev
```

### 常用组合

| 目标 | 命令 |
| :--- | :--- |
| 克隆后快速开始 | hatch env create rag_data_dev |
| 运行测试 | hatch run test |
| 类型检查 | hatch run types |
| 代码风格 | hatch run check 或 hatch run format |
| 构建发行包 | hatch build |

脚本定义见 pyproject.toml 的 [tool.hatch.envs.*.scripts]。

---

## 测试

```bash
# 全量
hatch run test
hatch run rag_data_dev:test

# 单文件与单用例
hatch run rag_data_dev:pytest tests/test_llm.py -rs
hatch run rag_data_dev:pytest tests/test_llm.py::test_builtin_providers_declared -rs

# 关键字筛选
hatch run rag_data_dev:pytest -k parser -rs

# 跳过清单与覆盖率
hatch run rag_data_dev:pytest -rs
hatch run rag_data_dev:pytest -rs --cov=rag_data --cov-report=term-missing
```

### 标记与门控

| 项 | 现状 |
| :--- | :--- |
| addopts | -s -q -rs -m 'not integration' |
| markers | integration 与 eval 已在 pyproject 声明 |
| 用例使用情况 | 当前没有用例带 integration 或 eval 标记 |
| --integration 选项 | conftest 已注册，暂无消费方 |

如需恢复集成测试，按 pyproject.toml 中 addopts 的注释行切换为 -m integration，并在命令行显式传 --integration。

### 跳过信息

用例在缺可选依赖（spaCy 及中文模型、pdfplumber、python-docx、EbookLib）或缺本机样例文件时跳过；跳过提示由各测试模块内的局部助手函数打印，配合 -rs 可查看清单。

### conftest 说明

tests/conftest.py 负责测试环境的公共准备：

| 内容 | 说明 |
| :--- | :--- |
| 路径注入 | 把 src 与 tests 加入 sys.path，确认 src 布局可直接导入 |
| 环境变量 | 加载仓库根的 .env_ai_key |
| 分区 JSON | MILVUS、PG、MYSQL、SQLITE、VECTOR、EMBEDDING、LLM、PARSING、SAMPLES、CASES |
| 配置对象 | PARSING_SETTINGS、EMBEDDING_SETTINGS、LLM_SETTINGS 由对应 JSON 实例化 |
| embedding_api_key | 从 .env_ai_key 读取后注入 EMBEDDING_SETTINGS.api_key |
| settings 夹具 | 返回 RagSetting()，即按 RAG_ 前缀读取环境 |
| 命令行选项 | --integration |

SAMPLES_JSON 里的 epub、txt、pdf、docx 指向本机桌面文件，换机器后需按实际路径修改；MILVUS、PG、MYSQL、VECTOR、CASES 分区在集成用例移除后暂无消费方，保留作为后续集成测试的落点。密钥只放 .env_ai_key，不提交仓库。

## 检查与格式化

```bash
hatch run types     # mypy src tests
hatch run check     # ruff check src tests
hatch run format    # ruff format src tests

hatch run rag_data_dev:types
hatch run rag_data_dev:check
hatch run rag_data_dev:format
```

mypy 配置：python_version 为 3.8，启用 pydantic 插件，warn_unused_ignores 为 true，ignore_missing_imports 为 true。

---

## 评测脚本运行

四个脚本位于 src/rag_data/eval/rag，属隐式命名空间包，按文件路径执行：

```bash
# 1. 解析、切块、向量化并写入 Milvus
python src/rag_data/eval/rag/build_rag_data.py

# 2. 检索、邻居窗口扩展并调用大模型回答
python src/rag_data/eval/rag/rag_data_search.py

# 3. 合成 Ragas 测试集，输出 testset.csv
python src/rag_data/eval/rag/ragas_eval_dataset.py

# 4. 指标打分，输出 rag_evaluation_report.csv
python src/rag_data/eval/rag/evaluate_rag.py
```

| 前置条件 | 说明 |
| :--- | :--- |
| .env_rag_eval | 提供 milvus_url、db_name、collection_name、dataset_path、rag_eval_output_path 与两组模型凭据 |
| Milvus | 可连接的 Milvus 服务，建议 2.5 及以上（需要 BM25 函数与稀疏向量） |
| dataset/ 与 eval_outpu/ | 测试集与报告的输出目录，仓库内均为空目录 |
| 额外依赖 | pandas、python-dotenv、datasets 未列入 extras，需在环境中手动安装 |

脚本现状：build_rag_data.py 与 rag_data_search.py 各存一份建集合与连接逻辑，集合名固定为 eval_rag_data，部分样例路径硬编码在 main 中；ragas_eval_dataset.py 的 BASE_DIR 层级与另三个脚本不同，若读不到 .env_rag_eval 请先核对路径。

---

## 目录结构与测试清单

```text
src/rag_data/
- __about__.py      版本号（hatch 版本来源）
- __init__.py       包根，以 from-import 暴露 core 的 19 个名称
- rag_engine.py     RagEngine 与 _build_nlp、_build_embedder、_build_llm
- core/            config.py（分区配置与三类来源）、exceptions.py、registry.py
- nlp/             spacy.py（句柄双缓存）、lemmatization.py（BM25 词形）
- parsers/         base、registry_parser、parser_handler、markdown、txt、epub、pdf、word
- embedding/       base、registry_embedder、openai_embedder、qwen_embedder、embedder_handler
- llm/             base、registry_llm、openai_llm、qwen_llm、deepseek_llm、llm_handler
- eval/rag/        build_rag_data、rag_data_search、ragas_eval_dataset、evaluate_rag
- util/            snowflake.py（雪花 ID）
```

测试文件与用例规模（def test_ 静态计数，共 177 项，未展开参数化）：

| 文件 | 用例数 | 覆盖范围 |
| :--- | :--- | :--- |
| conftest.py | 无 | 路径注入、.env_ai_key、分区 JSON 到 Settings、settings 夹具、--integration 选项 |
| test_config.py | 18 | RagSetting 三类来源、分区与字段校验、环境隔离 |
| test_parsers_base.py | 16 | 分句、重叠前缀与递归切块 |
| test_parsers_dispatch.py | 19 | 派发、注册表与预实例化复用 |
| test_parsers_txt.py | 11 | TxtParser 契约、解析与切块 |
| test_parsers_markdown.py | 11 | MarkdownParser |
| test_parsers_epub.py | 10 | EpubParser |
| test_parsers_pdf.py | 10 | PdfParser |
| test_parsers_word.py | 12 | WordParser |
| test_embedding.py | 21 | 注册表、内置 embedder 与 EmbedderHandler |
| test_llm.py | 36 | 注册表、内置 provider 与 LLMHandler |
| test_spacy.py | 13 | spaCy 分词与中文 BM25 词形 |

## 待实现（TODO）

- 包导出：包根 __all__ 只声明 __version__，与 19 个 from-import 名称不一致，需补齐或改为显式导出
- eval：脚本无 __init__.py，依赖 pandas、python-dotenv、datasets 未声明，建集合与连接逻辑重复，main 中的路径与集合名硬编码，ragas_eval_dataset 的 BASE_DIR 层级待统一
- parsers：_process_text_recursive 的 nlp 默认值应为 None；PDF 双栏与 Word 表格等版面级解析待细化
- nlp：clear_cache 未清 _PIPELINES_LEMMA；lemmatize_for_bm25 目前只有测试在用
- 评测：报告与阈值未纳入 CI，基线需人工比对
- openai extra 目前为 openai>=3.26.0，升级前建议核对可用版本

## 构建与发布

```bash
hatch build      # 构建 sdist 与 wheel，产物在 dist/
hatch publish    # 发布到 PyPI
```

构建产物位于 dist 目录；发布前需配置 PyPI 凭据。仓库地址见 pyproject.toml 的 [project.urls]。
