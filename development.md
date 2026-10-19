# rag-data 开发指南

开发、构建与测试相关的说明；功能特性与 API 用法见 README.md。

## 环境与依赖管理

本项目使用 hatch 管理构建与运行环境，按其 features 声明自动装配依赖。

### 环境要求

- Python 3.8 及以上
- hatch（构建与环境管理）
- 核心依赖：pydantic、pydantic-settings

### 命令分类索引

下表按用途归类本文命令，便于按需查找；每个分类对应下方一个小节。

| 分类 | 用途 | 小节 |
| :--- | :--- | :--- |
| 工具安装 | 安装 hatch | 安装 hatch |
| 环境管理 | 创建、查看、清理 hatch 环境，进入交互式 shell | 环境管理（hatch） |
| 测试 | 全量、单文件、单用例与跳过清单 | 测试 |
| 检查与格式化 | 类型检查、静态检查、格式化 | 检查与格式化 |
| 构建与发布 | 构建发行包、发布到 PyPI | 构建与发布 |
| 集成测试 | 配置环境变量后运行需外部服务的用例 | 集成测试 |

### 安装 hatch

```bash
# hatch：构建与多环境管理
pip install hatch
```

### 环境管理（hatch）

hatch 环境直接读取 pyproject.toml 的依赖声明，已定义两套；创建环境时按 features 安装对应依赖，无需单独的依赖安装步骤：

| 环境 | 名称 | 包含依赖 |
| :--- | :--- | :--- |
| 默认环境 | default | parsers、embedding、epub、loguru、structlog、dev |
| 开发环境 | rag_data_dev | parsers、embedding、milvus、epub、pdf、word、openai、loguru、structlog、dev |

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
| 缺可选依赖 | 未装 yaml、loguru、structlog、pdfplumber、python-docx | 依赖名、模块名、安装命令 |
| 缺外部服务 | MySQL、PG、Milvus 驱动缺失或服务不可达 | 服务名、目标地址、异常类型与原文 |
| 缺本机示例 | 未在 conftest 的 samples 中填写路径 | 需填写的字段名 |
| 仅适用某后端 | 用例限定 PG 后端 | 当前实际后端名 |

与本机示例相关的配置：测试目录下 conftest.py 的 samples 分组（txt、md、pdf、docx、epub 五个字段）。

### 集成测试

集成用例需真实外部服务，默认由 addopts 的 -m not integration 跳过，人工显式指定标记执行：

| 文件 | 依赖 |
| :--- | :--- |
| tests/test_integration_qwen_epub_milvus.py | DashScope API Key、Milvus 服务 |
| tests/test_vector_store.py（integration 段） | 真实 Milvus 或 PostgreSQL 向量库 |

```bash
# 安装 dev 依赖后按文件或用例筛选
hatch run rag_data_dev:pytest -m integration --integration -v -s

# 只跑向量库后端或用例名匹配的部分
hatch run rag_data_dev:pytest -m integration --integration -v -s -k vector_store
hatch run rag_data_dev:pytest -m integration --integration -v -s -k chunk_returns_non_empty_blocks
```

连接信息、密钥、后端选择与样例路径统一写在测试目录下的 conftest.py：只改顶部的各分区 JSON 变量，无需再导出环境变量。

集成用例的配置项与安全约定见各测试文件头部注释，仓库与代码中不落任何密钥。

## 构建与发布

```bash
# 构建 sdist 与 wheel，产物在 dist/
hatch build

# 发布到 PyPI
hatch publish
```

说明：构建产物位于 dist 目录；发布前需配置 PyPI 凭据。

