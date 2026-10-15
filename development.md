# rag-data 开发指南

开发、构建与测试相关的说明；功能特性与 API 用法见 README.md。

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
uv pip install -e ".[epub,loguru]"

# 安装开发依赖
uv pip install -e ".[dev]"

# 或一次性安装全部可选与开发依赖
uv pip install -e ".[parsers,embedding,epub,pdf,word,openai,milvus,loguru,structlog,dev]"
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
| 默认环境 | default | parsers、embedding、epub、loguru、structlog、dev |
| 开发环境 | rag_data_dev | parsers、embedding、milvus、epub、pdf、word、openai、loguru、structlog、dev |

```bash
# 创建环境
hatch env create
hatch env create rag_data_dev

# 查看已有环境
hatch env show

# 清理环境
hatch env prune #清理所有环境
hatch env prune -e parsers # 使用 -e 或 --env 参数指定某个具体的环境名称（如 dev、parsers、test）：****

# 查看已安装依赖
hatch run rag_data_dev:pip list

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
hatch run rag_data_dev:pytest -rs #查看具体信息
hatch run rag_data_dev:pytest -- tests/test_integration_qwen_epub_milvus.py -rs
hatch run rag_data_dev:pytest tests/test_integration_qwen_epub_milvus.py::test_chunk_returns_non_empty_blocks -rs

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

hatch run rag_data_dev:pytest -- tests/test_integration_postgres.py -rs
```

若使用 uv 直接管理依赖：

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

### 集成测试

tests/test_integration_qwen_epub_milvus.py 是手工集成测试：真实解析 EPUB、切块、
调用 Qwen 嵌入、写入 Milvus 并做相似度检索。默认由 addopts 的 -m 'not integration' 跳过，
需备好凭据与 Milvus 服务后显式指定标记执行：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxx"
$env:RAG_IT_MILVUS_URI = "http://127.0.0.1:19530"
hatch run rag_data_dev:pytest -m integration -v -s
```

环境变量、默认集合名与安全约定见该文件头部注释。

详细设计见 plan/DEVELOPMENT_PLAN.md。
