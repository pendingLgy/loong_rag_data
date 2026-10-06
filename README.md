# rag-data

历史文本数据导入与向量化管道。范围聚焦离线批量导入链路：文档解析、分块、向量化；向量落库流程待后续重新设计，不含在线召回模块。

## 特性

- 按格式切块：切块规则由各解析器自持（递归切分 + 四分之一重叠前缀）；spaCy 句柄由门面加载并缓存，逐级注入切块
- 文档解析：TXT、Markdown、EPUB、PDF、Word；EPUB 以 ebooklib 与 BeautifulSoup 提取并排除目录页，PDF 以 pdfplumber 按物理坐标提取并过滤页眉页脚，Word 以 python-docx 提取段落
- pydantic v2 数据模型与 pydantic-settings 配置，边界即校验
- 日志适配层：标准库 logging、loguru、structlog 三后端可切换
- 向量化：Embedder 封装 provider（openai、qwen），字符串直接编码为稠密向量
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
uv pip install -e ".[epub,loguru]"

# 安装开发依赖
uv pip install -e ".[dev]"

# 或一次性安装全部可选与开发依赖
uv pip install -e ".[parsers,embedding,epub,pdf,word,loguru,structlog,dev]"
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
| 开发环境 | rag_data_dev | parsers、embedding、epub、pdf、word、loguru、structlog、dev |

```bash
# 创建环境
hatch env create
hatch env create rag_data_dev

# 查看已有环境
hatch env show

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

app = RagData.create()                 # 一次装配全部流程
text = app.parse("README.md")          # 流程：解析为纯文本
chunks = app.chunk("README.md", text)  # 流程：按格式切块
vectors = app.vectorize(["一段文本"])    # 流程：字符串向量化
```

## 公共 API

本包对外只暴露一个入口文件，即包根 __init__.py；外部调用无需关心内部子包路径：

```python
import rag_data
from rag_data import Settings, RagData
```

导出内容按用途分组：

| 分组 | 导出符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | Settings、ParsingSettings、EmbeddingSettings、LoggingSettings、load_env_overrides、ENV_PREFIX、ENV_NESTED_DELIMITER |
| 解析 | parse_document |
| 向量化 | Embedder、BaseEmbeddingProvider、OpenAIEmbeddingProvider、QwenEmbeddingProvider、register_embedding、register_embedding_provider、available_embedding_providers、is_embedding_registered、resolve_embedding_provider、create_embedding_provider |
| 流程门面 | RagData、vectorize、build_settings、build_logger、build_nlp、build_embedder |
| 日志适配 | LoggerAdapter、configure_logging、get_logger |
| 异常 | RagDataError、ConfigError、DataError、OptionalDependencyError、ParserDependencyError、ParseError、EmbeddingError |

解析产出纯文本，切块产出分块列表，向量化直接返回稠密向量；落库流程待存储层重新设计后补全。


## 流程门面

facade.py 把各流程的调用方法集中封装，调用方不必再逐个导入子模块：

```python
from rag_data import RagData

app = RagData.create()                # 装配配置、日志、句柄与向量化
text = app.parse("a.md")               # 流程：解析文件为文本
chunks = app.chunk("a.md", text)       # 流程：按格式切块
app.vectorize(["一段文本"])             # 流程：向量化字符串
```

也可以按需只装配单个流程：

| 方法 | 流程 |
| :--- | :--- |
| build_settings(source, **overrides) | 加载配置（分区字典硬编码）|
| build_logger(settings) | 装配日志适配层 |
| build_nlp(settings, logger) | 加载 spaCy 句柄，不可用则降级为 None |
| build_embedder(settings, logger, model) | 装配向量化组件 |
| vectorize(texts, source) | 一键向量化字符串 |

```python
from rag_data import build_settings, build_logger, build_embedder

settings = build_settings(embedding={"provider": "qwen"})
logger = build_logger(settings)
embedder = build_embedder(settings, logger)
```

RagData 一次装配后可反复调用，配置、日志、句柄与向量化组件都挂在实例上：

```python
from rag_data import RagData

app = RagData.create()
text = app.parse("doc.md")           # 文件 → 纯文本
vectors = app.vectorize(["一段文本"])   # 字符串 → 稠密向量
```

字符串向量化返回与输入同序的向量列表；
app.vectorize_text(text) 返回单个向量。两者都经过 Embedder，维度按 embedding.dim 校验。


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
    "embedding": { "provider": "qwen" },
})
```

环境变量亦可：

```bash
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__MODEL=qwen3.7-text-embedding
DASHSCOPE_API_KEY=sk-xxx
```

### embedding 配置项

| 字段 | 默认值 | 说明 |
| :--- | :--- | :--- |
| provider | openai | 注册表中的 provider 名 |
| model | 空 | 模型名，留空用 provider 默认模型 |
| dim | 1024 | 期望输出维度，向量化结果按此校验 |
| batch_size | 128 | 每批文本条数，与 provider 上限取较小者 |
| api_key | 空 | 留空时读取 provider 约定的环境变量 |
| base_url | 空 | 留空时用 provider 默认地址，可指向自建网关 |
| timeout | 60.0 | 单次请求超时秒数 |

输出维度需与 embedding.dim 一致，不一致会抛 EmbeddingError。

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


## 配置

配置按模块分区，默认读取环境变量，也可在代码中硬编码，模型定义在 config.py。

优先级由高到低：

| 优先级 | 来源 | 示例 |
| :--- | :--- | :--- |
| 1 | 代码硬编码 | Settings.load(embedding={"provider": "qwen"}) |
| 2 | 环境变量与 .env | RAG_EMBEDDING__PROVIDER=qwen |
| 3 | 字段默认值 | 各分区模型内声明 |

两者同时存在时以代码硬编码为准，未硬编码的字段由环境变量填充。

分区与主要字段：

| 分区 | 主要字段 |
| :--- | :--- |
| parsing | spacy_model、max_chars |
| embedding | provider、model、dim、batch_size、api_key、base_url、timeout |
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
    embedding={"provider": "qwen", "dim": 1024},
    chunking={"max_chars": 180},
)
settings = Settings.load({"embedding": {"provider": "qwen"}})   # 也可直接传字典
```

门面与一键入口同样支持：

```python
from rag_data import RagData, build_settings

settings = build_settings(embedding={"provider": "qwen"})
app = RagData.create(overrides={"embedding": {"provider": "qwen"}})
```

### 环境变量

分区与字段之间用双下划线分隔，名称大写：

```bash
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__DIM=512
RAG_LOGGING__LEVEL=DEBUG
RAG_LOGGING__JSON=true
```

取值按字段类型自动转换，对象与数组按 JSON 书写。
分区名或字段名写错会抛 ConfigError，并列出可用值。

## 目录结构

```text
src/rag_data
- __init__.py        公共入口，统一导出全部 API
- config.py          pydantic-settings 配置
- registry.py        通用注册表基类，provider 注册表复用
- exceptions.py      领域异常
- facade.py          流程门面，一站式封装各流程调用
- parsers            文档解析器：DocumentParser 基类 + 各格式子类（markdown、txt、epub、pdf、word）
- embedding          向量化接口、provider 注册表与 openai、qwen 实现
- logging            日志适配层
```

## 待实现（TODO）

以下复杂逻辑以 TODO 标注，待人工补全：

- 解析细化：PDF 双栏、Word 表格等版面级处理
- 向量化：兼容 embedding 接口的本地模型（如 sentence-transformers）provider
- 数据保存：向量落库流程待存储层重新设计后补全

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

存储层移除后，原 Qwen + Milvus 集成测试已一并删除；落库流程重新设计后按需重建。

详细设计见 plan/DEVELOPMENT_PLAN.md。
