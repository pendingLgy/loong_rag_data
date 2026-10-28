# rag-data

历史文本导入与向量化管道：文档解析、按格式切块、稠密向量化，并附一套离线评测脚本（Milvus + Ragas）。

![python](https://img.shields.io/badge/python-3.8+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

开发、构建与测试说明见 [development.md](development.md)；模块职责、分层与扩展点见 [.vcl/struct.md](.vcl/struct.md)。

---

## 特性

- **五类文档解析**：TXT、Markdown、EPUB、PDF、Word。继承 DocumentParser 并声明 backend 与 SUFFIXES 即自动注册，无需改动装配代码。
- **按格式切块**：切块规则由各解析器自持（递归切分 + 四分之一重叠前缀）；spaCy 句柄按模型名缓存并逐级注入。
- **分区配置**：pydantic v2 与 pydantic-settings 双层模型，支持 Mapping、环境变量、YAML 三种来源，实例化即校验；每个分区带 enable 开关。
- **向量化**：EmbedderHandler 按配置装配 openai 或 qwen embedder，支持注入自定义模型绕过配置，并按 dim 逐条校验维度。
- **大模型**：LLMHandler 按配置装配 openai、qwen、deepseek，提供 chat 与 complete 两个入口。
- **离线评测**：eval/rag 下的四个脚本，覆盖 EPUB 入库、检索问答、Ragas 测试集生成与指标打分。
- **纯注册表**：解析器、嵌入模型、大模型三张注册表只登记实现类，实例化在各自 handler 中进行。
- **惰性可选依赖**：spacy、pymilvus、ragas 等均在使用处导入，未安装时库仍可正常 import。

## 环境要求

- Python 3.8 及以上
- 核心依赖：pydantic>=2.5、pydantic-settings>=2.1、PyYAML>=6.0.3

## 安装

```bash
# 仅核心依赖：配置、异常、注册表
pip install -e .

# 解析与切块（spaCy 与中文模型，切块必需）
pip install -e '.[parsers]'

# 按需叠加格式支持
pip install -e '.[parsers,epub,pdf,word]'

# 评测脚本所需
pip install -e '.[parsers,openai,milvus,ragas-eval]'
```

| extra | 内容 |
| :--- | :--- |
| parsers | spacy 与 zh_core_web_sm、zh_core_web_md（切块必需） |
| epub | EbookLib、beautifulsoup4 |
| pdf | pdfplumber |
| word | python-docx |
| milvus | pymilvus |
| openai | openai SDK（openai 与 qwen provider、embedder 共用） |
| ragas-eval | ragas、langchain-core、langchain-openai、langchain-community 等 |
| dev | pytest、pytest-asyncio、pytest-cov、mypy、ruff |

---

## 快速开始

```python
from rag_data import RagSetting
from rag_data.rag_engine import RagEngine

settings = RagSetting.load_mapping(
    parsing={'enable': True, 'spacy_model': 'zh_core_web_sm', 'max_chars': 500},
    embedding={'enable': True, 'provider': 'qwen', 'model': 'qwen3.7-text-embedding', 'dim': 1024},
)

app = RagEngine(settings)
text = app.parse_document('README.md')            # 文件 -> 纯文本
chunks = app.chunk_document('README.md', text)    # 纯文本 -> 切块
vectors = app.embedder.encode(chunks)             # 切块 -> 稠密向量
```

要点：

- 各分区默认关闭。parsing.enable 为 False 时 app.nlp 为 None，chunk_document 会因缺句柄抛 ParserDependencyError；embedding.enable 为 False 时 app.embedder 为 None。
- api_key 留空时按 provider 约定读取环境变量：OPENAI_API_KEY 或 DASHSCOPE_API_KEY。
- 库内不负责落库：向量由调用方写入自己的向量库（评测脚本直接调用 pymilvus）。

## 配置

### 三类来源

| 方法 | 来源 | 是否读环境变量 |
| :--- | :--- | :--- |
| RagSetting.load_mapping(source=None, **overrides) | JSON 对象（分区字典） | 否 |
| RagSetting.load_env(environ=None, **overrides) | 环境变量 | 是（可显式传 environ） |
| RagSetting.load_yaml(path, **overrides) | YAML 文件 | 否（惰性导入 PyYAML） |

优先级：overrides 高于来源数据，来源数据高于字段默认值，均以字段为单位合并。

```python
settings = RagSetting.load_mapping(embedding={'provider': 'qwen'})
settings = RagSetting.load_mapping({'embedding': {'dim': 1024}}, parsing={'max_chars': 180})
settings = RagSetting.load_yaml('config.yml')
settings = RagSetting.load_env()   # 显式按环境变量加载，不读 .env
```

### 分区与字段（括号内为默认值）

| 分区 | 字段 |
| :--- | :--- |
| parsing | enable(false)、spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | enable(false)、provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |

RagSetting 只有 parsing 与 embedding 两个分区：分区名写错抛 ConfigError，字段名写错抛 ValidationError。

### 大模型配置

LLMSettings 是独立模型，不挂在 RagSetting 下，需要时自行构造后交给 _build_llm 或 LLMHandler：

| 字段 | 默认值 | 说明 |
| :--- | :--- | :--- |
| enable | false | 分区开关 |
| provider | openai | openai、qwen、deepseek |
| model | 空 | 留空回落到 provider 的 default_model |
| temperature | 0.7 | 采样温度，范围 0 到 2 |
| max_tokens | 0 | 0 表示不显式传入，由服务端决定 |
| api_key、base_url | 空 | 分别回落到环境变量与默认地址 |
| timeout | 60.0 | 请求超时秒数 |

### 环境变量

分区名与字段名大写，双下划线分隔，取值按字段类型自动转换：

```bash
RAG_PARSING__ENABLE=true
RAG_PARSING__SPACY_MODEL=zh_core_web_sm
RAG_PARSING__MAX_CHARS=1000
RAG_EMBEDDING__ENABLE=true
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__DIM=1024
RAG_EMBEDDING__BATCH_SIZE=10
```

RagSetting 默认还会读取项目根的 .env（env_file 配置项）；根层 extra 为 ignore，RAG_ 命名空间下的无关变量会被忽略。

### 维度一致性

embedding.dim 声明期望的输出维度（默认 1024），EmbedderHandler 据此校验每条向量，不一致抛 EmbeddingError。使用 qwen 时建议与默认维度 1024 保持一致。

---

## 文档解析与切块

```text
文件 → parse_document → 纯文本 → chunk_document → 切块
```

| 格式 | 后缀 | 解析器 | 切块规则 |
| :--- | :--- | :--- | :--- |
| TXT | .txt | TxtParser | 整篇递归切分 |
| Markdown | .md、.markdown | MarkdownParser | 按行首井号分章，逐章递归切分 |
| EPUB | .epub | EpubParser | 按 spine 提取章节（含标题），再按井号分章 |
| PDF | .pdf | PdfParser | pdfplumber 按物理坐标重组段落，按比例过滤页眉页脚 |
| Word | .docx | WordParser | python-docx 提取段落，同样按井号分章 |

ParserHandler：

| 成员 | 说明 |
| :--- | :--- |
| resolve_parser(path, max_chars=None, safe_max_chars=None) | 取该格式实例；长度与缓存一致时复用，不一致则临时新建 |
| parse_document(path) | 解析单个文档并返回纯文本 |
| chunk_document(path, text, nlp=None, max_chars=None, safe_max_chars=None) | 按扩展名派发切块 |
| get_instance(path) | 取缓存实例，不新建 |
| parser_classes、instances（属性） | 解析器类集合与已预实例化的实例副本 |

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

    def parse(self, path):
        ...

    def chunk(self, text, nlp=None):
        ...
```

切块参数与规则：

- 单块上限 max_chars 与安全上限 safe_max_chars 由 parsing 分区注入（默认 500 与 2000）；解析器自身默认是 1000 与 2000。
- 块间携带上一块尾部约四分之一的文本作为重叠前缀，保证跨块语义连续。
- 切块必须有 spaCy 句柄，缺失时抛 ParserDependencyError，并提示 pip install -e '.[parsers]'。
- 暂不支持的扩展名抛 ParseError；文件不存在同样抛 ParseError。

## 向量化

实现由 embedding.provider 选择，内置 openai 与 qwen：

```text
EmbedderHandler → resolve_embedder → Registry → OpenAIEmbedder
                                            └ QwenEmbedder
```

| embedder | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | qwen3.7-text-embedding | 1024 | DASHSCOPE_API_KEY | 10 |

EmbedderHandler(settings.embedding, model=None)：

| 成员 | 说明 |
| :--- | :--- |
| provider（属性） | 注入的 model 优先，否则按 provider 装配并缓存 |
| encode(texts) | 分批编码为稠密向量，空输入返回空列表，结果按 dim 校验 |

批次大小取 embedding.batch_size 与 embedder 的 max_batch_size 的较小者，因此 qwen 的 10 条上限自动生效。

自定义 embedder：继承 BaseEmbedder 并声明 backend 即被注册表收录。

```python
from rag_data.embedding import BaseEmbedder

class LocalEmbedder(BaseEmbedder):
    backend = 'local'
    default_model = 'bge-large-zh'
    default_dim = 1024
    default_base_url = 'http://localhost:8000/v1'
    api_key_env = 'LOCAL_API_KEY'

    def encode(self, texts):
        # 返回 List[List[float]]，顺序与输入一致
        ...
```

也可注入实现 encode(texts) 的对象绕过配置，便于测试：

```python
from rag_data.embedding import EmbedderHandler

embedder = EmbedderHandler(settings.embedding, model=my_model)
```

---

## 大模型

实现由 llm.provider 选择，内置 openai、qwen、deepseek；实例化在 LLMHandler 调用时进行。

```text
LLMHandler → resolve_llm_provider → Registry → OpenAILLMProvider
                                           ├ QwenLLMProvider
                                           └ DeepSeekLLMProvider
```

| provider | 默认模型 | 默认地址 | Key 环境变量 |
| :--- | :--- | :--- | :--- |
| openai | gpt-4o-mini | api.openai.com 的 v1 | OPENAI_API_KEY |
| qwen | qwen3.7-flash | DashScope compatible-mode | DASHSCOPE_API_KEY |
| deepseek | deepseek-chat | api.deepseek.com 的 v1 | DEEPSEEK_API_KEY |

LLMHandler(settings.llm, model=None)：

| 成员 | 说明 |
| :--- | :--- |
| provider（属性） | 注入的 model 优先，否则按配置装配并缓存 |
| chat(messages, **overrides) | 单轮对话，messages 为 role 与 content 组成的字典列表 |
| complete(prompt, system='', **overrides) | 单条用户消息完成一次生成，可选系统提示 |

```python
from rag_data import LLMSettings
from rag_data.rag_engine import _build_llm

llm_settings = LLMSettings(
    enable=True, provider='qwen', model='qwen3.7-flash', temperature=0.0,
)
llm = _build_llm(llm_settings)
print(llm.complete('用一句话概括 RAG', system='你是一名编辑'))
```

自定义 provider：继承 BaseLLMProvider 并声明 backend，实现 chat(messages, **overrides) 即可被注册表收录。


## 开发与测试

命令分类、测试清单与集成配置见 [development.md](development.md)。

## 许可证

MIT，见 [LICENSE](LICENSE)。
