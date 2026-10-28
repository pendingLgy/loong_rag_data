# rag-data 项目结构

> 版本 0.1.0，Python 3.8 及以上，src 布局。包根以 from-import 暴露 19 个公共名称，但 __all__ 目前只声明 __version__。

本文档描述仓库的物理结构与模块职责，是阅读代码与二次开发的导航图。使用方式见 README.md，开发、构建与测试说明见 development.md，评测落地计划见 plan 目录下的计划文档。

> 现状提醒：记忆层（memory）、关系库（storage/db）与向量库封装（storage/vector_store）已在重构中整体移除。当前只保留「解析 → 切块 → 向量化 → 大模型」的组件能力，落库与检索由 eval/rag 下的脚本直接调用 pymilvus 完成。

---

## 1. 概览

rag-data 提供两类能力：**解析向量化**（文件解析为分块并编码为稠密向量，也支持字符串直接向量化）与**离线评测**（EPUB 入库、检索问答、Ragas 测试集生成与打分）。两条线共用下层组件，但入口不同：RagEngine 负责解析与向量化，eval/rag 下的脚本自行组合解析器、嵌入器与 pymilvus 客户端。

| 层 | 包或文件 | 职责 |
| :--- | :--- | :--- |
| 配置层 | core/config.py | ParsingSettings、EmbeddingSettings、LLMSettings 与根配置 RagSetting；三类来源加载 |
| 契约层 | core/exceptions.py | 领域异常层次 |
| 通用设施 | core/registry.py、util/snowflake.py、nlp | 通用注册表、雪花 ID、spaCy 句柄与 BM25 词形 |
| 领域服务 | parsers、embedding、llm | 解析切块、向量化、大模型调用 |
| 评测脚本 | eval/rag | EPUB 入库、检索问答、Ragas 测试集生成与打分 |
| 门面层 | rag_engine.py | RagEngine 与私有装配方法 _build_* |
| 公共入口 | __init__.py | 以 from-import 暴露 core 的配置、异常与注册表 |

---

## 2. 目录结构

```text
rag-data/
├─ pyproject.toml                 构建与依赖声明（hatchling + hatch 环境）
├─ README.md                      使用说明
├─ development.md                 开发指南
├─ LICENSE                        MIT
├─ .env_rag_eval                  评测脚本的环境变量（含密钥，未纳入版本控制）
├─ .vcl/struct.md                本文件
├─ plan/RAG评测落地执行计划.md     评测落地计划
├─ dataset/                      评测数据集目录（当前为空）
├─ eval_outpu/                   评测报告输出目录（当前为空，目录名即 eval_outpu）
├─ src/rag_data/
│  ├─ __about__.py                版本号（hatch 版本来源）
│  ├─ __init__.py                 包根：以 from-import 暴露 core 的 19 个公共名称
│  ├─ py.typed                    类型标记（PEP 561）
│  ├─ rag_engine.py               RagEngine 门面与 _build_* 装配方法
│  ├─ core/
│  │  ├─ __init__.py              空
│  │  ├─ config.py                配置模型与三类来源加载
│  │  ├─ exceptions.py            领域异常
│  │  └─ registry.py              通用注册表（Registry、Ref）
│  ├─ nlp/
│  │  ├─ __init__.py              汇总导出 4 个符号
│  │  ├─ spacy.py                 load_nlp 与 get_nlp_lemma 双缓存、锁保护、失败也缓存
│  │  └─ lemmatization.py         BM25 词形：lemmatize_for_bm25、lemmatize_zh_for_bm25
│  ├─ parsers/
│  │  ├─ __init__.py              汇总导出与默认处理器的绑定方法
│  │  ├─ base.py                  DocumentParser 与递归切块原语
│  │  ├─ registry_parser.py       纯注册表
│  │  ├─ parser_handler.py        ParserHandler：预实例化并按扩展名派发
│  │  └─ markdown.py txt.py epub.py pdf.py word.py
│  ├─ embedding/                base.py registry_embedder.py openai_embedder.py qwen_embedder.py embedder_handler.py
│  ├─ llm/                      base.py registry_llm.py openai_llm.py qwen_llm.py deepseek_llm.py llm_handler.py
│  ├─ eval/rag/                （无 __init__.py，隐式命名空间包）
│  │  ├─ build_rag_data.py        EPUB 解析、切块、向量化并写入 Milvus
│  │  ├─ rag_data_search.py       向量检索 + 邻居窗口扩展 + 大模型回答
│  │  ├─ ragas_eval_dataset.py    Ragas 合成测试集并导出 CSV
│  │  └─ evaluate_rag.py          Ragas 指标打分并导出报告
│  └─ util/                     __init__.py（空）、snowflake.py（雪花 ID）
└─ tests/                       conftest.py 与 11 个 test_*.py
```

目录约定：

1. core 只承载配置、异常与注册表，不依赖任何领域服务；
2. eval/rag 没有 __init__.py，按隐式命名空间包使用，因此只能以模块路径导入（如 import rag_data.eval.rag.build_rag_data），不能 from rag_data.eval.rag import 具体符号；
3. 四个评测脚本都是可直接执行的入口脚本（含 __main__ 分支），不属于库的公共 API。

---

## 3. 分层与依赖方向

依赖自上而下，无环；较低层不反向导入较高层。

```text
__init__（包根，以 from-import 暴露 core 符号）
     ▲
rag_engine（RagEngine 门面，_build_nlp、_build_embedder、_build_llm）
     ├── core.config    配置模型与三类来源
     ├── nlp            spaCy 句柄与 BM25 词形
     ├── parsers        解析切块（ParserHandler）
     ├── embedding      向量化（EmbedderHandler）
     └── llm            大模型（LLMHandler）

core / util / nlp 为底层设施，不依赖 parsers、embedding、llm 与 rag_engine。
```

关键约定：

1. 可选依赖一律惰性使用（spacy、openai、PyYAML、EbookLib、pdfplumber、python-docx、pymilvus、pandas、dotenv、ragas、datasets、langchain-*），模块级不 import，缺依赖时包仍可导入；
2. 实现选择走注册表：声明 backend 即自动登记，新增实现不改动装配代码；
3. 注册表只登记实现类，实例化在各自的 handler 中进行；
4. 包根只暴露 core 的配置、异常与注册表名称，RagEngine 需从 rag_data.rag_engine 取；
5. nlp 不反向依赖 parsers 或 rag_engine，只提供句柄与词形函数；
6. eval/rag 不入门面：脚本只依赖 RagEngine、_build_llm、EmbeddingSettings 与 pymilvus。

---

## 4. 基础模块

### 4.1 __about__.py

仅保存 __version__（0.1.0），由 pyproject.toml 的 [tool.hatch.version] 读取。

### 4.2 __init__.py（包根）

以 from-import 暴露 19 个名称，但未列入 __all__：

| 分组 | 符号 |
| :--- | :--- |
| 版本 | __version__ |
| 配置 | ENV_NESTED_DELIMITER、ENV_PREFIX、SECTION_EMBEDDING、SECTION_PARSING、EmbeddingSettings、LLMSettings、ParsingSettings、RagSetting |
| 异常 | ConfigError、DataError、EmbeddingError、LLMError、OptionalDependencyError、ParseError、ParserDependencyError、RagDataError |
| 通用设施 | Ref、Registry |

__all__ 目前只含 __version__，所以 from rag_data import * 只能取到版本号；其余名称需显式 from rag_data import RagSetting，或按模块路径 from rag_data.core.config import RagSetting 取用。

---

### 4.3 core/config.py

职责：以嵌套模型承载配置，实例化即校验；提供三类来源的加载方法与一组私有工具函数。

```python
class ParsingSettings(BaseModel)    # enable、spacy_model、max_chars、safe_max_chars
class EmbeddingSettings(BaseModel)  # enable、provider、model、batch_size、dim、api_key、base_url、timeout
class LLMSettings(BaseModel)        # enable、provider、model、temperature、max_tokens、api_key、base_url、timeout
class RagSetting(BaseSettings)      # parsing、embedding
```

常量与映射：

| 符号 | 值或含义 |
| :--- | :--- |
| ENV_PREFIX | RAG_ |
| ENV_NESTED_DELIMITER | __（双下划线） |
| SECTION_PARSING、SECTION_EMBEDDING | parsing、embedding |
| MODULE_SECTIONS | [parsing, embedding]，RagSetting 允许的分区 |
| SECTION_MODEL_CLASSES | 分区名到分区模型的映射，用于补齐默认值 |

注意：LLMSettings 不由 RagSetting 承载（根配置只有 parsing 与 embedding 两个分区），它作为独立模型供 rag_engine 的 _build_llm 与评测脚本使用。

RagSetting 的模型配置：env_prefix 为 RAG_、env_nested_delimiter 为 __、env_file 为 .env、extra 为 ignore（根层忽略 RAG_ 命名空间下的无关变量）；三个分区模型均为 extra 为 forbid，未知字段直接抛 ValidationError。

类方法（语义对等，优先级均为 overrides 高于来源数据、来源数据高于字段默认值）：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| RagSetting.load_mapping(source=None, **overrides) | JSON 对象（分区字典） | 否 |
| RagSetting.load_env(environ=None, **overrides) | 环境变量（RAG_ 前缀） | 是 |
| RagSetting.load_yaml(path, **overrides) | YAML 文件 | 否（惰性导入 PyYAML） |

三个私有静态方法 _read_mapping、_read_env、_read_yaml 分别承接上述三类来源。

模块级函数：

| 函数 | 作用 |
| :--- | :--- |
| _load_env_overrides(environ=None) | 把 RAG_ 前缀环境变量组装为分区字典 |
| _filter_sections(data) | 只保留已知分区，忽略无关的 RAG_ 变量 |
| _validate_sections(*groups) | 校验分区名合法，非法抛 ConfigError |
| _merge_sections(base, overrides) | 按分区逐字段合并，实现覆盖 |
| _with_defaults(layered) | 用分区模型默认值补齐字段，屏蔽环境变量 |
| _with_model_defaults(model_cls, layered) | 按 model_fields 补齐顶层键；当前无调用方，为独立布局的设置类预留 |
| _as_mapping(source, loader) | 校验配置源为字典映射 |
| _as_yaml_mapping(path) | 读取 YAML 并校验为字典映射 |
| _assign_nested(data, parts, value) | 按路径写入嵌套字典 |
| _parse_env_value(raw) | JSON 对象与数组按 JSON 解析，其余交给 pydantic |
| _import_yaml() | 惰性导入 PyYAML，缺失时抛 OptionalDependencyError |

### 4.4 core/exceptions.py

单一根异常，便于上层统一捕获：

```text
RagDataError
├─ ConfigError             配置缺失或非法
├─ DataError               数据校验失败
├─ OptionalDependencyError 可选依赖未安装
│  └─ ParserDependencyError 文档解析依赖未安装
├─ ParseError              文档损坏或格式不支持
├─ EmbeddingError          向量化失败
├─ LLMError                大模型调用失败
└─ StorageError            存储读写失败（存储层已移除，当前无使用者）
```

### 4.5 core/registry.py

各子包注册表共用的 dict 子类，支持惰性导入；Ref 为注册值类型别名（点分路径或类对象）。

| 成员 | 说明 |
| :--- | :--- |
| Registry(kind, builtins) | kind 用于拼装报错信息，builtins 为初始登记表 |
| register(name, ref) | 登记类对象或点分路径 |
| decorator(name) | 装饰器形式的登记 |
| names() | 已登记名，字典序 |
| resolve(name) | 取实现类；路径登记的在此时惰性导入；未登记抛 ConfigError |

登记类对象而非路径，可让函数内定义的本地类也能注册；注册表不提供实例化方法。

### 4.6 util/snowflake.py

64 位雪花 ID：1 位符号 + 41 位毫秒时间戳 + 5 位数据中心 + 5 位工作节点 + 12 位序列，同一毫秒最多 4096 个 ID。

| 符号 | 说明 |
| :--- | :--- |
| EPOCH_MS | 起始纪元 1704067200000（2024-01-01 UTC，毫秒） |
| _WORKER_BITS 等位宽常量 | 5、5、12 位，配套掩码与位移常量 |
| SnowflakeGenerator(datacenter_id=0, worker_id=0) | 线程安全生成器，构造时校验节点位范围 |
| next_id() | 生成下一个 ID，加锁；时钟回拨时沿用最后时间 |
| __call__() | 等价于 next_id() |
| 模块级 next_id() | 用进程内单例 _DEFAULT_GENERATOR 取 ID |

当前仅被评测脚本用作 Milvus 主键；关系库已移除，故不再有表主键调用方。

### 4.7 日志（stdlib logging）

库内不提供日志配置：

| 约定 | 说明 |
| :--- | :--- |
| 模块级 logger | 各模块用 logging.getLogger(__name__) 自取 |
| 门面构造 | RagEngine(settings) 不接收 logger |
| 级别与格式 | 由调用方自行配置；库内只把字段拼为 key=value 后交给 logger 输出 |

---

## 5. 领域服务层

### 5.1 nlp（spaCy 句柄与 BM25 词形）

| 文件 | 内容 |
| :--- | :--- |
| __init__.py | 汇总导出 DEFAULT_MODEL、load_nlp、clear_cache、ensure_sentence_boundaries |
| spacy.py | 双缓存、锁保护，失败也缓存 |
| lemmatization.py | 两个纯函数：英文与中文 BM25 词形 |

spacy.py 符号：

| 符号 | 说明 |
| :--- | :--- |
| DEFAULT_MODEL | zh_core_web_md |
| _PIPELINES、_PIPELINES_LEMMA | 主句柄缓存与词形句柄缓存 |
| _FAILED | 失败哨兵；失败也写入缓存，避免逐次重试拖慢批量处理 |
| _lock | 两个缓存共用一把 threading.Lock，读路径先查缓存，写路径双重检查 |
| load_nlp(model_name) | 加载完整管线并补句边界；未装 spaCy 或模型缺失时告警并返回 None |
| get_nlp_lemma(model_name) | 加载词形管线（只 disable ner 与 parser），供 BM25 词形使用 |
| clear_cache() | 清空 _PIPELINES（含失败记录）；_PIPELINES_LEMMA 未清，见第 13 节 |
| ensure_sentence_boundaries(nlp) | 缺 parser 与 sentencizer 时补一个 sentencizer |

lemmatization.py：

| 函数 | 说明 |
| :--- | :--- |
| lemmatize_for_bm25(text) | 英文小写与词形还原，并追加 -ing 原词；spaCy 不可用时原样返回 |
| lemmatize_zh_for_bm25(text) | 中文场景：不做小写，保留下划线与连字符组成的复合标识符，过滤标点与停用词；写入侧与查询侧共用 |

### 5.2 parsers（解析与切块）

```text
文件 → parse_document → 纯文本 → chunk_document → 切块
```

| 文件 | 关键符号 | 说明 |
| :--- | :--- | :--- |
| registry_parser.py | BUILTIN_PARSERS、register_parser、register_parser_provider、is_parser_registered、available_parsers、resolve_parser_class、parser_classes | 纯注册表，只登记不实例化 |
| parser_handler.py | ParserHandler、PARSER_CLASSES | 构造时预实例化全部解析器，按扩展名派发 |
| base.py | DocumentParser | 抽象基类：SUFFIXES、backend、parse 与 chunk 契约，递归切块与四分之一重叠前缀 |
| txt.py | TxtParser | .txt，整篇递归切块 |
| markdown.py | MarkdownParser | .md、.markdown，按行首井号分章后递归切块 |
| epub.py | EpubParser | .epub，按 spine 提取章节后按井号分章递归切块 |
| pdf.py | PdfParser | .pdf，pdfplumber 按物理坐标提取、过滤页眉页脚后整篇递归切块 |
| word.py | WordParser | .docx，python-docx 提取段落，整篇递归切块 |

DocumentParser 的成员：

| 成员 | 说明 |
| :--- | :--- |
| SUFFIXES | 负责的扩展名集合（小写、含点），子类必须覆盖且不可重复 |
| backend | 声明即自动登记（__init_subclass__ 调用 register_parser） |
| __init__(max_chars=500, safe_max_chars=2000) | 切块长度上限；五个内置子类的默认 max_chars 均为 1000 |
| parse(path) | 抽象方法：文件到纯文本 |
| chunk(text, nlp=None) | 抽象方法：纯文本到切块；nlp 为 None 时各子类决定是否抛 ParserDependencyError |
| _read_text(path)（静态） | 读取 UTF-8 文本，供无结构格式共用 |
| split_sentences(text, nlp) | spaCy 分句后按换行符二次切分并清洗空行 |
| _extract_overlap_prefix(chunk_sentences) | 从段尾倒序拼够约四分之一字符作为下段前缀 |
| _process_text_recursive(text, overlap_prefix, nlp) | 递归切分主流程 |

切块规则要点：只有 text 长度超过 max_chars 才切句；首句超长时硬切；overlap_prefix 不占用 max_chars；用 find 推进消费偏移，偏移未推进时以 max_chars 兜底，避免死循环。

ParserHandler 成员：

| 成员 | 说明 |
| :--- | :--- |
| parser_classes、instances（属性） | 解析器类集合与已预实例化的实例副本 |
| get_instance(path) | 按扩展名取缓存实例，不新建 |
| resolve_parser(path, max_chars=None, safe_max_chars=None) | 长度与缓存一致时复用，不一致则临时新建 |
| parse_document(path) | 文件不存在抛 ParseError，否则派发 parse |
| chunk_document(path, text, nlp=None, max_chars=None, safe_max_chars=None) | 按扩展名派发 chunk |
| _build_index、_build_instances、_resolve、_needs_custom（静态） | 建后缀索引、建实例、按后缀取类、判断是否需临时新建 |

PARSER_CLASSES 在模块导入时由 registry_parser.parser_classes() 求得；后缀唯一性由构造时的断言保证；暂不支持的扩展名抛 ParseError。子包 __init__ 用默认处理器实例把 resolve_parser、parse_document、chunk_document 暴露为模块级函数，并再导出 DEFAULT_MODEL、load_nlp、clear_cache。

`pdf.PdfParser` 额外带 header_ratio 与 footer_ratio（默认 0.07），用于按页面高度比例过滤页眉页脚；`word.WordParser.parse` 遇到 .doc 直接抛 ParseError。

---

### 5.3 embedding（向量化）

```text
EmbedderHandler → resolve_embedder → Registry → OpenAIEmbedder
                                            └ QwenEmbedder
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseEmbedder 抽象：声明 backend 即自动登记；类属性 backend、default_model、default_dim、default_base_url、api_key_env、max_batch_size；属性 model、base_url、dim、api_key；抽象 encode |
| registry_embedder.py | BUILTIN_EMBEDDERS（openai、qwen）与 register_embedder、register_embedder_provider、is_embedder_registered、available_embedders、resolve_embedder |
| openai_embedder.py | OpenAIEmbedder：官方 openai SDK；client 惰性初始化，encode 调 embeddings.create，_build_request_kwargs 组装参数，_parse 按 index 还原顺序 |
| qwen_embedder.py | QwenEmbedder：继承 OpenAIEmbedder，改默认模型、端点、Key 变量与批量上限 |
| embedder_handler.py | EmbedderHandler：按配置解析并实例化 embedder，统一批次与维度校验 |

| embedder | 默认模型 | 默认维度 | Key 环境变量 | 批量上限 |
| :--- | :--- | :--- | :--- | :--- |
| openai | text-embedding-3-small | 1536 | OPENAI_API_KEY | 无 |
| qwen | qwen3.7-flash | 1024 | DASHSCOPE_API_KEY | 10 |

EmbedderHandler(settings, model=None)：注入的 model 优先且绕过配置；provider 首次访问时实例化并缓存。

| 成员 | 说明 |
| :--- | :--- |
| provider（属性） | 注入的 model 优先，否则按 embedding.provider 装配并缓存 |
| encode(texts) | 分批编码，空输入返回空列表，结果按 embedding.dim 校验 |
| _target() | 解析并实例化 embedder |
| _batch_size(target) | 取配置值与 max_batch_size 的较小者，上限为 0 表示不限制 |
| _encode_batch(target, batch) | 编码单批并转为 float 列表 |
| _validate_dim(vectors) | 维度与 embedding.dim 不一致时抛 EmbeddingError |

OpenAIEmbedder 只在配置维度与 default_dim 不同时才下发 dimensions，以兼容不支持该参数的模型。

### 5.4 llm（大模型调用）

```text
LLMHandler → resolve_llm_provider → Registry → OpenAILLMProvider
                                           ├ QwenLLMProvider
                                           └ DeepSeekLLMProvider
```

| 文件 | 内容 |
| :--- | :--- |
| base.py | BaseLLMProvider 抽象：声明 backend 即自动登记；属性 model、base_url、temperature、max_tokens、api_key；抽象 chat |
| registry_llm.py | BUILTIN_PROVIDERS（openai、qwen、deepseek）与 register_llm、register_llm_provider、is_llm_registered、available_llm_providers、resolve_llm_provider |
| openai_llm.py | OpenAILLMProvider：client 惰性初始化，chat 调 chat.completions.create，_build_request_kwargs 组装参数，_parse 取首条 content |
| qwen_llm.py | QwenLLMProvider：继承 OpenAILLMProvider，改默认模型、端点与 Key 变量 |
| deepseek_llm.py | DeepSeekLLMProvider：同上，指向 DeepSeek 开放平台 |
| llm_handler.py | LLMHandler：按配置解析并实例化 provider，统一调用入口 |

| provider | 默认模型 | 默认地址 | Key 环境变量 |
| :--- | :--- | :--- | :--- |
| openai | gpt-4o-mini | api.openai.com 的 v1 | OPENAI_API_KEY |
| qwen | qwen3.7-flash | DashScope compatible-mode | DASHSCOPE_API_KEY |
| deepseek | deepseek-chat | api.deepseek.com 的 v1 | DEEPSEEK_API_KEY |

LLMHandler(settings, model=None)：注入的 model 优先且绕过配置。

| 成员 | 说明 |
| :--- | :--- |
| provider（属性） | 注入的 model 优先，否则按 llm.provider 装配并缓存 |
| chat(messages, **overrides) | 单轮对话，返回回复文本 |
| complete(prompt, system='', **overrides) | 组装 system 与 user 消息后调用 chat |
| _target() | 解析并实例化 provider |

temperature 与 max_tokens 取自 llm 分区；max_tokens 为 0 时不显式传入，交由服务端决定上限。

---

## 6. 门面层 rag_engine.py

模块级装配方法均为私有实现，且受对应分区的 enable 门控（为 False 时返回 None）：

| 方法 | 说明 |
| :--- | :--- |
| _build_nlp(parsing_settings) | 调 nlp.spacy.load_nlp 加载 spaCy；分区关闭或模型不可用时返回 None |
| _build_embedder(embedding_settings, model=None) | 装配 EmbedderHandler |
| _build_llm(llm_settings, model=None) | 装配 LLMHandler；评测脚本 rag_data_search 与 ragas_eval_dataset 会直接复用它 |

RagEngine(settings: RagSetting)：

| 成员 | 说明 |
| :--- | :--- |
| settings、parser、nlp、embedder | 构造后挂载的实例变量。parser 始终是 ParserHandler（构造时把 parsing.max_chars 与 parsing.safe_max_chars 作为预置长度）；nlp 与 embedder 在分区关闭或不可用时为 None |
| parser_handler（属性） | 解析派发入口，返回 ParserHandler 实例 |
| parse_document(path) | 委托 parser 解析单文件为纯文本 |
| chunk_document(path, text) | 委托 parser 切块，长度取自 parsing 分区，并注入 nlp 句柄 |

__all__ 只含 RagEngine。门面不再持有 llm、db、vector，也没有 MemoryServer；类文档字符串中仍留有 MemoryServer 的表述，见第 13 节。

---

## 7. 评测脚本 eval/rag

四个入口脚本，位于隐式命名空间包下，不属于库的公共 API。环境变量由 python-dotenv 从仓库根的 .env_rag_eval 读取。

| 脚本 | 职责 | 关键符号 |
| :--- | :--- | :--- |
| build_rag_data.py | EPUB 解析、切块、向量化并写入 Milvus | RAG_EVAL_JSON、PARSING_SETTINGS、EMBEDDING_SETTINGS、LLM_SETTINGS、COLLECTION、DIM、MILVUS_*、_create_milvus_collection、_connect_milvus、_build_rag_engine、_build_rows、_insert_milvus、main |
| rag_data_search.py | 向量检索 + 邻居窗口扩展 + 大模型回答 | fetch_neighbor_contexts、rag_call、_create_milvus_collection、_connect_milvus |
| ragas_eval_dataset.py | 从集合取 chunks，用 ragas TestsetGenerator 合成测试集并导出 CSV | _connect_milvus（AsyncMilvusClient）、build_milvus_client、safe_generate_testset、build_chunk_rag_test_docs、main |
| evaluate_rag.py | 逐条检索与生成，再用 ragas 指标打分并导出报告 | MyRAGSystem（_get_client、embed_query、retrieve_from_milvus、generate_answer）、build_openai_client、main、setup_windows_event_loop |

Milvus 集合 schema（build_rag_data 与 rag_data_search 各存一份）：id（INT64 主键）、vector（FLOAT_VECTOR，维度取嵌入配置）、text、text_lemma（enable_analyzer）、title、title_md5、source、sparse（BM25 函数由 text_lemma 生成）；索引为 vector_index（AUTOINDEX、COSINE）、sparse_index（SPARSE_INVERTED_INDEX、BM25）与 title_md5_index（INVERTED）。集合名固定为 eval_rag_data。

环境变量（.env_rag_eval）：

| 键 | 用途 |
| :--- | :--- |
| milvus_url、db_name、collection_name | Milvus 连接与集合 |
| dataset_path | evaluate_rag 读取的测试集 CSV |
| rag_eval_output_path | 评测报告输出目录 |
| embedding_model、embedding_url、embedding_api_key | 嵌入客户端 |
| llm_model、llm_url、llm_api_key | 大模型客户端 |

评测指标：context_precision、context_recall、faithfulness 与 AnswerAccuracy，统一用 LangchainLLMWrapper 包装 ChatOpenAI，结果写到 rag_evaluation_report.csv。Windows 下显式切换 Proactor 事件循环策略。

---

## 8. 数据流

### 8.1 文件向量化

```text
1. RagEngine(settings)                装配 parser、nlp、embedder（按 enable 门控）
2. app.parse_document(path)           解析文件为纯文本
3. app.chunk_document(path, text)     按格式切块（需要 nlp 句柄）
4. app.embedder.encode(chunks)        分块编码为稠密向量
5. 调用方自行落库                    库内不再提供向量库封装，评测脚本直接用 pymilvus
```

### 8.2 字符串向量化

```text
app.embedder.encode(texts)            批量字符串到向量列表（与输入同序）
    ├─ 分批：embedding.batch_size 与 embedder 上限取较小者
    └─ 校验：逐条按 embedding.dim 校验，不一致抛 EmbeddingError
```

### 8.3 评测链路

```text
build_rag_data.py      文件 → RagEngine → chunks → encode → Milvus（text_lemma 由 lemmatize_zh_for_bm25 生成）
rag_data_search.py     query → encode → Milvus search → fetch_neighbor_contexts → LLMHandler.complete
ragas_eval_dataset.py  Milvus query(limit) → ragas TestsetGenerator → testset.csv
evaluate_rag.py        testset.csv → 逐条检索与生成 → Ragas aevaluate → rag_evaluation_report.csv
```

---

## 9. 配置项

RagSetting 的分区与字段（括号内为默认值）：

| 分区 | 字段 |
| :--- | :--- |
| parsing | enable(false)、spacy_model(zh_core_web_sm)、max_chars(500)、safe_max_chars(2000) |
| embedding | enable(false)、provider(openai)、model(空)、batch_size(128)、dim(1024)、api_key(空)、base_url(空)、timeout(60.0) |

独立模型 LLMSettings（不属于 RagSetting 的分区，由调用方自行构造并传给 _build_llm）：

| 字段 | 默认值 |
| :--- | :--- |
| enable | false |
| provider | openai |
| model | 空（回落到 provider 的 default_model） |
| temperature | 0.7 |
| max_tokens | 0（不显式传入） |
| api_key、base_url | 空（分别回落到环境变量与默认地址） |
| timeout | 60.0 |

三类来源：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| RagSetting.load_mapping | JSON 对象（分区字典） | 否 |
| RagSetting.load_env | 环境变量 | 是（可显式传 environ） |
| RagSetting.load_yaml | 配置文件 | 否（惰性导入 PyYAML） |

环境变量命名：RAG_ 前缀，分区与字段名大写，双下划线分隔。

```bash
RAG_PARSING__ENABLE=true
RAG_PARSING__SPACY_MODEL=zh_core_web_sm
RAG_PARSING__MAX_CHARS=1000
RAG_EMBEDDING__ENABLE=true
RAG_EMBEDDING__PROVIDER=qwen
RAG_EMBEDDING__DIM=1024
```

enable 语义与联动：

| 分区 | 关闭后的表现 |
| :--- | :--- |
| parsing | RagEngine.nlp 为 None；切块时各子类发现句柄缺失会抛 ParserDependencyError |
| embedding | RagEngine.embedder 为 None；字符串向量化不可用 |

维度一致性：embedding.dim 声明期望维度（默认 1024），EmbedderHandler 据此逐条校验；provider 不支持自定义维度时保持与 default_dim 一致，避免请求被拒。

---

## 10. 扩展点

各领域的实现选择都走注册表：继承抽象基类并声明 backend 即被收录，新增实现无需改动装配代码。

| 扩展点 | 抽象基类 | 声明方式 | 配置键 | 内置 |
| :--- | :--- | :--- | :--- | :--- |
| 文档解析 | DocumentParser | backend 与 SUFFIXES | 无（按扩展名派发） | txt、markdown、epub、pdf、word |
| 嵌入模型 | BaseEmbedder | backend 类属性 | embedding.provider | openai、qwen |
| 大模型 | BaseLLMProvider | backend 类属性 | llm.provider | openai、qwen、deepseek |

自定义实现也可显式登记：register_parser、register_embedder、register_llm，或使用装饰器形式 register_parser_provider、register_embedder_provider、register_llm_provider。

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

---

## 11. 测试

测试目录共 12 个文件（conftest.py 加 11 个用例文件），按 def test_ 静态计数 177 项：

| 文件 | 用例数 | 覆盖范围 |
| :--- | :--- | :--- |
| conftest.py | 无 | 路径注入、加载 .env_ai_key、分区 JSON 到 Settings 对象、settings 夹具、--integration 选项 |
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

标记与门控：pyproject 声明 integration 与 eval 两个标记，addopts 为 -s -q -rs -m 'not integration'；当前没有用例使用这两个标记，conftest 只注册了 --integration 选项且暂无消费方。

依赖缺失时用例用局部助手函数打印提示并跳过（如句柄缺失、pdfplumber 与 python-docx 未安装）。

---

## 12. 依赖

核心依赖三项：pydantic、pydantic-settings 与 PyYAML（供 load_yaml 使用）。其余按用途分组为可选依赖（extras），声明见 pyproject.toml。

| 组 | 包 |
| :--- | :--- |
| 核心 | pydantic>=2.5、pydantic-settings>=2.1、PyYAML>=6.0.3 |
| parsers | unstructured、spacy>=3.8.16、zh_core_web_sm、zh_core_web_md |
| epub | EbookLib>=0.20、beautifulsoup4>=4.15.0 |
| pdf | pdfplumber==0.11.10 |
| word | python-docx>=1.0 |
| milvus | pymilvus>=2.3 |
| openai | openai>=3.26.0（内置 openai 与 qwen provider、embedder 依赖） |
| ragas-eval | langchain-core>=1.7.0、langchain-openai>=1.7.2、langchain-google-community>=5.1.1、langchain-community==0.4.1、ragas>=0.4.3 |
| dev | pytest>=7.4、pytest-asyncio>=1.4.0、pytest-cov>=4.1、mypy>=1.8、ruff>=0.1.9 |

hatch 环境（按 features 自动安装）：

| 环境 | features |
| :--- | :--- |
| default | parsers、openai、dev |
| rag_data_dev | parsers、milvus、dev、word、pdf、epub、ragas-eval |

注意：评测脚本还直接 import pandas、dotenv（python-dotenv）与 datasets，这三个包未在任何 extra 中声明，需手动安装。

---

## 13. 待补全

| 位置 | 事项 |
| :--- | :--- |
| 包根 | __all__ 只声明 __version__，与 19 个 from-import 名称不一致；应补齐或改为显式导出 |
| rag_engine | 类文档字符串仍提到 MemoryServer 与记忆侧装配，需与现状对齐 |
| config | _with_model_defaults 已无调用方（原为独立布局的设置类预留），可删或复用 |
| exceptions | StorageError 与 DataError 当前无使用者 |
| eval | eval 与 eval/rag 均无 __init__.py，只能按模块路径导入脚本 |
| eval | pandas、python-dotenv、datasets 未在 extras 中声明 |
| eval | ragas_eval_dataset.py 的 BASE_DIR 层级与另三个脚本不同，.env_rag_eval 可能定位失败 |
| eval | build_rag_data.main 的文件路径、rag_data_search 的问题文本与集合名均为硬编码，宜改为参数或环境变量 |
| eval | build_rag_data 中 batch = chunks[:1] 为调试残留，且 hashlib 重复 import |
| eval | build_rag_data 与 rag_data_search 各自复制了一份建集合与连接逻辑，可抽为共用模块 |
| parsers | _process_text_recursive 的 nlp 默认值是 typing.Any 对象而非 None，调用时应显式传句柄 |
| parsers | 版面级解析仍待细化：PDF 双栏、Word 表格 |
| nlp | clear_cache 只清主缓存，未清 _PIPELINES_LEMMA，热重建会读到陈旧词形管线 |
| nlp | lemmatize_for_bm25 目前只有测试在用，英文路径未接入业务 |
| embedding | 缺少本地模型后端（如 sentence-transformers） |
| tests | conftest 的 MILVUS、PG、MYSQL、VECTOR、CASES 分区与 --integration 选项在集成用例移除后已无消费方 |
| tests | test_spacy.py 以打印结果为主，断言覆盖偏弱 |
| 目录 | dataset/ 与 eval_outpu/ 为空目录，后者拼写为 eval_outpu |
| 评测 | 评测报告与阈值未纳入 CI，基线需人工比对 |
