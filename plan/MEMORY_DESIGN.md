# rag-data memory 子模块架构设计（同步四流程）

> 版本：草案 v0.1　　状态：待评审
> 参考：`.vcl\memory_main_analysis.md`（mem0 主模块代码分析）、`rag_artifact.md`、`plan\DEVELOPMENT_PLAN.md`
> 范围：**仅同步方法**，实现 `add`、`search`、`update`、`delete` 四条流程。异步版本、`delete_all`、`reset` 等均不在本期。

---

## 0. 背景与定位

rag-data 现有能力止于「解析、切块、向量化、存储」四段，缺少把它们串成「记忆增删改查」的业务层。mem0 的 `Memory` 类正是这一层，但其实现同时承载了同步、异步、实体图谱、时序记忆、衰减策略、遥测与 UX 提示等十余种关注点，单文件 3868 行。

本设计的目标是：**在 rag-data 既有分层与注册表机制之上，落地一个职责收敛的同步记忆层**，只覆盖最核心的四条流程，为后续扩展留出清晰接缝，而不复制 mem0 的复杂度。

设计原则：

1. **复用优先**：向量读写走 `storage.vector_store.VectorEngine`，历史与消息走 `storage.db.SQLDatabase`，嵌入走 `embedding.EmbedderHandler`，大模型走 `llm.LLMHandler`，不新增第二套存储或客户端。
2. **单流程单一入口**：四条流程各对应一个公共方法，内部实现拆成不可再分的私有步骤，每步可独立单测。
3. **失败可判别**：不静默吞异常；按 rag-data 既有异常层次抛错，让调用方能区分「配置错」「依赖缺」「模型不可用」「存储失败」。
4. **作用域不可越权**：`user_id`、`agent_id`、`run_id` 三元组只从显式参数进入，metadata 中的同名字段一律剥离。


---

## 1. 范围界定

### 1.1 本期实现

| 流程 | 公共方法 | 一句话职责 |
| :--- | :--- | :--- |
| 写入 | `add(messages, **scope)` | 把对话或文本转成记忆条目，支持 LLM 抽取与原始写入两种模式 |
| 检索 | `search(query, **scope)` | 语义与关键词双路召回，融合排序后返回带分数的记忆条目 |
| 更新 | `update(memory_id, text=None, metadata=None)` | 改文本或元数据，必要时重算向量 |
| 删除 | `delete(memory_id)` | 软删记忆，保留历史审计 |

配套实现：作用域构造、消息归一、抽取提示词、哈希去重、批量嵌入、历史写入、结果格式化。

### 1.2 本期不实现（明确排除）

| 项目 | 排除理由 | 后续接缝 |
| :--- | :--- | :--- |
| `AsyncMemory` | 异步版与同步版平均源码相似度 0.815，双份维护易漂移；先固化同步语义 | handler 内业务逻辑与 I/O 分离，后续可加 `to_thread` 适配层 |
| `delete_all`、`reset` | 涉及分页循环删除与集合重建，属运维接口 | 复用 `SQLDatabase.reset` 与 `VectorEngine.reset` |
| 流程记忆（procedural） | 需额外提示词模板与元数据类型字段 | `add` 的 `memory_type` 参数先保留位、不实现分支 |
| 过期记忆、时序记忆、衰减策略 | mem0 中与平台能力耦合，rag-data 无对应配置 | 预留 metadata 键 `expiration_date`，仅透传不判定 |
| 遥测、首次运行提示、慢查询提示 | 属产品侧 UX，非架构必需 | 以日志替代 |


### 1.3 与 mem0 的取舍对照

| 关注点 | mem0 做法 | 本设计 | 理由 |
| :--- | :--- | :--- | :--- |
| 构造参数 | `config: MemoryConfig = MemoryConfig()` | `settings: MemorySettings` | 规避可变默认参数在导入期共享同一实例 |
| 关键词召回 | 仅用于对语义候选重排 | **BM25 候选并入候选集**后再融合 | 修复 mem0 的召回缺陷：纯关键词命中会被丢弃 |
| 数字 ID 映射 | 建立 `uuid_mapping` 后未使用 | 不做该映射 | 死代码，且 LLM 抽取改用文本去重 |
| 错误类型 | 同一语义混用 `ValueError` 与 `Mem0ValidationError` | 统一 `ConfigError`、`DataError`、`StorageError`、`LLMError`、`EmbeddingError` | 与 rag-data 异常体系一致，调用方可按类型分流 |
| 大方法 | `_add_to_vector_store` 328 行 | 拆为 6 个私有步骤，单步可测 | 可读性与可测性 |
| 线程池 | 每次检索新建 `ThreadPoolExecutor(4)` | 本期串行执行 | 四流程无并发需求，避免线程抖动 |
| 主键 | UUID 字符串 | **雪花 ID（BIGINT）** | 与 `storage.db` 现有 schema 对齐，时间趋势递增 |


---

## 2. 依赖关系与分层

memory 位于门面层之下、存储与模型服务之上，只依赖下层能力，不反向被下层引用。

```text
RagData / MemoryServer（门面）
     │
MemoryHandler（本期新增）
     ├── config        读取 MemorySettings（llm、db、vector 及各自 enable 开关）
     ├── embedding     文本到向量（EmbedderHandler）
     ├── llm           事实抽取（LLMHandler）
     ├── storage.db    历史与消息（SQLDatabase）
     └── storage.vector_store  向量读写（VectorEngine）
```

**强依赖约束**：

1. `memory` 不得直接 `import` 具体方言实现（`SQLiteDB`、`MilvusVectorStore` 等），只依赖抽象基类与 engine。
2. `memory` 不得新开数据库连接或向量客户端，一律从门面注入实例。
3. 四条流程均为同步阻塞，不引入 `asyncio`。


---

## 3. 目录结构

```text
src\rag_data\memory\
├─ __init__.py        汇总导出：MemoryHandler、MemoryBase、MemoryItem 等
├─ base.py            MemoryBase 抽象接口（四流程契约）
├─ handler.py         MemoryHandler：装配四个组件并实现四条流程
├─ scope.py           作用域构造与校验（纯函数，无外部依赖）
├─ prompts.py         抽取提示词（逐字沿用 mem0）
├─ models.py          MemoryItem、AddResult、SearchResult 等数据对象
└─ stores.py          向量库与关系库的薄封装（统一 id 转换与 payload 布局）
```

**新增文件与复用边界**：

| 能力 | 来源 |
| :--- | :--- |
| 文本向量化 | 复用 `embedding.EmbedderHandler` |
| 事实抽取 | 复用 `llm.LLMHandler` |
| 历史与消息持久化 | 复用 `storage.db.SQLDatabase`（`add_history`、`batch_add_history`、`save_messages`、`get_last_messages`） |
| 向量读写 | 复用 `storage.vector_store.VectorEngine`（`insert`、`search`、`keyword_search`、`update`、`delete`、`get`） |
| 雪花 ID | 复用 `storage.db.snowflake.next_id` |

**明确不引入的能力**：本期不建立实体图谱，也不创建第二套向量集合。

| 项 | 说明 |
| :--- | :--- |
| 实体抽取 | 不调用实体识别，不产生实体记录 |
| 实体加权 | 检索打分只由语义分与关键词分构成，不含实体提升项 |
| 实体清理 | `update` 与 `delete` 不做实体反向链接维护 |

去除后 `add`、`update`、`delete` 少一次嵌入与一次外部调用，`search` 少一轮并发查询；代价是「同一实体不同表述」的召回只能依赖向量与关键词，不再有图谱扩散。

---

## 4. 配置分区

`MemorySettings` 独立定义于 `config.py`，**不是 `RagSetting` 的子分区**：`RagSetting` 只保留 `parsing`、`embedding`，记忆层配置自成一套并带独立前缀 `RAG_MEMORY_`，使 Rag 与 Memory 可分别装配、分别配置。`MODULE_SECTIONS` 与 `SECTION_MODEL_CLASSES` 均只含这两个分区，故把 `memory` 作为 `RagSetting` 的分区键传入会抛 `ConfigError`。

```python
MEMORY_ENV_PREFIX = ENV_PREFIX + "MEMORY_"   # 取值为 "RAG_MEMORY_"


class MemorySettings(BaseSettings):
    """记忆分区：与 RagSetting 相互独立，自带 llm、db、vector 三类子配置。"""

    model_config = SettingsConfigDict(
        env_prefix=MEMORY_ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        # 键范围已由前缀限定，故保持严格模式：未知字段直接报错，避免拼写错误被静默忽略。
        extra="forbid",
    )

    # 分区开关；关闭时门面不装配 MemoryHandler。
    enable: bool = Field(default=False)

    # 三类依赖：大模型（事实抽取）、关系库（历史与消息）、向量库（记忆检索）。
    llm: LLMSettings = Field(default_factory=LLMSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
```

字段清单（括号内为默认值）：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `enable` | `bool`（`False`） | 记忆分区开关；关闭时 `MemoryServer.memory` 为 `None` |
| `embedding` | `EmbeddingSettings` | 事实向量化所用嵌入器，独立于解析侧的 embedding 分区 |
| `llm` | `LLMSettings` | 事实抽取所用大模型，沿用 `llm` 分区字段 |
| `db` | `DBSettings` | 历史与消息持久化，沿用 `db` 分区字段 |
| `vector` | `VectorSettings` | 记忆检索所用向量库，沿用 `vector` 分区字段 |

**自带 `embedding` 子分区**：记忆侧的嵌入器与解析侧分属两套配置，`MemoryServer` 据此装配独立的 `EmbedderHandler`，互不影响。

**检索与写入默认值下沉为模块常量**：`threshold`、`top_k`、`context_top_k`、`keyword_weight`、`history_messages` 不再作为配置字段，改由 `memory` 包 `handler.py` 的常量承接，避免「配置项与代码默认值」双份漂移：

| 常量 | 值 | 用途 |
| :--- | :--- | :--- |
| `DEFAULT_THRESHOLD` | `0.1` | 语义召回的相似度下限 |
| `DEFAULT_TOP_K` | `20` | 检索默认返回条数 |
| `DEFAULT_CONTEXT_TOP_K` | `10` | 抽取前召回的既有记忆条数 |
| `DEFAULT_KEYWORD_WEIGHT` | `0.5` | BM25 分在融合时的权重 |
| `DEFAULT_HISTORY_MESSAGES` | `10` | 写路径给模型的历史消息条数 |

记忆集合名沿用 `VectorSettings.collection`，记忆侧不再提供覆盖项。

**已删除的旧子配置**：`MemoryModelSettings`、`MemoryEmbeddingSettings`、`MemoryLLMSettings`、`MemoryNLPSettings`、`MemoryRerankerSettings` 一并移除——抽取模型由 `llm` 分区承接，嵌入由注入的 `EmbedderHandler` 承接，spaCy 句柄由 `parsing` 分区承接，reranker 无内置后端故不保留配置位。

### 4.1 三类来源

与 `RagSetting` 对等，`MemorySettings` 提供三条加载入口：

| 方法 | 来源 | 是否读环境 |
| :--- | :--- | :--- |
| `load_mapping(source=None, **overrides)` | JSON 对象（字段字典） | 否 |
| `load_env(environ=None, **overrides)` | 环境变量 | 是（可显式传 `environ`） |
| `load_yaml(path, **overrides)` | YAML 文件 | 否（`PyYAML` 惰性导入） |

`load_mapping` 与 `load_yaml` 严格不读环境：两者先用 `model_fields` 的默认值补齐字段（`_with_model_defaults`）再叠加来源数据，借此屏蔽 `BaseSettings` 构造期自动读取环境变量的行为；只有 `load_env` 与直接构造实例才读环境。三类来源统一按「`overrides` 高于来源数据、来源数据高于字段默认值」合并。

环境变量命名：前缀 `RAG_MEMORY_`，嵌套用双下划线，且兼容 `RAG_MEMORY__DB__BACKEND` 与 `RAG_MEMORY_DB__BACKEND` 两种写法（剥离前缀后再去一个可选下划线）：

```bash
RAG_MEMORY__ENABLE=true
RAG_MEMORY__DB__ENABLE=true
RAG_MEMORY__VECTOR__BACKEND=pg
```

`extra="forbid"` 使未知字段（含已移除的 `nlp`、`threshold` 等）直接抛 `ValidationError`。

### 4.2 装配与门控

| 条件 | 行为 |
| :--- | :--- |
| `enable` 为假 | `facade._build_memory` 返回 `None`，`MemoryServer.memory` 为 `None`；四条流程经 `_require()` 抛 `ConfigError` |
| `vector.enable` 为假 | `MemoryHandler` 构造期 `_validate()` 抛 `ConfigError`：向量库是硬依赖 |
| `db.enable` 为假 | `ConfigError`：历史与消息持久化是硬依赖 |
| `llm.enable` 为假 | `ConfigError`：写入一律经大模型抽取，llm 是硬依赖 |
| `embedding.enable` 为假 | `ConfigError`：记忆层需要嵌入器，请启用 embedding 分区 |

装配分工：`MemoryServer` 只接收 settings，内部按 `settings.embedding`、`settings.llm`、`settings.db`、`settings.vector` 装配四类组件；各模块用 `logging.getLogger(__name__)` 自取日志句柄，门面不接收也不透传 logger；四项硬依赖齐备后构造 `MemoryHandler` 并在构造期完成上述校验。

---

## 5. 数据模型与主键策略

### 5.1 主键

沿用 `storage.db` 的雪花 ID（BIGINT），保证「按创建时间可排序」且与 `history` 表 `memory_id` 列类型一致。三处表示需显式转换：

| 位置 | 类型 | 转换 |
| :--- | :--- | :--- |
| 记忆主键（业务视角） | `int` | `next_id()` 生成 |
| 向量库 id | `str` | `str(memory_id)` |
| 历史表 `memory_id` | `int` | 原值 |

转换集中在 `stores.py`，避免各处重复 `str()` 造成不一致。

### 5.2 payload 布局

向量库 payload 固定核心键，其余键原样保留为业务元数据：

| 键 | 类型 | 说明 |
| :--- | :--- | :--- |
| `data` | `str` | 记忆正文（检索结果回填为 `memory` 字段） |
| `text_lemmatized` | `str` | BM25 词形还原文本，供关键词检索 |
| `hash` | `str` | 正文 MD5，用于去重 |
| `created_at` | `str` | ISO 8601，UTC |
| `updated_at` | `str` | ISO 8601，UTC |
| `user_id`、`agent_id`、`run_id` | `str` | 作用域，不可经 metadata 设置 |
| `actor_id`、`role` | `str` | 可选，来源为消息字段 |

### 5.3 对外数据对象

```python
class MemoryItem(BaseModel):
    id: int
    memory: str
    score: Optional[float] = None
    hash: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
```

`add` 返回 `AddResult`（含 `results` 列表，每项为 `id`、`memory`、`event`），`search` 返回 `SearchResult`（含 `results` 列表，每项为 `MemoryItem`）。


---

## 6. 模块职责与接口签名

### 6.1 `base.py`：抽象契约

```python
class MemoryBase(ABC):
    @abstractmethod
    def add(self, messages, *, user_id=None, agent_id=None, run_id=None,
            metadata=None): ...

    @abstractmethod
    def search(self, query, *, filters=None, top_k=None, threshold=None): ...

    @abstractmethod
    def update(self, memory_id, text=None, metadata=None): ...

    @abstractmethod
    def delete(self, memory_id): ...
```

四个方法即本期全部公共契约；参数默认值一律 `None`，真实默认值从配置读取，避免「方法与配置双份默认值漂移」。

### 6.2 `scope.py`：作用域（纯函数）

| 函数 | 签名 | 职责 |
| :--- | :--- | :--- |
| `validate_entity_id` | `(value, name) -> Optional[str]` | 非字符串强转、去首尾空白、拒空、拒内部空白 |
| `strip_identity_keys` | `(metadata, existing, context) -> Dict` | 剥离 `user_id`、`agent_id`、`run_id`、`actor_id` |
| `build_scope` | `(*, user_id, agent_id, run_id, metadata, filters) -> Tuple[Dict, Dict]` | 返回「存储用 payload 模板」与「查询用 filters」，强制至少一个实体 ID |
| `session_scope` | `(filters) -> str` | 生成确定性会话键，供消息表归组 |
| `validate_search_params` | `(threshold, top_k) -> None` | 阈值属于零到一、条数为非负整数（排除布尔） |

除 `build_scope` 外均无副作用；`build_scope` 只做深拷贝与字典拼接。


### 6.3 `models.py` 与 `prompts.py`

| 文件 | 内容 |
| :--- | :--- |
| `models.py` | `MemoryItem`、`AddResult`、`SearchResult`、`UpdateResult`、`DeleteResult`（均为 pydantic 模型，`extra=ignore`） |
| `prompts.py` | 逐字沿用 mem0：常量 `ADDITIVE_EXTRACTION_PROMPT`、`AGENT_CONTEXT_SUFFIX`、`PAST_MESSAGE_TRUNCATION_LIMIT`；渲染函数 `generate_additive_extraction_prompt` 及其 6 个私有辅助 |

提示词与 mem0 完全一致、不做改写：系统提示走 `ADDITIVE_EXTRACTION_PROMPT`（agent 作用域追加 `AGENT_CONTEXT_SUFFIX`），用户提示由 `generate_additive_extraction_prompt` 渲染，输出段落固定为 Summary、Last k Messages、Recently Extracted Memories、Existing Memories、New Messages、Observation Date、Current Date、Output 与可选的 Custom Instructions。

### 6.4 `handler.py`：门面实现

```python
class MemoryHandler:
    def __init__(self, settings: MemorySettings,
                 embedder: Optional[EmbedderHandler] = None,
                 llm: Optional[LLMHandler] = None,
                 db: Optional[BaseDB] = None,
                 vector: Optional[VectorEngine] = None) -> None: ...
```

**构造约定**：四个组件均可注入，便于测试；未注入时由 `MemoryServer` 按 `MemorySettings` 的对应子分区自行装配。构造期执行 `_validate()` 校验四项硬依赖，但不发起任何网络或数据库请求（连接由被注入实例自行持有）。

| 成员 | 说明 |
| :--- | :--- |
| `settings`、`embedder`、`llm`、`db`、`vector` | 构造后挂载 |
| `add`、`search`、`update`、`delete` | 四条流程公共方法 |


---

## 7. 四条流程设计

### 7.1 `add`：写入

公共入口只做参数归一与分派，真正的写入在三个私有步骤中完成。

| 步骤 | 方法 | 输入 | 输出 | 失败语义 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `_normalize_messages` | 字符串、字典或列表 | 统一的 `List[Message]` | 类型非法抛 `DataError` |
| 2 | `_extract_facts` | 归一消息、既有记忆、历史 | `List[str]` 事实文本 | 模型失败抛 `LLMError`；响应非 JSON 时降级为空列表并告警 |
| 3 | `_persist_facts` | 事实文本列表、payload 模板 | `AddResult` | 嵌入失败抛 `EmbeddingError`，写库失败抛 `StorageError` |

写入一律经大模型抽取；`infer` 开关已移除，避免出现「不抽取也算记忆」的歧义路径。

去重规则（两重）：

1. **库内去重**：抽取前用 `DEFAULT_CONTEXT_TOP_K` 召回既有记忆，取其 `hash` 集合；
2. **批内去重**：本批次内已出现的 `hash` 不再写入 |

`hash` 取正文 MD5，与 mem0 一致，便于比对历史数据。

**写入顺序**（保证可恢复）：

```text
1. 逐条或批量嵌入事实文本        （失败即中止，不落任何数据）
2. 批量写向量库                 （失败回退逐条写，仍失败抛 StorageError）
3. 批量写历史表                 （失败回退逐条，仍失败仅告警）
4. 保存原始消息到消息表          （失败仅告警）
```

先写向量库再写历史：向量库是检索的事实来源，历史表是审计。二者不一致时以向量库为准。


### 7.2 `search`：检索

| 步骤 | 方法 | 说明 |
| :--- | :--- | :--- |
| 1 | `validate_query` | 非空、非纯空白，否则抛 `DataError` |
| 2 | `_embed_query` | 查询向量化，失败抛 `EmbeddingError` |
| 3 | `_recall` | 语义召回与关键词召回，**两路候选取并集** |
| 4 | `_fuse_and_rank` | 归一化 BM25 分、按权重融合、阈值过滤、`top_k` 截断 |
| 5 | `_format_results` | payload 拆为核心字段、提升字段、其余 metadata |

**召回策略（本设计对 mem0 的关键修正）**：

```text
semantic_hits = vector.search(query, vec, top_k=recall_k, filters=filters)
keyword_hits  = vector.keyword_search(lemma, top_k=recall_k, filters=filters)
candidates    = union_by_id(semantic_hits, keyword_hits)      # 并集，非交集
其中 recall_k = max(top_k * 4, 60)
```

mem0 只用语义结果做候选集，关键词分数仅用于重排，导致「语义未命中但关键词命中」的记忆被丢弃。本设计把关键词命中也纳入候选，关键词分数为 0 的语义候选同样保留。

**融合打分**：

```text
final = w_sem * norm(semantic_score) + w_bm25 * norm(bm25_score)
默认 w_sem = 1.0，w_bm25 = 0.5（`DEFAULT_KEYWORD_WEIGHT`）；权重下沉为模块常量，便于调优且不与配置字段双份维护
关键词检索不受支持时（后端未实现 keyword_search），w_bm25 自动降为 0 并告警一次
```

**过滤顺序**：先按 `threshold` 剔除低分，再按 `top_k` 截断，最后格式化。避免先截断导致高分项被低分项挤掉。


### 7.3 `update`：更新

| 步骤 | 方法 | 说明 |
| :--- | :--- | :--- |
| 1 | `_load_existing` | 取当前条目；不存在抛 `DataError`（消息含 memory_id）；后端故障原样上抛 |
| 2 | `_merge_payload` | 在旧 payload 上合并新值：`strip_identity_keys` 剥离身份键；重算 `hash` 与 `text_lemmatized`；刷新 `updated_at`；保留 `created_at` |
| 3 | `_reembed_if_needed` | 仅正文变化时重新嵌入，避免无谓调用 |
| 4 | `_persist_update` | 写向量库（失败抛 `StorageError`）→ 写历史（`event=UPDATE`，失败仅告警） |

不变量：

1. `created_at` 永不改写，`updated_at` 每次必刷新。
2. 身份键（三个实体 ID 与 `actor_id`）不可通过 `metadata` 修改；值未变时静默忽略，值变化时告警并忽略。
3. 正文未变且仅改 metadata 时，不调用嵌入服务。

### 7.4 `delete`：删除

| 步骤 | 方法 | 说明 |
| :--- | :--- | :--- |
| 1 | `_load_existing` | 取当前条目；不存在抛 `DataError` |
| 2 | `_persist_delete` | 从向量库物理删除 → 写历史（`event=DELETE`、`is_deleted=1`） |

语义说明：**向量库物理删除，历史表软删**。理由：向量库用于召回，残留会污染检索结果；历史表用于审计，需保留可追溯。

配套：`get_last_messages` 与消息表不受 `delete` 影响，会话上下文仍在。


---

## 8. 关键调用树

### 8.1 `add`

```text
add()
├── validate_entity_id() x3             scope
├── strip_identity_keys()               scope
├── build_scope()                       scope
├── _normalize_messages()               handler
├── [抽取]
│   ├── session_scope()                 scope
│   ├── db.get_last_messages()          历史上下文
│   ├── _embed_query()                  embedding
│   ├── vector.search()                 既有记忆
│   ├── render_extraction_prompt()      prompts
│   ├── llm.chat()                      抽取
│   └── _parse_extraction()             JSON 解析
├── _dedup_hashes()                     md5 去重
├── _persist_facts()
│   ├── embedder.encode()               批量嵌入
│   ├── vector.insert()                 向量库
│   ├── db.batch_add_history()          历史
│   └── db.save_messages()              消息
└── AddResult                          models
```

### 8.2 `search`

```text
search()
├── validate_search_params()            scope
├── validate_query()                    scope
├── build_scope()                       scope
├── _embed_query()                      embedding
├── _recall()
│   ├── vector.search()                 语义召回
│   └── vector.keyword_search()         关键词召回
├── _fuse_and_rank()
│   ├── normalize_scores()              归一化
│   └── filter_then_truncate()          阈值与条数
└── _format_results()                   models
```

### 8.3 `update` 与 `delete`

```text
update()
├── _load_existing()
│   └── vector.get()
├── _merge_payload()
│   ├── strip_identity_keys()           scope
│   └── hash 与 text_lemmatized 重算
├── _reembed_if_needed()
│   └── embedder.encode()               仅正文变化时
└── _persist_update()
    ├── vector.update()
    └── db.add_history(event=UPDATE)

delete()
├── _load_existing()
│   └── vector.get()
└── _persist_delete()
    ├── vector.delete()
    └── db.add_history(event=DELETE, is_deleted=1)
```

---

## 9. 异常与错误处理

沿用 rag-data 既有异常层次，不新增异常类型。

| 场景 | 异常 | 触发点 |
| :--- | :--- | :--- |
| 分区开关冲突、缺少必要组件 | `ConfigError` | 构造期 |
| 消息类型非法、实体 ID 非法、`top_k` 或 `threshold` 越界、记忆不存在 | `DataError` | 流程入口 |
| 嵌入服务失败 | `EmbeddingError` | `_persist_facts`、`_reembed_if_needed` |
| 大模型抽取失败 | `LLMError` | `_extract_facts` |
| 向量库或关系库写入失败 | `StorageError` | `_persist_*` |

**降级边界**（仅告警，不中断）：

| 操作 | 降级行为 |
| :--- | :--- |
| 写历史失败 | 告警并继续，返回结果仍成功 |
| 保存消息失败 | 告警并继续 |
| 关键词检索不支持 | 权重降为 0，仅语义检索 |
| 抽取响应非 JSON | 视为未抽取到事实，保存消息后返回空结果 |


---

## 10. 幂等、并发与性能

### 10.1 幂等

| 场景 | 保障 |
| :--- | :--- |
| 重复 `add` 同一文本 | `hash` 去重（库内与批内两重），已存在则不重复写入 |
| 重复 `delete` | 第二次 `_load_existing` 抛 `DataError`，不产生重复历史 |
| 重复 `update` 相同内容 | 正文未变不重嵌入；`updated_at` 仍刷新，历史多一条 UPDATE |

### 10.2 并发

| 项 | 约定 |
| :--- | :--- |
| 线程安全 | 沿用的 `SQLDatabase` 与 `VectorEngine` 自身持锁；`MemoryHandler` 不自建锁 |
| 单实例并发 | 允许；批内去重仅在单次调用生效，跨调用去重依赖库内 `hash` |
| 跨实例并发 | 依赖存储层唯一性；同一文本可能被两个实例同时写入，属已知取舍 |

### 10.3 性能

| 环节 | 措施 |
| :--- | :--- |
| 嵌入 | 一次批量编码，失败再逐条回退 |
| 写库 | 批量插入；单条失败不阻塞其余条目 |
| 召回 | `recall_k` 与 `top_k` 解耦，避免过度召回 |
| 长文本 | 正文超长时截断后再入 `text_lemmatized`，与存储层字段上限对齐 |

---

## 11. 测试策略

| 层级 | 范围 | 方式 |
| :--- | :--- | :--- |
| 纯函数 | `scope.py`、`prompts.py` | 无需依赖，直接断言归一、剥离、渲染结果 |
| handler 单测 | 四条流程 | 注入伪造的 embedder、llm、db、vector（不连任何服务），断言调用序列与返回值 |
| 契约测试 | 四流程的异常分支 | 逐条覆盖第 9 节表格 |
| 集成测试 | 真实 SQLite 加内存向量桩，或真实向量库 | 标记 `integration`，默认跳过 |

关键断言点：

1. `add` 每次都调用大模型抽取，不存在跳过抽取的路径。
2. `add` 去重命中时不产生新的向量写入。
3. `search` 的候选集包含仅关键词命中的条目（验证与 mem0 的差异）。
4. `search` 先过滤阈值再截断。
5. `update` 仅改 metadata 时不调用嵌入服务。
6. `delete` 后 `vector.get` 返回空，历史表存在 DELETE 记录。


---

## 12. 实施里程碑

| 里程碑 | 内容 | 产出 | 预估 |
| :--- | :--- | :--- | :--- |
| MS1 | 契约与配置 | `base.py`、`models.py`、`MemorySettings`、`config.py` 接入 | 0.5 人天 |
| MS2 | 作用域与提示词 | `scope.py`、`prompts.py` 及其单测 | 1 人天 |
| MS3 | 存储封装 | `stores.py`：id 转换、payload 布局、批量与回退策略 | 1 人天 |
| MS4 | 写入流程 | `handler.add` 与四个私有步骤 | 1.5 人天 |
| MS5 | 检索流程 | `handler.search`、双路召回与融合打分 | 1.5 人天 |
| MS6 | 更新与删除 | `handler.update`、`handler.delete` | 1 人天 |
| MS7 | 门面接入 | `facade._build_memory`、`MemoryServer` 四条流程代理、开关联动校验 | 0.5 人天 |
| MS8 | 文档与测试 | `README`、`.vcl\struct.md`、`.vcl\memory_main_analysis.md` 对照更新 | 1 人天 |

合计约 8 人天。落地顺序严格按 MS1 到 MS8，每步完成后可独立验证。

---

## 13. 待决事项

| # | 事项 | 备选方案 | 倾向 |
| :--- | :--- | :--- | :--- |
| 1 | 语义与关键词的融合权重是否配置化 | 硬编码为常量、放入配置字段 | **已定**：下沉为模块常量 `DEFAULT_KEYWORD_WEIGHT`（配置字段易与代码默认值双份漂移） |
| 2 | ~~`db.enable` 为假时是否允许运行~~ | **已定**：抛 `ConfigError` | 记忆层必须依赖关系库 |
| 3 | 记忆集合与向量库集合的关系 | 共用集合加 `kind` 字段、独立集合 | 独立集合（隔离检索面） |
| 4 | 是否需要 `get(memory_id)` | 本期四流程外、后续补 | 后续补（实现成本极低） |
| 5 | 抽取结果是否需要 `attributed_to` | 保留、丢弃 | 保留（多说话人场景需要） |
| 6 | 是否复用 `storage.db` 的消息表 | 复用、新建表 | 复用（已有 `save_messages` 与 `get_last_messages`） |

