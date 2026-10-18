# memory 主模块（main.py）代码分析

> 分析对象：mem0.memory.main（3868 行，171 KB）
> 分析方式：AST 结构提取（类与方法边界、装饰器、同步与异步方法差集、源码相似度比对）＋ 全文分段通读
> 说明：全部结论来自静态阅读，**未执行任何测试**。

---

## 1. 总结：代码作用

`mem0.memory.main` 是 mem0（长时间记忆层）的**核心业务实现**，对外提供两套等价 API：

| 类 | 行范围 | 行数 | 职责 |
| :--- | :--- | :--- | :--- |
| `Memory` | L487-2169 | 1683 | 同步门面，实现 `MemoryBase` 抽象接口 |
| `AsyncMemory` | L2172-3868 | 1697 | 异步门面，用 `asyncio.to_thread`（67 处）把同步 I/O 卸载到线程池 |

核心能力（围绕「LLM 驱动的记忆增删改查」）：

1. **写入 `add()`**：消息归一化 → LLM 抽取事实 → 哈希去重 → 批量嵌入 → 批量落库 → 批量实体关联 → 落历史 → 遥测。
2. **检索 `search()`**：语义检索（稠密向量）＋ **BM25 关键词重排** ＋ **实体图谱加权**，三路融合后按 `threshold` 过滤、`top_k` 截断，可选 rerank。
3. **实体图谱**：独立向量集合 `<collection>_entities`（懒加载），记录「实体 → 关联 memory_id」，用于检索时的实体加权与写入时的精确与语义去重。
4. **多租户作用域**：`user_id`、`agent_id`、`run_id` 三元组构成 session scope，`_strip_identity_keys` 阻止调用方通过 metadata 越权改作用域。
5. **横切能力**：SQLite 历史审计（ADD、UPDATE、DELETE）、过期记忆过滤（`expiration_date`）、遥测、首次运行与规模阈值与慢查询与时序与衰减等 UX 提示。

---

## 2. 全局变量（模块级常量与配置）

| 名称 | 类型 | 作用 |
| :--- | :--- | :--- |
| `logger` | `logging.Logger` | **定义两次**（L82 供工具函数、L456 在 `setup_config()` 后重建），冗余但无害 |
| `_RUNTIME_FIELDS` | `frozenset` | 运行时对象字段白名单（`http_auth`、`client` 等），反序列化时**必须保留**，优先级最高 |
| `_SENSITIVE_FIELDS_EXACT` | `frozenset`（18 项） | 敏感字段精确黑名单（`api_key`、`password`、`_token` 类） |
| `_SENSITIVE_SUFFIXES` | `tuple` | 敏感后缀黑名单（`_password`、`_secret`、`_token` 等），兜住 `db_password` 这类命名 |

| `ENTITY_PARAMS` | `frozenset` | `{user_id, agent_id, run_id}`，实体作用域参数 |
| `_IDENTITY_KEYS` | `frozenset` | `ENTITY_PARAMS` 并上 `actor_id`，**不可经 metadata 设置**的身份键 |
| `DELETE_ALL_BATCH_SIZE` | `int = 1000` | `delete_all` 单批删除量 |
| `_UNSET` | `object()` | 哨兵值，区分 `expiration_date=None`（清空）与「未传」 |
| `_PROJECT_UPDATE_UNSUPPORTED_ERROR` | `str` | OSS 版 `project.update` 统一不可用提示 |
| `setup_config()` | 导入期副作用 | 模块 import 即执行（建目录、写配置），属**隐式副作用** |

---

## 3. Public 方法

### 3.1 `Memory`（同步）

| 方法 | 签名要点 | 返回值 | 职责 |
| :--- | :--- | :--- | :--- |
| `__init__` | `config: MemoryConfig = MemoryConfig()` | - | 装配 embedder、vector_store、llm、SQLite、reranker；构建遥测影子库；检测关键词检索支持 |
| `from_config` | `@classmethod (config_dict)` | `Memory` | 字典转 `MemoryConfig`（`ValidationError` 记日志后重抛） |
| `project` | `@property` | `_OSSProject` | 返回不支持 `update` 的占位对象 |
| `entity_store` | `@property` | `VectorStoreBase` | **懒初始化**实体集合；Qdrant 共享 client 规避 RocksDB 锁竞争 |
| `add` | `(messages, *, user_id, agent_id, run_id, metadata, timestamp, expiration_date, infer=True, memory_type, prompt)` | `{results: [...]}` | 记忆写入总入口；`infer=False` 走原始写入，`memory_type=procedural_memory` 走流程记忆分支 |

| `get` | `(memory_id)` | `dict` 或 `None` | 单条读取；payload 拆成「核心字段 + 提升字段 + 其余 metadata」 |
| `get_all` | `(*, filters, top_k=20, show_expired=False, **kwargs)` | `{results: [...]}` | 列表查询；未展开过期时**超额拉取** `max(top_k*4, 60)` |
| `search` | `(query, *, top_k=20, filters, threshold=0.1, rerank=False, explain=False, reference_date, show_expired, **kwargs)` | `{results: [...]}` | 混合检索主入口（详见第 6 节） |
| `update` | `(memory_id, text=None, metadata=None, expiration_date=_UNSET, data=None)` | `{message: ...}` | 更新；`data` 为已弃用别名；身份键不可改 |
| `delete` | `(memory_id)` | `{message: ...}` | 单条删除（含历史审计与实体清理） |
| `delete_all` | `(user_id, agent_id, run_id)` | `{message: ...}` | 按作用域**分页循环**删除（防 `list()` 默认 100 条截断） |
| `history` | `(memory_id)` | `list` | 从 SQLite 取变更历史 |
| `reset` | `()` | - | 清库、重置 vector store、清实体库 |
| `close` | `()` | - | 释放 SQLite 连接（置 `None`，可重复调用） |
| `chat` | `(query)` | - | **占位**，抛 `NotImplementedError` |

### 3.2 `AsyncMemory`

公共方法与上表**一一对应**（`add`、`get`、`get_all`、`search`、`update`、`delete`、`delete_all`、`history`、`reset`、`close`、`chat` 均为 `async`），语义差异见第 6 节与第 7 节风险清单。

---

## 4. Private 方法

### 4.1 模块级纯函数

| 函数 | 职责 |
| :--- | :--- |
| `_vector_store_list_rows` | 归一化 `list()` 的返回形状（部分后端返回嵌套列表） |
| `_strip_identity_keys` | 丢弃 metadata 中的身份键；更新路径「值未变则静默、变了才告警」 |
| `_reject_top_level_entity_params` | `search`、`get_all` 拒绝顶层 `user_id` 等，强制走 `filters` |
| `_validate_and_trim_entity_id` | 实体 ID 归一：非 str 强转、去空白、拒空、**拒内部空白** |
| `_validate_search_params` | `threshold` 属于 `[0,1]`、`top_k` 非负整数（显式排除 `bool`） |
| `_validate_and_trim_search_query` | 查询非空、非纯空白 |
| `_is_sensitive_field` | 三级判定：运行时白名单、精确黑名单、后缀黑名单 |
| `_safe_deepcopy_config` | `deepcopy` 失败时降级为 dict 克隆，并恢复运行时字段、脱敏敏感字段 |
| `_normalize_iso_timestamp_to_utc` | 仅把**带时区**的 ISO 字符串转 UTC，naive 原样返回 |
| `_build_filters_and_metadata` | 核心作用域构造器：返回「存储用 metadata 模板」与「查询用 filters」，并强制至少一个实体 ID |
| `_escape_scope_value`、`_build_session_scope` | 生成确定性 session scope 字符串（`%`、`&`、`=` 转义） |
| `_entity_collection_name` | 实体集合命名（`s3_vectors` 用 `-`，其余用 `_`） |
| `_normalize_expiration_date`、`_payload_is_expired` | 过期日期归一与判定（无效值按未过期处理） |


### 4.2 `Memory` 实例私有方法

| 方法 | 行数 | 职责 |
| :--- | :--- | :--- |
| `_add_to_vector_store` | **328** | 写入主流水线（`infer=False` 逐条原始写入 + 8 个阶段：上下文、检索既有、LLM 抽取、批量嵌入、哈希去重、批量落库、批量历史、批量实体关联） |
| `_create_memory` | 31 | 单条落库与历史（供原始写入与流程记忆复用） |
| `_create_procedural_memory` | 44 | 流程记忆：LLM 摘要、写入、历史 |
| `_update_memory` | 61 | 读旧值、合并 metadata（剥离身份键）、重嵌入、update、历史、文本变化时**重建实体链接** |
| `_delete_memory` | 29 | 删除、历史（`is_deleted=1`）、实体清理 |
| `_search_vector_store` | 104 | 混合检索实现（第 6 节） |
| `_compute_entity_boosts` | 81 | 查询实体、批量嵌入、`ThreadPoolExecutor(4)` 并发查实体库、加权 |
| `_process_metadata_filters` | 76 | 高级操作符转到后端可识别格式 |
| `_has_advanced_operators` | 26 | 检测 `AND`、`OR`、`NOT`、比较操作符、通配符 |
| `_existing_entities_by_text` | 18 | 按归一化文本精确索引实体（失败降级为空，退化为语义去重） |
| `_upsert_entity` | 46 | 单实体 upsert：精确匹配与语义匹配（`score >= 0.95`） |
| `_remove_memory_from_entity_store` | 54 | 从实体 `linked_memory_ids` 移除该 memory；空列表则删实体，否则**重嵌入**后 update |
| `_link_entities_for_memory` | 22 | 新文本抽取实体并逐个 `_upsert_entity` |


`AsyncMemory` 对应私有方法：`_add_to_vector_store`(332)、`_create_memory`(36)、`_create_procedural_memory`(61)、`_update_memory`(63)、`_delete_memory`(30)、`_search_vector_store`(102)、`_compute_entity_boosts_async`(73)、`_process_metadata_filters`(76)、`_has_advanced_operators`(26)、`_existing_entities_by_text`(18)、`_upsert_entity_async`(49)、`_bulk_clear_entity_store`(20)、`_remove_memory_from_entity_store`(44)、`_link_entities_for_memory`(18)。

**仅存在于一侧的方法**：

- 同步独有：`_upsert_entity`、`_compute_entity_boosts`
- 异步独有：`_upsert_entity_async`、`_compute_entity_boosts_async`、`_bulk_clear_entity_store`

---

## 5. 静态方法

全文件仅 **8 个装饰成员**，其中静态方法 2 个（两类各一）：

| 方法 | 装饰器 | 职责 |
| :--- | :--- | :--- |
| `Memory._normalize_entity_text` | `@staticmethod` | `" ".join(value.strip().lower().split())`，实体文本归一（去首尾空白、折叠内部空白、小写） |
| `AsyncMemory._normalize_entity_text` | `@staticmethod` | 同上（**完全重复实现**） |

其余装饰成员：`project`（`@property` 两处）、`entity_store`（`@property` 两处，懒加载）、`from_config`（`@classmethod` 两处）。


---

## 6. 方法嵌套调用树

### 6.1 写入 `add()`

```text
add() -- 记忆写入总入口
│-- _normalize_expiration_date()        -- 过期日期归一
│-- detect_temporal_usage_from_metadata()  -- 时序列用法检测（提示用）
│-- _build_filters_and_metadata()       -- 作用域构造
│   │-- _strip_identity_keys()          -- 剥离 metadata 身份键
│   │-- _validate_and_trim_entity_id()  -- 三重校验三元组
│-- [procedural 分支] _create_procedural_memory()
│   │-- self.llm.generate_response()
│   │-- remove_code_blocks()
│   │-- self.embedding_model.embed()
│   │-- _create_memory()
│       │-- self.embedding_model.embed()
│       │-- lemmatize_for_bm25()
│       │-- self.vector_store.insert()
│       │-- self.db.add_history()
│-- parse_vision_messages()             -- 多模态消息处理
│-- _add_to_vector_store()  ** 主流水线（328 行）
│   │-- [infer=False] 逐条：embed() 然后 _create_memory()
│   │-- [infer=True] 8 阶段：
│       │-- P0 _build_session_scope() 然后 db.get_last_messages() 与 parse_messages()
│       │-- P1 embed() 然后 vector_store.search()
│       │-- P2 generate_additive_extraction_prompt() 然后 llm.generate_response()
│       │      │-- remove_code_blocks() 与 extract_json()
│       │-- P3 embed_batch()   [降级：逐条 embed()]
│       │-- P4-5 lemmatize_for_bm25() 与 md5 哈希去重
│       │-- P6 vector_store.insert()    [降级：逐条 insert()]
│       │      db.batch_add_history()   [降级：逐条 add_history()]
│       │-- P7 extract_entities_batch() 然后 embed_batch()
│       │      │-- _normalize_entity_text()      (Static)
│       │      │-- _existing_entities_by_text()
│       │      │-- entity_store.search_batch()
│       │          │-- entity_store.update() 或 entity_store.insert()
│       │-- P8 db.save_messages() 然后 process_telemetry_filters() 然后 capture_event()
│-- detect_scale_threshold_from_add_result() 然后 display_notice()
```

### 6.2 检索 `search()`

```text
search() -- 混合检索入口
│-- _reject_top_level_entity_params()   -- 拒绝顶层实体参数
│-- _validate_search_params()           -- threshold 与 top_k
│-- _validate_and_trim_search_query()
│-- _validate_and_trim_entity_id()      -- 校验 filters 内实体 ID 三处
│-- _has_advanced_operators()
│   │-- _process_metadata_filters()     -- 高级操作符转后端格式
│-- process_telemetry_filters() 然后 capture_event()
│-- _search_vector_store()  ** 融合检索
│   │-- lemmatize_for_bm25()
│   │-- extract_entities()
│   │-- embedding_model.embed()
│   │-- vector_store.search()           -- 语义召回 internal_limit = max(top_k*4, 60)
│   │-- vector_store.keyword_search()   -- 关键词召回
│   │-- get_bm25_params() 与 normalize_bm25()
│   │-- _compute_entity_boosts()
│   │   │-- _normalize_entity_text()    (Static)
│   │   │-- embed_batch()
│   │   │-- entity_store.search()       -- 经 ThreadPoolExecutor(4) 并发
│   │   │-- ENTITY_BOOST_WEIGHT 与数量衰减加权
│   │-- score_and_rank()                -- 三路融合、threshold、top_k
│-- [rerank=True] self.reranker.rerank()
│-- display_notice()                    -- 首次运行、规模、慢查询提示
```

### 6.3 更新与删除

```text
update()
│-- _normalize_expiration_date()
│-- embedding_model.embed()
│-- _update_memory()
│    │-- vector_store.get()
│    │-- _strip_identity_keys()
│    │-- lemmatize_for_bm25() 与 embedding_model.embed()
│    │-- vector_store.update()
│    │-- db.add_history()
│    │-- [文本变化] _remove_memory_from_entity_store()  -- 列表、重嵌入、update 或 delete
│    │   _link_entities_for_memory()
│                  │-- extract_entities() 然后 _upsert_entity()
│                      │-- _existing_entities_by_text()
│                      │-- entity_store.search() 然后 update 或 insert

delete() 然后 _delete_memory()
│-- vector_store.get()
│-- vector_store.delete()
│-- db.add_history(is_deleted=1)
│-- _remove_memory_from_entity_store()
```

---

## 7. 架构观察与风险清单

### 7.1 高

| # | 问题 | 位置 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | **可变默认参数** | `__init__(self, config: MemoryConfig = MemoryConfig())` | pydantic 实例在**导入期创建并由所有实例共享**。任一调用方改 `memory.config.*` 会污染后续 `Memory()` 的默认配置 |
| 2 | **`reset()` 同步与异步行为不一致** | L2131 vs L3828 | 同步用 `VectorStoreFactory.reset`，**不关闭旧 client、不 `gc.collect`**；异步先 `delete_col`、`gc.collect()`、`client.close()`、重建。同步路径可能残留连接（Milvus、Qdrant 长连接场景易泄漏） |
| 3 | **关键词召回不参与候选集** | L1667-1688 | 候选集**只来自语义召回**，`bm25_scores` 仅用于对已召回候选重排。**纯关键词命中（语义未召回）的记忆会被丢弃**，属混合检索的召回缺陷，而非单纯排序问题 |

### 7.2 中

| # | 问题 | 位置 | 说明 |
| :--- | :--- | :--- | :--- |
| 4 | **`uuid_mapping` 死代码** | L935-938、L2597-2600 | 建立「索引到 UUID」映射后**从未使用**；注释称 anti-hallucination 但链路未闭合 |
| 5 | **`telemetry_config` 赋值被覆盖** | L533 与 L540 | 第 533 行 `_safe_deepcopy_config(...)` 的结果在 540 行被完全覆盖；同步路径还保留 519-527 的手工 `hasattr` 提取逻辑，而异步版已用一行替代，属遗留实现未清理 |
| 6 | **`hasattr(self.vector_store, reset)` 恒真** | L2144 | `reset` 是 `VectorStoreBase` 抽象方法，`hasattr` 永远为 `True`，`else` 分支（`delete_col` 加重建）**不可达** |
| 7 | **错误类型不统一** | 多处 | 同一语义两种异常：`add` 的 `memory_type` 非法时同步抛 `Mem0ValidationError`、异步抛 `ValueError`；`_validate_and_trim_entity_id` 抛 `ValueError`，`_build_filters_and_metadata` 抛 `Mem0ValidationError`。上层难以按类型分流 |
| 8 | **大方法难以单测** | `_add_to_vector_store` 328 与 332 行 | 单方法内混合 8 个阶段、3 类降级策略、2 种写入模式，无法独立验证单个阶段 |
| 9 | **实体库全量扫描** | `top_k=10000`（L590、L670），`_compute_entity_boosts` 中 `top_k=500` | 数据量增长后，`add`、`update`、`delete` 的实体维护成本随时间线性上升 |
| 10 | **每次检索新建线程池** | L1779 | `_compute_entity_boosts` 每次调用都 `ThreadPoolExecutor(max_workers=4)`（上下文管理器即时销毁），高并发下线程创建销毁开销显著 |

### 7.3 低（设计取舍或小瑕疵）

| # | 观察 | 说明 |
| :--- | :--- | :--- |
| 11 | **同步与异步大幅重复** | 19 个同名方法，平均源码相似度 **0.815**；`_normalize_entity_text`、`_has_advanced_operators`、`_process_metadata_filters` 近乎逐字复制。双份维护易漂移（风险 2 与 7 即漂移产物） |
| 12 | **导入期副作用** | L455 `setup_config()` 在建目录、写配置，属隐式行为，测试与只读部署环境需注意 |
| 13 | **`logger` 定义两次** | L82、L456，后者覆盖前者 |
| 14 | **`get_all` 可能少于 `top_k`** | `show_expired=False` 时先取 `max(top_k*4, 60)` 再过滤过期，若过期比例超过 75 会返回不足，属已知近似策略 |
| 15 | **降级路径静默** | 实体库相关大量 `except Exception` 仅打日志（`_remove_memory_from_entity_store`、`_link_entities_for_memory`、Phase 7 整体包裹），实体一致性失败对调用方不可见 |
| 16 | **`chat()` 两侧均未实现** | `MemoryBase` 未声明该方法，仅 `NotImplementedError` 占位 |

### 7.4 值得肯定的设计

- **安全边界清晰**：`_IDENTITY_KEYS` 在写入与更新两条路径都做了剥离，配合 `_reject_top_level_entity_params` 堵住「metadata 越权改作用域」（对应 issue 4490、6277、6655）。
- **降级策略完备**：批量嵌入、批量插入、批量历史、实体批量关联均有逐条回退，可用性优先。
- **LLM 失败不再静默**：`_add_to_vector_store` 明确把抽取异常转为 `LLMError` 重抛（注释说明此前 `return []` 会让上游无法区分限流、超时与真实空结果）。
- **防截断删除**：`delete_all` 分页循环加 `seen_batches` 去重，规避后端 `list()` 默认 100 条上限导致删除不彻底。
- **可观测性分层**：遥测（`capture_event`）与 UX 提示（`display_notice`）分离，提示失败不阻断主流程。

---

## 8. 说明

- 分析方式：AST 结构提取（类与方法边界、装饰器、同步与异步方法差集、源码相似度比对）加全文分段通读。
- **未执行任何测试**，结论均来自静态阅读；风险 2、3、12 建议人工用真实后端（Milvus、Qdrant）验证连接残留与纯关键词召回场景。
- 可选的后续工作：**(a)** 同步与异步去重重构（提取共享流水线加适配器）；**(b)** 风险 3 的最小改动方案（关键词候选并入候选集）；**(c)** 拆分 `_add_to_vector_store` 为分阶段函数并补单测骨架。

