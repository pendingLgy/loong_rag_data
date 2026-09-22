# 历史文本数据导入与长期记忆构建架构设计文档

## 1. 架构总览与数据流图

系统整体采用 **离线/批量导入管道（Offline Batch Ingestion Pipeline）** 与 **在线记忆召回引擎（Online Memory Retrieval Engine）** 双路径解耦设计：

```text
                       【离线/批量历史数据导入管道】
[历史文本/文档] ──► 1. 文档解析与结构化 ──► 2. 两阶段语义切块 ──► 3. 实体抽取与链接
(PDF/Markdown/TXT)      (Layout-Aware)       (spaCy 句级切分)      (spaCy NER / Entity Store)
                                                                       │
                                                                       ▼
                                                         4. 向量化与批量写入
                                                            (Milvus / Qdrant)

───────────────────────────────────────────────────────────────────────────────────

                       【在线 Agent 记忆召回引擎】
[Agent 当前 Input] ──► 1. 向量相似度粗筛 ──► 2. 衰减与巩固重排 ──► 3. 上下文注入 & 触发巩固
                         (Top-K 粗筛)         (Rerank Algorithm)     (Async Update: Recall Count+1)
```

---

## 2. 核心模块技术实现规范

### 模块一：历史文本导入与两阶段语义切块 (Batch Chunking)

为了兼顾“大段落上下文完整”与“小句子嵌入精度”，采用**两阶段切块策略**：

1. **一阶段（版面与段落结构切分）**：
   * **输入**：PDF、Word、Markdown 等历史文档。
   * **处理**：使用 `MinerU` 或 `Unstructured` 提取带 Markdown 层级（`# Header`）的文本，自动消除页眉页脚与换行断词。
2. **二阶段（精准语义句级切分 + Overlap）**：
   * **处理**：利用 `spaCy` 的 `doc.sents` 句法模型对段落做精准切句，保证每个 Chunk 字符数在 150~300 字，并包含 1 句重叠（Overlap）。

```python
import spacy

nlp = spacy.load("zh_core_web_sm")
nlp.disable_pipes("ner", "parser")
nlp.add_pipe("sentencizer")


def build_semantic_chunks(
    text: str, max_chars: int = 250, overlap_sents: int = 1
) -> list[str]:
  doc = nlp(text)
  sents = [s.text.strip() for s in doc.sents if s.text.strip()]
  chunks, current_sents, current_len = [], [], 0

  for sent in sents:
    if current_len + len(sent) <= max_chars or not current_sents:
      current_sents.append(sent)
      current_len += len(sent)
    else:
      chunks.append(" ".join(current_sents))
      current_sents = (
          current_sents[-overlap_sents:] if overlap_sents > 0 else []
      )
      current_sents.append(sent)
      current_len = sum(len(s) for s in current_sents)

  if current_sents:
    chunks.append(" ".join(current_sents))
  return chunks
```

---

### 模块二：数据 Schema 设计 (Vector DB / Milvus)

在向量数据库中，每条长期记忆单元需包含**向量索引**与**衰减控制标量**：

| 字段名 (Field) | 类型 (Type) | 索引类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `memory_id` | `VARCHAR / INT64` | Primary Key | 唯一记忆 ID |
| `user_id` | `VARCHAR` | Scalar Index / Partition | 用户/租户隔离标识 |
| `text_payload` | `VARCHAR` | None | 原始记忆文本块 |
| `vector` | `FLOAT_VECTOR` | HNSW / IVFLAT | 文本语义 Embedding (如 1024 维) |
| `entities` | `ARRAY<VARCHAR>` | Scalar Index | 提炼出的实体列表 (用于图谱过滤/Exact Match) |
| `created_at` | `DOUBLE / INT64` | Scalar Index | 历史数据录入时间/创建时间戳 (秒) |
| `last_accessed_at` | `DOUBLE / INT64` | Scalar Index | 上次成功召回并注入上下文的时间戳 |
| `recall_count` | `INT64` | Scalar Index | 累积召回频次 (历史导入初始值为 0) |

---

### 模块三：多维度召回与记忆衰减重排算法 (Rerank Engine)

在线召回时，分两步计算：先通过向量数据库查出 **Top-N (如 Top-30)** 的候选集，再在 Python 内存层做**时间衰减与召回频次巩固重排**。

#### 1. 综合评分算法公式

$$\text{Final Score} = S_{\text{similarity}} \times \text{Decay Factor} \times \text{Reinforcement Factor}$$

* **语义相似度 ($S_{\text{similarity}}$)**：向量余弦相似度，归一化至 $[0, 1]$。
* **时间衰减因子 ($\text{Decay Factor}$)**：基于“上次访问时间/创建时间”计算指数衰减：
  $$\text{Decay Factor} = e^{-\lambda \cdot (t_{\text{now}} - t_{\text{last\_accessed}})}, \quad \lambda = \frac{\ln(2)}{T_{\text{half\_life}} \cdot 86400}$$
  *(取半衰期 $T_{\text{half\_life}} = 14$ 天)*
* **巩固强化因子 ($\text{Reinforcement Factor}$)**：基于召回次数做对数平滑与上限截断 (Cap)：
  $$\text{Reinforcement Factor} = \min\left(1.0 + \beta \cdot \ln(1 + C_{\text{recall}}), \, 1.5\right)$$
  *(取巩固系数 $\beta = 0.2$，最高增益上限为 1.5 倍)*

#### 2. 重排与异步反馈闭环代码实现

```python
from datetime import datetime
import math


class MemoryReranker:

  def __init__(self, half_life_days: float = 14.0, beta: float = 0.2):
    self.lambda_decay = math.log(2) / (half_life_days * 86400)
    self.beta = beta

  def rerank(self, candidates: list[dict], top_k: int = 5) -> list[dict]:
    now = datetime.now().timestamp()

    for item in candidates:
      sim_score = item["score"]  # 向量相似度 (0 ~ 1)
      last_accessed = item.get("last_accessed_at", item.get("created_at", now))
      recall_count = item.get("recall_count", 0)

      # 1. 计算时间衰减
      delta_t = max(0, now - last_accessed)
      decay_factor = math.exp(-self.lambda_decay * delta_t)

      # 2. 计算召回巩固 (对数平滑 + 1.5 倍 Cap 限制)
      reinforcement = min(1.0 + self.beta * math.log(1 + recall_count), 1.5)

      # 3. 综合最终得分
      item["final_score"] = sim_score * decay_factor * reinforcement

    # 按综合得分降序排列并截取 Top-K
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]


# 后台异步更新状态（仅对真正注入 Prompt 的 Top-K 触发）
def async_feedback_loop(selected_memory_ids: list[str], db_client):
  now = datetime.now().timestamp()
  for mem_id in selected_memory_ids:
    db_client.update_scalar(
        id=mem_id,
        updates={"last_accessed_at": now},
        increments={"recall_count": 1},
    )
```