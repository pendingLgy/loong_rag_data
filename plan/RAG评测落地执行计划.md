针对基于 **Milvus 数据库** 与 **EPUB 书籍数据集** 的 RAG 系统，结合 **Ragas 评测框架**，为您梳理一份**端到端的 RAG 评测落地执行计划**。

计划分为 **4 个核心阶段**，覆盖数据准备、自动化生成、端到端测试与评估闭环：

---

```
[阶段 1: 数据提取与格式化] ➔ [阶段 2: Ragas 测试集生成与清洗] ➔ [阶段 3: Milvus 检索与生成执行] ➔ [阶段 4: Ragas 自动化评估与归因]

```

---

## 阶段 1：数据抽取与测试集生成准备 (Day 1)

从 Milvus 中提取具备代表性的 Chunk 数据，转为 Ragas 能够读取的结构。

### 1.1 从数据库抽取样本

* **策略**：按 `title_md5`（按书籍/章节）进行随机或分层抽样，选取 **50~100 个 Chunk** 作为生成基底。
* **数据映射**：将数据库字段映射为 LangChain `Document` 结构：
* `page_content` ⬅️ `text`（原始切片文本）
* `metadata` ⬅️ `{ "id": id, "title": title, "source": source }`



### 1.2 编写数据提取脚本

```python
from langchain_core.documents import Document
from pymilvus import MilvusClient

# 从 Milvus 读取 100 条样本
client = MilvusClient(uri="http://localhost:19530")
res = client.query(
    collection_name="your_epub_collection",
    filter="", 
    limit=100,
    output_fields=["id", "text", "title", "source"]
)

docs = [
    Document(
        page_content=item["text"],
        metadata={"chunk_id": item["id"], "title": item["title"], "source": item["source"]}
    )
    for item in res
]

```

---

## 阶段 2：构建黄金测试集 (Golden Dataset) (Day 1~2)

利用 Ragas 的 `TestsetGenerator` 结合大模型自动合成测试集，并进行人工校验落盘。

### 2.1 自动化合成 QA 数据集

* **题型分布配置**：针对 EPUB 书籍，采用以下比例：
* `simple` (50%)：单 Chunk 事实检索问题
* `reasoning` (25%)：逻辑推理与深度解析问题
* `multi_context` (25%)：跨章节/跨 Chunk 综合提问



```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 初始化合成器
generator = TestsetGenerator.from_langchain(
    ChatOpenAI(model="gpt-4o"),
    ChatOpenAI(model="gpt-4o"),
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# 生成 50 条测试集
testset = generator.generate_with_langchain_docs(
    docs, 
    test_size=50, 
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)

df_golden = testset.to_pandas()

```

### 2.2 人工巡检与清洗（清洗标准）

* **剔除条件**：无主语问题（如“文中提到的那个公式是什么？”）、巨型总结类问题（如“总结整本书”）。
* **格式化导出**：保存为本地文件 `golden_dataset.json`。

---

## 阶段 3： Milvus 检索与生成链路运行 (Day 3)

使用准备好的黄金测试集，在你的 RAG 系统（Milvus + LLM）上跑一遍推理，收集实际检索到的上下文与生成的回答。

### 3.1 编写 Milvus 检索接口

```python
def milvus_retriever(query_text: str, top_k: int = 5):
    # 1. 对 query 生成向量（与数据库相同的 Embedding 模型）
    query_vector = get_embedding(query_text)
    
    # 2. 从 Milvus 检索
    res = client.search(
        collection_name="your_epub_collection",
        data=[query_vector],
        anns_field="vector",
        limit=top_k,
        output_fields=["id", "text"]
    )
    
    # 提取召回的原始文本内容
    retrieved_contexts = [hit["entity"]["text"] for hit in res[0]]
    return retrieved_contexts

```

### 3.2 运行测试集生成数据集

对 `golden_dataset.json` 中的每一个问题：

1. 调用 `milvus_retriever(question)` 获得 `contexts`
2. 将 `question` + `contexts` 喂给 LLM 生成 `answer`
3. 收集为 Ragas 评测所需的统一字典结构：
* `question`: 用户问题
* `contexts`: Milvus **实际检索出**的文本片段列表
* `answer`: LLM **实际生成**的回答
* `ground_truth`: 测试集中的**标准答案**



---

## 阶段 4： Ragas 自动化评估与瓶颈归因 (Day 4)

运行 Ragas 指标计算，根据输出的得分分析 RAG 系统的瓶颈并进行优化。

### 4.1 执行 Ragas 评估

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 转换为 HuggingFace Dataset
eval_dataset = Dataset.from_dict({
    "question": eval_questions,
    "contexts": eval_contexts,
    "answer": eval_responses,
    "ground_truth": eval_ground_truths
})

# 运行评测
results = evaluate(
    dataset=eval_dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
)

print(results.to_pandas())

```

### 4.2 Bad Case 诊断与系统调优策略

| 异常指标 | 问题诊断 | 针对 Milvus / 系统优化的解决方案 |
| --- | --- | --- |
| **`context_recall` 偏低** | **检索漏检**：关键信息未召回 | 1. 增加 Milvus `limit` 召回数量；<br>

<br>2. 结合 `text_lemma` 开启向量 + BM25 **多路混合检索**。 |
| **`context_precision` 偏低** | **噪声过多**：召回了太多不相关切片 | 1. 调整 Milvus 向量相似度阈值（`score_threshold`）；<br>

<br>2. 引入 Reranker（如 BGE-Reranker）对召回结果做二次重排。 |
| **`faithfulness` 偏低** | **幻觉严重**：回答不符合检索内容 | 1. 优化 RAG Prompt，增加“无依据则拒绝回答”约束；<br>

<br>2. 换用逻辑推理能力更强的生成模型。 |
| **`answer_relevancy` 偏低** | **答非所问** | 1. 在检索前增加 Query Rewriting（问题重写/消歧）；<br>

<br>2. 检查 Chunk 切分粒度是否过小破坏了语义连贯性。 |

---

### 💡 交付物清单

1. **`golden_dataset.json`**：包含 50~100 条带 Ground Truth 的高质量黄金测试集。
2. **`eval_pipeline.py`**：可复用的自动化评测 Python 脚本。
3. **`evaluation_report.csv`**：针对当前系统各维度的基准得分（Baseline Score）。

请问您计划在**哪个阶段开始执行**？如果需要具体的代码文件模板或 Prompt 调优建议，我可以为您提供进一步的细节。