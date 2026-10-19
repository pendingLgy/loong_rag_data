# 手工集成测试：真实解析 EPUB、切块、调用 Qwen 嵌入并写入 Milvus。
#
# 配置唯一来源为 conftest 的 *_JSON 与各分区设置对象：
#
#   EMBEDDING_SETTINGS           api_key、模型、维度与批量上限
#   PARSING_SETTINGS             spaCy 模型与切块长度
#   MILVUS_SETTINGS              Milvus 地址、令牌与数据库名
#   samples.epub                 样例 EPUB 路径，未配置则相关用例跳过
#   cases.qwen_epub.collection   集合名，须以 vector.prefix 开头
#   vector.keep 与 vector.prefix 是否保留集合、允许清理的前缀
#
# 安全约定：集合名以 vector.prefix 开头且 vector.keep 未置真时才删除集合。

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Union
from openai import OpenAI

import pytest

from conftest import CASES_CONFIG, SAMPLES_CONFIG, VECTOR_CONFIG, EMBEDDING_SETTINGS, MILVUS_SETTINGS, PARSING_SETTINGS

from rag_data import RagData, RagSetting

# 凭据与连接信息一律来自 conftest 的 *_JSON 与各分区设置对象。
MILVUS_URI = MILVUS_SETTINGS.uri
MILVUS_TOKEN = MILVUS_SETTINGS.token
MILVUS_DB = MILVUS_SETTINGS.db_name
COLLECTION = CASES_CONFIG["qwen_epub"]["collection"]
MODEL = EMBEDDING_SETTINGS.model
API_KEY = EMBEDDING_SETTINGS.api_key
SAMPLE_CHARS = SAMPLES_CONFIG["sample_chars"]
KEEP_COLLECTION = VECTOR_CONFIG["keep"]

MAX_CHARS = PARSING_SETTINGS.max_chars
SAFE_MAX_CHARS = PARSING_SETTINGS.safe_max_chars
SPACY_MODEL = PARSING_SETTINGS.spacy_model

VECTOR_DIM = EMBEDDING_SETTINGS.dim
QWEN_MAX_BATCH = EMBEDDING_SETTINGS.batch_size
COLLECTION_PREFIX = VECTOR_CONFIG["prefix"]
ALIAS = 'autotest'


def _local_epub() -> str:
    # 样例路径来自 conftest 的 samples.epub，未配置时相关用例跳过。
    return SAMPLES_CONFIG["epub"]

EPUB_PATH = _local_epub()

def _requires_api_key():
    if not API_KEY:
        pytest.skip("未配置 API Key：请在 conftest 的 embedding.api_key 中填入")

def _requires_pymilvus():
    if importlib.util.find_spec('pymilvus') is None:
        pytest.skip("未安装 pymilvus，请先执行 pip install pymilvus")

def _requires_epub():
    if not EPUB_PATH or not os.path.exists(EPUB_PATH):
        pytest.skip("未配置样例：请在 conftest 的 samples.epub 中填入文件路径")


# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if "--integration" not in sys.argv:
    # 你可以直接在代码里写原因，或者写死一个统一的提示
    skip_reason = f"[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）"

    pytest.skip(skip_reason, allow_module_level=True)

# ---------------- 夹具 ----------------


@pytest.fixture(scope='module')
def it_settings() -> RagSetting:
    # 按集成测试需要构造配置：qwen 嵌入；凭据与模型来自环境变量。
    return RagSetting.load_mapping(
        parsing={
            'enable': True,
            'spacy_model': SPACY_MODEL,
            'max_chars': MAX_CHARS,
            'safe_max_chars': SAFE_MAX_CHARS,
        },
        embedding={
            'enable': True,
            'provider': 'qwen',
            'model': MODEL,
            'api_key': API_KEY,
            'dim': VECTOR_DIM,
            'batch_size': QWEN_MAX_BATCH,
        }
    )


@pytest.fixture(scope='module')
def app(it_settings) -> RagData:
    return RagData(it_settings)

@pytest.fixture(scope='module')
def pymilvus():
    _requires_pymilvus()

    # 惰性导入，未安装时由 pytestmark 先行跳过。
    import pymilvus as module

    return module


@pytest.fixture(scope='module')
def collection(pymilvus):
    # 1. 初始化客户端连接
    client_kwargs = {'uri': MILVUS_URI}
    if MILVUS_TOKEN:
        client_kwargs['token'] = MILVUS_TOKEN
    if MILVUS_DB:
        client_kwargs['db_name'] = MILVUS_DB

    client = pymilvus.MilvusClient(**client_kwargs)

    # 2. 安全清理逻辑：如果集合已存在，先强制删除
    # 安全防护：只有当 COLLECTION 名称以测试前缀开头，且不要求强行保留时才清理
    is_safe_to_drop = COLLECTION.startswith(COLLECTION_PREFIX)
    if not KEEP_COLLECTION and is_safe_to_drop:
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)

    # 双重保障：如果由于上次异常中断导致集合残留，这里如果匹配前缀也直接清理掉，避免索引冲突
    if client.has_collection(COLLECTION) and is_safe_to_drop:
        client.drop_collection(COLLECTION)

    # 3. 定义 Schema 并创建集合
    schema = client.create_schema(
        auto_id=False,
        description='rag-data epub 集成测试集合',
    )
    schema.add_field(field_name='id', datatype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name='source', datatype=pymilvus.DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='text', datatype=pymilvus.DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name='vector', datatype=pymilvus.DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    # 4. 配置索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name='vector',
        index_type='HNSW',
        metric_type='COSINE',
        params={'M': 16, 'efConstruction': 200}
    )

    # 5. 创建集合并应用 Schema 和 Index
    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=index_params
    )

    # 确保加载到内存
    client.load_collection(COLLECTION)

    yield client

    # 6. 测试模块退出后的清理
    if not KEEP_COLLECTION and is_safe_to_drop:
        if client.has_collection(COLLECTION):
            client.drop_collection(COLLECTION)
    client.close()

# ---------------- 嵌入 ----------------


def test_provider_is_qwen(app):
    # 确认装配的确实是配置指定的 qwen provider。
    assert app.settings.embedding.provider == 'qwen'
    assert type(app.embedder.provider).__name__ == 'QwenEmbedder'
    assert app.embedder.provider.model == MODEL
    assert app.embedder.provider.dim == VECTOR_DIM


# ---------------- 解析与切块 ----------------

def test_chunk_returns_non_empty_blocks(app:RagData):
    _requires_epub()

    text = app.parse_document(EPUB_PATH)
    assert text
    chunks = app.chunk_document(EPUB_PATH, text[:SAMPLE_CHARS])
    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert app.settings.parsing.max_chars == MAX_CHARS
    print('chunks', len(chunks), 'max_chars', MAX_CHARS, 'sizes', [len(chunk) for chunk in chunks])

# ---------------- 向量化 ----------------

def test_vectorize_chunks_returns_vectors(app:RagData):
    _requires_api_key()

    result = app.embedder.encode(['向量数据库', 'Milvus 相似度检索'])
    assert len(result) == 2
    assert all(len(vector) == VECTOR_DIM for vector in result)



# ---------------- 写入 Milvus ----------------
def test_insert_epub_chunks(app:RagData, collection):
    _requires_api_key()
    _requires_epub()


    # 1. 解析与切块
    text = app.parse_document(EPUB_PATH)
    chunks = app.chunk_document(EPUB_PATH, text)
    assert len(chunks) > 0

    # 2. 处理并插入 Milvus
    batch_chunks = [chunks[0]]
    batch_vectors = app.embedder.encode(batch_chunks)

    # 构建当前批次数据行
    batch_rows = []
    for idx_in_batch, (chunk_text, vector) in enumerate(zip(batch_chunks, batch_vectors)):
        batch_rows.append({
            'id': f'epub-{idx_in_batch}',
            'source': EPUB_PATH,
            'text': chunk_text,
            'vector': vector,
        })

    # 批次写入 Milvus
    collection.insert(collection_name=COLLECTION,data=batch_rows)

    # 3. 所有批次插入完成后统一下盘刷新数据，提高性能
    collection.flush(collection_name=COLLECTION)


def test_milvus_insert_and_search(collection):
    """测试 Milvus 集合的数据插入与向量相似度检索"""
    client = collection  # 对应 fixture 传进来的 MilvusClient 实例

    # 确保向量维度与建表时一致（假设 VECTOR_DIM 已在全局定义，例如 1536 或 768）
    # 这里用 [0.1] * VECTOR_DIM 模拟一个测试向量
    mock_vector_1 = [0.1] * VECTOR_DIM
    mock_vector_2 = [0.9] * VECTOR_DIM

    # 1. 准备要插入的数据（支持批量字典列表）
    rows = [
        {
            "id": "doc_001",
            "source": "epub_chapter_1.epub",
            "text": "这是一段关于人工智能与大语言模型长期记忆的测试文本。",
            "vector": mock_vector_1
        },
        {
            "id": "doc_002",
            "source": "epub_chapter_2.epub",
            "text": "这是另一段完全无关的测试内容，用于验证向量检索准确性。",
            "vector": mock_vector_2
        }
    ]

    # 2. 执行插入操作
    insert_result = client.insert(
        collection_name=COLLECTION,
        data=rows
    )
    print(f"插入结果: {insert_result}")

    # 验证插入成功（Milvus 插入后通常会返回成功的主键列表和插入条数）
    assert insert_result["insert_count"] == 2

    # 3. 刷新数据（可选：Milvus 是准实时写入，显式 flush 可以确保立即可查）
    client.flush(collection_name=COLLECTION)

    # 4. 执行向量检索 (Search)
    # 构造一个与 doc_001 相似的查询向量
    query_vector = [0.11] * VECTOR_DIM

    search_results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],  # 支持批量查询，这里传单个向量的二维列表
        anns_field="vector",  # 指定做向量检索的字段
        search_params={
            "metric_type": "COSINE",  # 必须与建表时指定的 metric_type 一致
            "params": {"ef": 32}  # HNSW 检索参数
        },
        limit=1,  # 返回最相似的 Top 1
        output_fields=["id", "text", "source"]  # 希望同时返回的标量字段
    )

    # 5. 断言验证检索结果
    print(f"检索结果: {search_results}")

    # 验证返回了结果集
    assert len(search_results) > 0
    top_match = search_results[0][0]  # 取第一个 Query 的第一条命中结果

    # 因为 query_vector 更接近 doc_001 的向量，所以 Top 1 应该是 doc_001
    assert top_match["id"] == "doc_001"
    assert top_match["entity"]["source"] == "epub_chapter_1.epub"
    print(f"命中内容: {top_match['entity']['text']} (相似度分值: {top_match['distance']})")


def fetch_neighbor_contexts(
    collection: Any,
    search_results: Union[List[Any], Any],
    window_size: int = 1,
) -> str:
    """适配现有 Schema [id, source, text, vector] 的邻居窗口扩展函数。
    支持 search_results 数组（单 Query 或多 Query 检索结果）。
    要求 id 格式为 '{doc_prefix}_{chunk_index}'，例如 'a1b2c3_0', 'a1b2c3_1'。
    """

    # 1. 扁平化展开 search_results 数组 (支持多 Query/Hits 嵌套)
    all_hits = []
    if isinstance(search_results, (list, tuple)):
        for item in search_results:
            if hasattr(item, "__iter__") and not isinstance(item, (dict, str)):
                all_hits.extend(item)
            else:
                all_hits.append(item)
    else:
        all_hits = list(search_results)

    # 2. 从 id 字段解析 doc_prefix 与 chunk_index
    doc_ranges = {}  # 格式: {doc_prefix: set(chunk_indices)}

    for hit in all_hits:
        entity = getattr(hit, "entity", hit)

        # 兼容 PyMilvus Hit 对象和 dict 对象
        if isinstance(entity, dict):
            hit_id = entity.get("id")
        else:
            hit_id = getattr(entity, "id", None) or entity.get("id")

        if not hit_id or "-" not in str(hit_id):
            continue

        # 从右侧切割一次：左边是完整的文档唯一标识(doc_prefix)，右边是切片序号(chunk_index)
        parts = str(hit_id).rsplit("-", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue  # 抛弃后缀不是纯数字的非标准 id

        doc_prefix, idx_str = parts[0], parts[1]
        curr_idx = int(idx_str)

        start_idx = max(0, curr_idx - window_size)
        end_idx = curr_idx + window_size

        # 核心：按完整的 doc_prefix 进行分组（自动按文档聚合）
        if hit_id not in doc_ranges:
            doc_ranges[hit_id] = set()

        doc_ranges[hit_id].update(range(start_idx, end_idx + 1))

    # 若无法根据 id 解析出任何邻居范围，退回直接提取当前命中的原始文本
    if not doc_ranges:
        raw_texts = []
        for hit in all_hits:
            entity = getattr(hit, "entity", hit)
            t = entity.get("text") if isinstance(entity, dict) else entity.get("text")
            if t:
                raw_texts.append(t)
        return "\n\n---\n\n".join(raw_texts)

    merged_chunks: List[str] = []

    # 3. 按文档前缀构造 id 列表批量查询相邻块
    for hit_id, indices_set in doc_ranges.items():
        sorted_indices = sorted(list(indices_set))
        target_ids = [f"{str(hit_id).rsplit("-",1)[0]}-{i}" for i in sorted_indices]

        # 从 Milvus 根据主键 id 批量查询
        expr = f'id in {target_ids}'
        records = collection.query(
            expr=expr,
            output_fields=["id", "text"],
        )

        # 根据 id 中的物理序号升序排列，保证还原上下文连续性
        def parse_idx(item: dict) -> int:
            return int(item["id"].rsplit("-", 1)[1])

        sorted_records = sorted(records, key=parse_idx)
        doc_text = "\n\n".join([item["text"] for item in sorted_records])
        merged_chunks.append(doc_text)

    return "\n\n---\n\n".join(merged_chunks)

@pytest.mark.skip(reason="功能尚未开发完成")
def test_rag_call(app: RagData, collection):
    _requires_api_key()

    # 1. 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://llm-aa9vce2460xvbj3o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"  # 如有代理或兼容接口可在此配置
    )

    question = "帐上帐下，但有犯者，即鞭挞之，多有鞭死者。给我补充它的上下内容"

    # 2. 从 Milvus / Collection 中检索真实上下文（或传入测试 context）
    query_vector = app.embedder.encode([question])

    # 使用之前实现好的带窗口扩展/检索逻辑获取上下文
    # 示例中使用 search 拉取 top-K，或者使用你的 pipeline 检索
    search_results = collection.search(
        data=query_vector,
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=3,
        output_fields=["text"],
    )

    # 3. 核心修改：向前后扩展 1 个邻居窗口 (拉取 [idx-1, idx, idx+1])
    if search_results and search_results[0]:
        context = fetch_neighbor_contexts(
            collection=collection,
            search_results=search_results,
            window_size=1,  # 向前向后各扩 1 个块
        )
    else:
        context = "暂无数据"

    # 4. 构造 OpenAI 原生 Message 消息列表
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的 RAG 系统问答助手。请根据提供的上下文回答问题。",
        },
        {
            "role": "user",
            "content": f"上下文：\n{context}\n\n问题：{question}",
        },
    ]

    # 4. 调用 OpenAI Chat Completions 接口
    response = client.chat.completions.create(
        model="qwen3.7-max",
        messages=messages,
        temperature=0,
        timeout=10
    )

    answer = response.choices[0].message.content
    print("OpenAI 输出结果：\n", answer)

    # 5. 断言检查
    assert answer is not None
    assert len(answer) > 0

