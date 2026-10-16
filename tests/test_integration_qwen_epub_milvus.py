# 手工集成测试：真实解析 EPUB、切块、调用 Qwen 嵌入并写入 Milvus。
#
# 环境变量（凭据与连接信息在运行时单独提供，代码与仓库中不落任何密钥）：
#
# | 变量 | 必填 | 默认值 | 说明 |
# | :--- | :--- | :--- | :--- |
# | DASHSCOPE_API_KEY | 是 | 无 | Qwen 的 API Key |
# | RAG_IT_API_KEY | 否 | 无 | 覆盖上一项，便于切换多套凭据 |
# | RAG_IT_MILVUS_URI | 否 | http://127.0.0.1:19530 | Milvus 地址 |
# | RAG_IT_MILVUS_TOKEN | 否 | 无 | Milvus 鉴权令牌 |
# | RAG_IT_MILVUS_DB | 否 | default | Milvus 数据库名 |
# | RAG_IT_COLLECTION | 否 | rag_data_it_qwen_epub | 集合名 |
# | RAG_IT_MODEL | 否 | qwen3.7-text-embedding | 嵌入模型 |
# | RAG_IT_DIM | 否 | 1024 | 向量维度，须与模型输出一致 |
# | RAG_IT_MAX_CHARS | 否 | 1000 | 单块字符上限，注入各解析器 |
# | RAG_IT_SAFE_MAX_CHARS | 否 | 2000 | 单块安全上限，注入各解析器 |
# | RAG_IT_EPUB | 否 | 桌面首个 epub | 待处理的 EPUB 路径 |
# | RAG_IT_SAMPLE_CHARS | 否 | 3000 | 仅取解析结果前若干字符，控制请求量 |
# | RAG_IT_KEEP | 否 | 空 | 置 1 则跑完保留集合，默认删除 |
#
# 安全约定：只有集合名以 rag_data_it 开头、且 RAG_IT_KEEP 未置 1 时才删除集合，
# 避免误删他人数据。
#
# 说明：切块长度取自 parsing 分区，注入各解析器的 max_chars 与 safe_max_chars。

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Union
from openai import OpenAI

import pytest

from rag_data import RagData, Settings

# 凭据与连接信息一律来自运行时环境变量，代码中不写默认密钥。
MILVUS_URI = os.environ.get('RAG_IT_MILVUS_URI', 'http://127.0.0.1:19530')
MILVUS_TOKEN = os.environ.get('RAG_IT_MILVUS_TOKEN', 'rag')
MILVUS_DB = os.environ.get('RAG_IT_MILVUS_DB', 'autotest_rag_data')
COLLECTION = os.environ.get('RAG_IT_COLLECTION', 'rag_data_epub')
MODEL = os.environ.get('RAG_IT_MODEL', 'qwen3.7-text-embedding')
API_KEY = os.environ.get('RAG_IT_API_KEY') or os.environ.get('DASHSCOPE_API_KEY', '')
SAMPLE_CHARS = int(os.environ.get('RAG_IT_SAMPLE_CHARS', '3000'))
KEEP_COLLECTION = os.environ.get('RAG_IT_KEEP', '') == '1'

# 切块长度上限，经 parsing 分区注入各解析器。
MAX_CHARS = int(os.environ.get('RAG_IT_MAX_CHARS', '500'))
SAFE_MAX_CHARS = int(os.environ.get('RAG_IT_SAFE_MAX_CHARS', '2000'))
SPACY_MODEL = os.environ.get('RAG_IT_SPACY_MODEL', 'zh_core_web_sm')

VECTOR_DIM = int(os.environ.get('RAG_IT_DIM', '1024'))
# DashScope 兼容接口单次最多接受 10 条文本。
QWEN_MAX_BATCH = 10
# 只允许删除该前缀的集合。
COLLECTION_PREFIX = 'autotest'
# 连接别名，避免与其它进程的默认连接串号。
ALIAS = 'autotest'


def _local_epub() -> str:
    # 返回待处理的 epub 路径：环境变量优先，其次取桌面首个 epub。
    # override = os.environ.get('RAG_IT_EPUB',"C:\\Users\\loong\\Desktop\\三国演义 ([明] 罗贯中 著  [清] 毛宗岗 等批) (z-library.sk, 1lib.sk, z-lib.sk).epub")
    override = os.environ.get('RAG_IT_EPUB',"C:\\Users\\loong\\Desktop\\test_chunk.txt")
    if override:
        return override
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    if not os.path.isdir(desktop):
        return ''
    files = sorted(name for name in os.listdir(desktop) if name.lower().endswith('.epub'))
    return os.path.join(desktop, files[0]) if files else ''

EPUB_PATH = _local_epub()

def _requires_api_key():
    if not API_KEY:
        pytest.skip("未提供 API Key：请设置环境变量 DASHSCOPE_API_KEY 或 RAG_IT_API_KEY")

def _requires_pymilvus():
    if importlib.util.find_spec('pymilvus') is None:
        pytest.skip("未安装 pymilvus，请先执行 pip install pymilvus")

def _requires_epub():
    if not EPUB_PATH or not os.path.exists(EPUB_PATH):
        pytest.skip("未提供 EPUB：请设置环境变量 RAG_IT_EPUB 或在桌面放置 epub")


# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if "--integration" not in sys.argv:
    # 你可以直接在代码里写原因，或者写死一个统一的提示
    skip_reason = f"[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）"

    pytest.skip(skip_reason, allow_module_level=True)

# ---------------- 夹具 ----------------


@pytest.fixture(scope='module')
def it_settings() -> Settings:
    # 按集成测试需要构造配置：qwen 嵌入；凭据与模型来自环境变量。
    return Settings.load_mapping(
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
        },
        logging={
            'enable': True,
            'backend':'loguru',
        },
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

