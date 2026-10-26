# 运行：pytest tests 下的 test_integration_async_milvus_vector_store.py --integration -m integration -v -s
# 集合名取自 conftest 的 cases.vector_store.collection，必须以 autotest_ 开头。
# 异步用例依赖 pytest-asyncio（已在 dev 依赖中声明），缺插件时整个文件跳过。
# 集合与同步用例共用同一份配置，需要隔离时改 cases.vector_store.collection。

import inspect
import random
import sys
import time
from datetime import datetime, timezone

import pytest

from conftest import MILVUS_SETTINGS, VECTOR_SETTINGS, VECTOR_CONFIG

# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）',
                allow_module_level=True)

# 3. 异步用例需要 pytest-asyncio：缺插件时整体跳过，避免 async def 被静默忽略而误判为通过。
pytest_asyncio = pytest.importorskip('pytest_asyncio')

from rag_data.exceptions import StorageError  # noqa: E402
from rag_data.storage.vector_store import milvus_store as milvus_module  # noqa: E402
from rag_data.storage.vector_store.milvus_store import AsyncMilvusVectorStore, MilvusVectorStore  # noqa: E402
from rag_data.storage.vector_store.registry_vector import resolve_vector_store  # noqa: E402

COLLECTION = VECTOR_CONFIG['prefix'] + 'async_memories'
# 主键为 INT64：按当前秒数取余到分钟级再抬高基数，并追加随机位，避免重复运行撞主键。
BASE_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)


# ---------------- 前置校验与替身 ----------------


def _requires_async_pymilvus() -> None:
    '''缺 pymilvus 或版本低于 2.5 时跳过：异步客户端与 BM25 函数都要求 2.5 及以上。'''
    if milvus_module.AsyncMilvusClient is None:
        pytest.skip('未安装 pymilvus 或版本低于 2.5，请执行 pip install pymilvus>=2.5')


def _vector(index):
    '''构造可区分的单位向量：第 index 个位置为 1，其余为 0。'''
    size = VECTOR_SETTINGS.vector_size
    vector = [0.0] * size
    vector[index % size] = 1.0
    return vector


def _payload(index, tenant=BASE_ID, user_id=BASE_ID + 1, text='配置加载优先级', lemma='配置 加载 优先级'):
    '''最小合法 payload：tenant_id 必填；时间列统一用带时区的 ISO 字符串。'''
    stamp = datetime.now(tz=timezone.utc).isoformat()
    return {
        'tenant_id': tenant,
        'user_id': user_id,
        'text': text,
        'text_lemma': lemma,
        'created_at_tz': stamp,
        'doc_id': 'doc_' + str(index),
    }


async def _flush(client) -> None:
    '''刷新写入：部分版本的异步客户端不提供 flush，缺失时忽略。'''
    flusher = getattr(client, 'flush', None)
    if flusher is None:
        return
    await flusher(collection_name=COLLECTION)


# ---------------- 类结构与同步实现的差异 ----------------


def test_async_class_keeps_sync_backend_registration() -> None:
    '''异步实现不声明 backend：注册表里的 milvus 仍指向同步类，装配层行为不变。'''
    assert AsyncMilvusVectorStore.backend == 'milvus_async'
    assert resolve_vector_store('milvus_async') is AsyncMilvusVectorStore


def test_async_class_implements_all_abstract_methods() -> None:
    '''同步基类的抽象方法都被异步实现覆盖，可直接实例化。'''
    assert not AsyncMilvusVectorStore.__abstractmethods__


def test_async_public_api_is_coroutine() -> None:
    '''所有 IO 入口都是协程函数，漏加 async 会在这里暴露。'''
    names = ('start', 'create_col', 'insert', 'delete', 'update', 'get', 'search', 'keyword_search', 'list')
    for name in names:
        assert inspect.iscoroutinefunction(getattr(AsyncMilvusVectorStore, name)), name


def test_async_class_drops_sync_collection_management() -> None:
    '''删集合与重置未随异步实现提供：调用方需显式处理，避免误以为存在同名协程。'''
    assert 'delete_col' not in AsyncMilvusVectorStore.__dict__
    assert 'reset' not in AsyncMilvusVectorStore.__dict__


def test_async_store_reuses_sync_record_and_filter() -> None:
    '''记录组装与过滤表达式完全复用同步实现，两条路径的写入结构必须一致。'''
    sync_store = object.__new__(MilvusVectorStore)
    sync_store.collection_name = COLLECTION
    async_store = AsyncMilvusVectorStore(None, collection_name=COLLECTION, vector_size=VECTOR_SETTINGS.vector_size)
    payload = _payload(1)
    assert async_store._record(7, _vector(0), payload) == sync_store._record(7, _vector(0), payload)
    filters = {'tenant_id': BASE_ID, 'doc_id': 'doc_1'}
    assert async_store._create_filter(filters) == sync_store._create_filter(filters)


# ---------------- 真实服务：写入、检索、更新与删除 ----------------


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def async_store():
    '''按 conftest 的连接参数装配真实集合；服务不可用或缺 pymilvus 时跳过。'''
    _requires_async_pymilvus()
    if not COLLECTION.startswith('autotest_'):
        pytest.skip('集合名必须以 autotest_ 开头，当前为 ' + COLLECTION)
    try:
        store = await AsyncMilvusVectorStore(
            uri=MILVUS_SETTINGS.uri,
            token=MILVUS_SETTINGS.token,
            db_name=MILVUS_SETTINGS.db_name,
            collection_name=COLLECTION,
            vector_size=VECTOR_SETTINGS.vector_size,
        ).start()

    except Exception as exc:
        pytest.skip('跳过：Milvus 不可用（' + MILVUS_SETTINGS.uri + '）：' + type(exc).__name__ + ' ' + str(exc))

    try:
        await store.client.load_collection(collection_name=COLLECTION)
    except Exception:
        pass

    yield store


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def seeded(async_store):
    '''写入三条可区分的数据并刷新，供检索与更新用例复用。'''
    ids = [BASE_ID + index for index in range(3)]
    vectors = [_vector(index) for index in range(3)]
    payloads = [_payload(index) for index in range(3)]
    await async_store.insert(vectors, payloads, ids)
    await _flush(async_store.client)
    return ids


@pytest.mark.asyncio(loop_scope='module')
async def test_insert_and_get_roundtrip(async_store, seeded) -> None:
    '''写入后按 INT64 主键回查：id 与元数据都能读回。'''
    result = await async_store.get(seeded[0])
    assert result is not None
    assert result.id == seeded[0]
    assert result.metadata['doc_id'] == 'doc_0'


@pytest.mark.asyncio(loop_scope='module')
async def test_dense_search_returns_first_vector(async_store, seeded) -> None:
    '''用首条向量检索时，首位命中应为自身。'''
    hits = await async_store.search('配置', _vector(0), top_k=3, filters={'tenant_id': BASE_ID})
    assert hits
    assert hits[0].id == seeded[0]
    assert hits[0].score is not None


@pytest.mark.asyncio(loop_scope='module')
async def test_keyword_search_matches_text_lemma(async_store, seeded) -> None:
    '''BM25 走 text_lemma：中文查询应命中写入的中文词形文本。'''
    hits = await async_store.keyword_search('配置', top_k=3, filters={'tenant_id': BASE_ID})
    assert hits, 'BM25 未命中：确认 text_lemma 已开启 chinese 分析器且 pymilvus 支持 BM25'
    assert seeded[0] in [hit.id for hit in hits]


@pytest.mark.asyncio(loop_scope='module')
async def test_list_filters_by_tenant(async_store, seeded) -> None:
    '''按 tenant_id 过滤时只返回本租户数据；外层列表按既有约定包一层。'''
    rows = await async_store.list(filters={'tenant_id': BASE_ID}, top_k=10)
    assert len(rows) >= 1
    assert all(row.tenant_id == BASE_ID for row in rows)


@pytest.mark.asyncio(loop_scope='module')
async def test_list_returns_empty_for_other_tenant(async_store, seeded) -> None:
    '''其他租户过滤无命中，返回只含一个空列表的结构。'''
    assert await async_store.list(filters={'tenant_id': BASE_ID + 1}, top_k=10) == []


@pytest.mark.asyncio(loop_scope='module')
async def test_update_replaces_payload(async_store, seeded) -> None:
    '''更新元数据后回查：doc_id 被覆盖，旧向量沿用。'''
    payload = _payload(1)
    payload['doc_id'] = 'doc_updated'
    await async_store.update(seeded[1], payload=payload)
    await _flush(async_store.client)
    result = await async_store.get(seeded[1])
    assert result is not None
    assert result.metadata['doc_id'] == 'doc_updated'


@pytest.mark.asyncio(loop_scope='module')
async def test_update_reports_missing_id(async_store) -> None:
    '''更新不存在的 id 抛 StorageError，而不是静默插入新行。'''
    with pytest.raises(StorageError):
        await async_store.update(BASE_ID + 999999, payload=_payload(1))
