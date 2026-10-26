# 运行：pytest tests 目录下的 test_integration_async_pg_vector_store.py --integration -m integration -v -s
# 表名由 conftest 的 vector.prefix 生成，必须以 autotest_ 开头。
import random
import sys
import time
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from conftest import PG_VECTOR_SETTINGS, VECTOR_CONFIG, VECTOR_SETTINGS

# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）',
                allow_module_level=True)

from rag_data.exceptions import StorageError  # noqa: E402
from rag_data.storage.vector_store import pg_store as pg_module  # noqa: E402
from rag_data.storage.vector_store.pg_store import AsyncPGVectorStore  # noqa: E402

# 表名即集合名：由 conftest 的前缀生成，既满足删除守卫，也不与 Milvus 用例争用同名对象。
TABLE = VECTOR_CONFIG['prefix'] + 'async_memories'
DIM = VECTOR_SETTINGS.vector_size
# 原有逻辑放大，并加上 4 位随机数（0 ~ 9999）
BASE_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)


# ---------------- 前置校验与辅助函数 ----------------


def _requires_sqlalchemy_async() -> None:
    '''缺 sqlalchemy、asyncpg 或 pgvector 时跳过。'''
    if not pg_module._SQLALCHEMY_AVAILABLE:
        pytest.skip('未安装 sqlalchemy、asyncpg 或 pgvector，请执行 pip install sqlalchemy asyncpg pgvector')


def _bare_store(table=TABLE, dim=None):
    '''构造不连服务的实例：只供纯逻辑方法使用，避免装配时连库。'''
    store = object.__new__(AsyncPGVectorStore)
    store.collection_name = table
    store.vector_size = DIM if dim is None else dim
    if pg_module._SQLALCHEMY_AVAILABLE:
        store.metadata_obj = pg_module.MetaData()
    return store


def _vector(index):
    '''构造可区分的单位向量：第 index 个位置为 1，其余为 0。'''
    size = VECTOR_SETTINGS.vector_size
    vector = [0.0] * size
    vector[index % size] = 1.0
    return vector


def _payload(index, tenant=BASE_ID, user_id=BASE_ID + 1, text='异步配置加载优先级', lemma='异步 配置 加载 优先级'):
    '''PG 的行结构：作用域、正文与词形文本是顶层列，其余业务字段进 JSONB metadata。'''
    stamp = datetime.now(tz=timezone.utc)
    return {
        'tenant_id': tenant,
        'user_id': user_id,
        'text': text,
        'text_lemma': lemma,
        'updated_at_tz': stamp,
        'created_at_tz': stamp,
        'metadata': {'doc_id': 'async_doc_' + str(index)},
    }


def _sql(expression) -> str:
    '''把 SQLAlchemy 表达式编译为带字面量的 SQL 文本，便于断言生成的 WHERE 条件。'''
    from sqlalchemy.dialects import postgresql

    return str(expression.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))


# ---------------- 表名与删除守卫 ----------------


def test_col_returns_safe_table_name() -> None:
    '''合法表名原样返回，作为后续所有异步 SQL 的表名。'''
    assert _bare_store()._col() == TABLE


@pytest.mark.parametrize('name', ['bad name', 'autotest_x;drop', 'autotest-x', ''])
def test_col_rejects_illegal_table_name(name) -> None:
    '''只放行字母数字与下划线组合，其余一律拒绝，避免拼接出越权 SQL。'''
    with pytest.raises(StorageError):
        _bare_store(table=name)._col()


# ---------------- 表结构与过滤表达式验证 ----------------


def test_table_declares_integer_scope_columns() -> None:
    '''主键与作用域标识统一为 BIGINT，与 Milvus 的 INT64 对齐。'''
    _requires_sqlalchemy_async()
    table = _bare_store()._get_table()
    for name in ('id', 'tenant_id', 'user_id'):
        assert isinstance(table.c[name].type, pg_module.BigInteger)
    assert table.c.id.primary_key is True
    assert table.c.tenant_id.nullable is False


def test_filter_matches_direct_column() -> None:
    '''顶层列直接比较列值，不落到 JSONB 里。'''
    _requires_sqlalchemy_async()
    table = _bare_store()._get_table()
    sql = _sql(pg_module._build_sqlalchemy_filter(table, {'tenant_id': BASE_ID}))
    assert 'tenant_id' in sql
    assert str(BASE_ID) in sql
    assert 'metadata' not in sql


# ---------------- 真实服务异步集成测试：写入、检索、更新与删除 ----------------


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def async_pg_store():
    '''按 conftest 的 pg 分区装配真实异步表；服务不可用或缺依赖时跳过。'''
    _requires_sqlalchemy_async()
    if not TABLE.startswith('autotest_'):
        pytest.skip('表名必须以 autotest_ 开头，当前为 ' + TABLE)
    try:
        store = AsyncPGVectorStore(
            connection_string=PG_VECTOR_SETTINGS.connection_string,
            host=PG_VECTOR_SETTINGS.host,
            port=PG_VECTOR_SETTINGS.port,
            user=PG_VECTOR_SETTINGS.user,
            password=PG_VECTOR_SETTINGS.password,
            database=PG_VECTOR_SETTINGS.database,
            collection_name=TABLE,
            vector_size=DIM,
            use_hnsw=PG_VECTOR_SETTINGS.use_hnsw,
            use_diskann=PG_VECTOR_SETTINGS.use_diskann,
            print_sql=False,
        )
        # 显式初始化建表（内部包含 extension 检查、table 创建及索引建立）
        await store.create_col(TABLE, DIM)
    except Exception as exc:
        pytest.skip('跳过：PG 异步服务不可用（' + PG_VECTOR_SETTINGS.host + '）：' + type(exc).__name__ + ' ' + str(exc))

    yield store
    await store.engine.dispose()


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def async_seeded(async_pg_store):
    '''异步写入三条可区分的数据，供检索与更新用例复用。'''
    ids = [BASE_ID + index for index in range(3)]
    vectors = [_vector(index) for index in range(3)]
    payloads = [_payload(index) for index in range(3)]
    await async_pg_store.insert(vectors, payloads, ids)
    return ids


@pytest.mark.asyncio(loop_scope='module')
async def test_async_insert_and_get_roundtrip(async_pg_store, async_seeded) -> None:
    '''异步写入后按 BIGINT 主键回查：id 与 JSONB 元数据都能读回。'''
    result = await async_pg_store.get(async_seeded[0])
    assert result is not None
    assert result.id == async_seeded[0]
    assert result.metadata['doc_id'] == 'async_doc_0'


@pytest.mark.asyncio(loop_scope='module')
async def test_async_search_returns_self_as_top_hit(async_pg_store, async_seeded) -> None:
    '''异步用首条向量检索时首位命中为自身，余弦相似度接近 1。'''
    hits = await async_pg_store.search('异步配置', _vector(0), top_k=3, filters={'tenant_id': BASE_ID})
    assert hits
    assert hits[0].id == async_seeded[0]
    assert hits[1].score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio(loop_scope='module')
async def test_async_search_filters_by_metadata_field(async_pg_store, async_seeded) -> None:
    '''异步按 JSONB 业务字段过滤：只返回命中的那一条。'''
    hits = await async_pg_store.search('异步配置', _vector(1), top_k=3, filters={'doc_id': 'async_doc_1','tenant_id': BASE_ID})
    assert async_seeded[1] in [hit.id for hit in hits]


@pytest.mark.asyncio(loop_scope='module')
async def test_async_keyword_search_matches_text_lemma(async_pg_store, async_seeded) -> None:
    '''异步全文检索走 text_lemma：中文查询应命中写入的分词文本。'''
    hits = await async_pg_store.keyword_search('异步 配置', top_k=3, filters={'tenant_id': BASE_ID})
    assert hits, '异步全文检索未命中：确认 text_lemma 已写入且 GIN 索引表达式一致'
    assert async_seeded[0] in [hit.id for hit in hits]


@pytest.mark.asyncio(loop_scope='module')
async def test_async_list_filters_by_tenant(async_pg_store, async_seeded) -> None:
    '''异步按 tenant_id 列过滤只返回本租户数据。'''
    rows = await async_pg_store.list(filters={'tenant_id': BASE_ID}, top_k=10)
    assert len(rows) >= 1
    assert all(row.metadata['doc_id'].startswith('async_doc_') for row in rows)


@pytest.mark.asyncio(loop_scope='module')
async def test_async_update_replaces_metadata(async_pg_store, async_seeded) -> None:
    '''异步更新元数据后回查：JSONB 内容被整体覆盖。'''
    await async_pg_store.update(async_seeded[1], payload={'doc_id': 'async_doc_updated'})
    result = await async_pg_store.get(async_seeded[1])
    assert result is not None
    assert result.metadata['doc_id'] == 'async_doc_updated'


@pytest.mark.asyncio(loop_scope='module')
async def test_async_update_missing_id_is_noop(async_pg_store) -> None:
    '''记录当前异步行为：不存在的 id 更新不报错，也不会新增行。'''
    await async_pg_store.update(BASE_ID + 9999, payload={'doc_id': 'missing'})
    assert await async_pg_store.get(BASE_ID + 9999) is None