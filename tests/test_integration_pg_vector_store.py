# 运行：pytest tests 目录下的 test_integration_pg_vector_store.py --integration -m integration -v -s
# 表名由 conftest 的 vector.prefix 生成，必须以 autotest_ 开头。

import sys
import time
from datetime import datetime, timezone

import pytest

from conftest import PG_VECTOR_SETTINGS, VECTOR_CONFIG, VECTOR_SETTINGS

# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)

from rag_data.exceptions import OptionalDependencyError, StorageError  # noqa: E402
from rag_data.storage.vector_store import pg_store as pg_module  # noqa: E402
from rag_data.storage.vector_store.pg_store import PGVectorStore  # noqa: E402

# 表名即集合名：由 conftest 的前缀生成，既满足删除守卫，也不与 Milvus 用例争用同名对象。
TABLE = VECTOR_CONFIG['prefix'] + 'memories'
DIM = VECTOR_SETTINGS.vector_size
# 主键与 tenant_id 共用同一基数（均为 BIGINT）：按当前秒数取余到分钟级，避免重复运行撞主键。
BASE_ID = divmod(int(time.time()), 60)[1] * 1000 + 900000


# ---------------- 前置校验与替身 ----------------


def _requires_sqlalchemy() -> None:
    '''缺 sqlalchemy 或 pgvector 时跳过：表结构与检索表达式都依赖二者。'''
    if not pg_module._SQLALCHEMY_AVAILABLE:
        pytest.skip('未安装 sqlalchemy 或 pgvector，请执行 pip install sqlalchemy pgvector')


def _bare_store(table=TABLE, dim=None):
    '''构造不连服务的实例：只供纯逻辑方法使用，避免装配时连库。'''
    store = object.__new__(PGVectorStore)
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


def _payload(index, tenant=BASE_ID, user_id=BASE_ID + 1, text='配置加载优先级', lemma='配置 加载 优先级'):
    '''PG 的行结构：作用域、正文与词形文本是顶层列，其余业务字段进 JSONB metadata。'''
    stamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return {
        'tenant_id': tenant,
        'user_id': user_id,
        'text': text,
        'text_lemma': lemma,
        'updated_at_tz': stamp,
        'created_at_tz': stamp,
        'metadata': {'doc_id': 'doc_' + str(index)},
    }


def _sql(expression) -> str:
    '''把 SQLAlchemy 表达式编译为带字面量的 SQL 文本，便于断言生成的 WHERE 条件。'''
    from sqlalchemy.dialects import postgresql

    return str(expression.compile(dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}))


class _Result:
    '''假执行结果：只实现 fetchone，供建表语句里的扩展探测使用。'''

    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class _Conn:
    '''假连接：记录执行过的语句，不真正连库。'''

    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return _Result(self.rows)


class _Transaction:
    '''假事务上下文：进入返回假连接，退出不提交。'''

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _Engine:
    '''假引擎：begin() 返回包住假连接的事务上下文。'''

    def __init__(self, rows):
        self.conn = _Conn(rows)

    def begin(self):
        return _Transaction(self.conn)


class _StubTable:
    '''假 Table：只记录建表调用，避免依赖真实连接执行 DDL。'''

    def __init__(self, statements):
        self.statements = statements

    def create(self, conn, checkfirst=True):
        flag = 'checkfirst' if checkfirst else 'force'
        self.statements.append('CREATE TABLE IF NOT EXISTS ' + flag)


def _recording_store(rows=1, dim=None, use_hnsw=True, use_diskann=False):
    '''构造只记录 DDL 的实例：假引擎与假表替换真实连接，返回实例与语句列表。'''
    store = _bare_store(dim=dim)
    engine = _Engine(rows)
    store.engine = engine
    store.use_hnsw = use_hnsw
    store.use_diskann = use_diskann
    store._get_table = lambda: _StubTable(engine.conn.statements)
    return store, engine.conn.statements


# ---------------- 表名与删除守卫 ----------------


def test_col_returns_safe_table_name() -> None:
    '''合法表名原样返回，作为后续所有 SQL 的表名。'''
    assert _bare_store()._col() == TABLE


@pytest.mark.parametrize('name', ['bad name', 'autotest_x;drop', 'autotest-x', ''])
def test_col_rejects_illegal_table_name(name) -> None:
    '''只放行字母数字与下划线组合，其余一律拒绝，避免拼接出越权 SQL。'''
    with pytest.raises(StorageError):
        _bare_store(table=name)._col()


# ---------------- 表结构（与 Milvus 集合字段对齐） ----------------


def test_direct_columns_cover_row_structure() -> None:
    '''顶层列名单与 insert 使用的键一一对应，过滤时才能把顶层键识别为真实列。'''
    assert pg_module.DIRECT_COLUMNS == {
        'id', 'tenant_id', 'user_id', 'vector', 'text', 'text_lemma',
        'updated_at_tz', 'created_at_tz',
    }


def test_text_index_expression_targets_text_lemma() -> None:
    '''全文索引表达式基于 text_lemma 列：与关键词检索保持一致，否则索引不生效。'''
    assert 'to_tsvector' in pg_module._TEXT_VECTOR
    assert 'text_lemma' in pg_module._TEXT_VECTOR


def test_table_declares_integer_scope_columns() -> None:
    '''主键与作用域标识统一为 BIGINT，与 Milvus 的 INT64 对齐。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    for name in ('id', 'tenant_id', 'user_id'):
        assert isinstance(table.c[name].type, pg_module.BigInteger)
    assert table.c.id.primary_key is True
    assert table.c.tenant_id.nullable is False


def test_table_declares_vector_and_text_columns() -> None:
    '''向量列维度取自实例声明，正文与词形文本各占一列。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    assert table.c.vector.type.dim == DIM
    assert isinstance(table.c.text.type, pg_module.Text)
    assert isinstance(table.c.text_lemma.type, pg_module.Text)


def test_table_is_cached_in_metadata() -> None:
    '''同一实例重复取表返回同一对象，避免重复反射与重复建表。'''
    _requires_sqlalchemy()
    store = _bare_store()
    assert store._get_table() is store._get_table()


# ---------------- 过滤表达式（_build_sqlalchemy_filter） ----------------


def test_filter_returns_none_without_conditions() -> None:
    '''空过滤返回 None，不给语句拼上多余 WHERE。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    assert pg_module._build_sqlalchemy_filter(table, None) is None
    assert pg_module._build_sqlalchemy_filter(table, {}) is None


def test_filter_matches_direct_column() -> None:
    '''顶层列直接比较列值，不落到 JSONB 里。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    sql = _sql(pg_module._build_sqlalchemy_filter(table, {'tenant_id': BASE_ID}))
    assert 'tenant_id' in sql
    assert str(BASE_ID) in sql
    assert 'metadata' not in sql


def test_filter_matches_metadata_field() -> None:
    '''非顶层键落到 JSONB：业务字段才能真正被过滤。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    sql = _sql(pg_module._build_sqlalchemy_filter(table, {'doc_id': 'doc_1'}))
    assert 'metadata' in sql
    assert 'doc_id' in sql
    assert 'doc_1' in sql


def test_filter_wildcard_checks_presence() -> None:
    '''通配值表示只要求字段存在：顶层列用 IS NOT NULL，JSONB 用存在运算符。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    assert 'IS NOT NULL' in _sql(pg_module._build_sqlalchemy_filter(table, {'tenant_id': '*'})).upper()
    assert '?' in _sql(pg_module._build_sqlalchemy_filter(table, {'doc_id': '*'}))


def test_filter_supports_comparison_operators() -> None:
    '''字典形式支持比较运算符，JSONB 数值字段先转 numeric 再比较。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    cases = (('eq', ' = '), ('ne', ' != '), ('gt', ' > '), ('gte', ' >= '), ('lt', ' < '), ('lte', ' <= '))
    for op, token in cases:
        sql = _sql(pg_module._build_sqlalchemy_filter(table, {'count': {op: 3}}))
        assert token in sql
        assert 'metadata' in sql


def test_filter_supports_in_and_nin() -> None:
    '''列表值与 in 生成 IN 条件，nin 生成 NOT IN 条件。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    ids = [BASE_ID, BASE_ID + 1]
    assert ' IN ' in _sql(pg_module._build_sqlalchemy_filter(table, {'tenant_id': ids})).upper()
    assert 'NOT IN' in _sql(pg_module._build_sqlalchemy_filter(table, {'tenant_id': {'nin': ids}})).upper()


def test_filter_supports_contains_and_icontains() -> None:
    '''contains 走 LIKE，icontains 走不区分大小写的比较。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    assert 'LIKE' in _sql(pg_module._build_sqlalchemy_filter(table, {'text': {'contains': '配置'}})).upper()
    assert 'LIKE' in _sql(pg_module._build_sqlalchemy_filter(table, {'text': {'icontains': '配置'}})).upper()


def test_filter_keeps_bool_as_json_literal() -> None:
    '''布尔值按 JSON 字面量比较，避免被当成列名或整数。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    assert 'true' in _sql(pg_module._build_sqlalchemy_filter(table, {'doc_id': True}))


def test_filter_rejects_unknown_operator() -> None:
    '''未知操作符直接报错，不静默忽略过滤条件。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    with pytest.raises(ValueError):
        pg_module._build_sqlalchemy_filter(table, {'count': {'between': 1}})


def test_filter_rejects_in_operator_without_list() -> None:
    '''in 与 nin 只接受列表值，标量值直接报错。'''
    _requires_sqlalchemy()
    table = _bare_store()._get_table()
    with pytest.raises(ValueError):
        pg_module._build_sqlalchemy_filter(table, {'count': {'in': 1}})


# ---------------- 建表与索引（create_col） ----------------


def test_create_col_builds_hnsw_and_fulltext_index() -> None:
    '''默认建 HNSW 向量索引与基于 text_lemma 的 GIN 全文索引。'''
    _requires_sqlalchemy()
    store, statements = _recording_store()
    store.create_col(TABLE, DIM)
    assert statements[0].startswith('CREATE EXTENSION IF NOT EXISTS vector')
    assert [item for item in statements if item.startswith('CREATE TABLE')]
    hnsw = [item for item in statements if '_hnsw_idx' in item]
    assert hnsw and 'hnsw (vector vector_cosine_ops)' in hnsw[0]
    fulltext = [item for item in statements if '_text_idx' in item]
    assert fulltext and 'gin (' in fulltext[0] and 'text_lemma' in fulltext[0]


def test_create_col_skips_vector_index_when_disabled() -> None:
    '''关闭 use_hnsw 时不建向量索引，但全文索引仍然创建。'''
    _requires_sqlalchemy()
    store, statements = _recording_store(use_hnsw=False)
    store.create_col(TABLE, DIM)
    assert not [item for item in statements if '_hnsw_idx' in item]
    assert [item for item in statements if '_text_idx' in item]


def test_create_col_uses_diskann_when_extension_exists() -> None:
    '''启用 DiskANN 且扩展存在时建 diskann 索引，不再建 HNSW。'''
    _requires_sqlalchemy()
    store, statements = _recording_store(rows=1, use_diskann=True)
    store.create_col(TABLE, DIM)
    assert [item for item in statements if '_diskann_idx' in item]
    assert not [item for item in statements if '_hnsw_idx' in item]


def test_create_col_ignores_diskann_over_dimension_limit() -> None:
    '''维度达到 DiskANN 上限时不再建 diskann 索引，退回 HNSW。'''
    _requires_sqlalchemy()
    limit = pg_module.DISKANN_MAX_DIMENSION
    store, statements = _recording_store(dim=limit, use_diskann=True)
    store.create_col(TABLE, limit)
    assert not [item for item in statements if '_diskann_idx' in item]
    assert [item for item in statements if '_hnsw_idx' in item]


# ---------------- 真实服务：写入、检索、更新与删除 ----------------


@pytest.fixture(scope='module')
def pg_store():
    '''按 conftest 的 pg 分区装配真实表；服务不可用或缺依赖时跳过。'''
    _requires_sqlalchemy()
    if not TABLE.startswith('autotest_'):
        pytest.skip('表名必须以 autotest_ 开头，当前为 ' + TABLE)
    try:
        store = PGVectorStore(
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
            print_sql=True,
        )
    except Exception as exc:
        pytest.skip('跳过：PG 不可用（' + PG_VECTOR_SETTINGS.host + '）：' + type(exc).__name__ + ' ' + str(exc))
    yield store
    # if not VECTOR_CONFIG.get('keep'):
    #     try:
    #         store.delete_col()
    #     except Exception:
    #         pass
    store.engine.dispose()


@pytest.fixture(scope='module')
def seeded(pg_store):
    '''写入三条可区分的数据，供检索与更新用例复用。'''
    ids = [BASE_ID + index for index in range(3)]
    vectors = [_vector(index) for index in range(3)]
    payloads = [_payload(index) for index in range(3)]
    pg_store.insert(vectors, payloads, ids)
    return ids


def test_insert_and_get_roundtrip(pg_store, seeded) -> None:
    '''写入后按 BIGINT 主键回查：id 与 JSONB 元数据都能读回。'''
    result = pg_store.get(seeded[0])
    assert result is not None
    assert result.id == seeded[0]
    assert result.metadata['doc_id'] == 'doc_0'


def test_search_returns_self_as_top_hit(pg_store, seeded) -> None:
    '''用首条向量检索时首位命中为自身，余弦相似度接近 1。'''
    hits = pg_store.search('配置', _vector(0), top_k=3, filters={'tenant_id': BASE_ID})
    assert hits
    assert hits[0].id == seeded[0]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_search_filters_by_metadata_field(pg_store, seeded) -> None:
    '''按 JSONB 业务字段过滤：只返回命中的那一条。'''
    hits = pg_store.search('配置', _vector(1), top_k=3, filters={'doc_id': 'doc_1'})
    assert [hit.id for hit in hits] == [seeded[1]]


def test_keyword_search_matches_text_lemma(pg_store, seeded) -> None:
    '''全文检索走 text_lemma：中文查询应命中写入的分词文本。'''
    hits = pg_store.keyword_search('配置', top_k=3, filters={'tenant_id': BASE_ID})
    assert hits, '全文检索未命中：确认 text_lemma 已写入且 GIN 索引表达式一致'
    assert seeded[0] in [hit.id for hit in hits]

def test_keyword_search_matches_text_lemmas(pg_store, seeded) -> None:
    '''全文检索走 text_lemma：中文查询应命中写入的分词文本。'''
    hits = pg_store.keyword_search('配置 优先级', top_k=3, filters={'tenant_id': BASE_ID})
    assert hits, '全文检索未命中：确认 text_lemma 已写入且 GIN 索引表达式一致'
    assert seeded[0] in [hit.id for hit in hits]

def test_list_filters_by_tenant(pg_store, seeded) -> None:
    '''按 tenant_id 列过滤只返回本租户数据；外层列表按既有约定包一层。'''
    rows = pg_store.list(filters={'tenant_id': BASE_ID}, top_k=10)
    assert len(rows) == 1
    assert len(rows[0]) >= 3
    assert all(row.metadata['doc_id'].startswith('doc_') for row in rows[0])


def test_list_returns_empty_for_other_tenant(pg_store, seeded) -> None:
    '''其他租户过滤无命中，返回只含一个空列表的结构。'''
    assert pg_store.list(filters={'tenant_id': BASE_ID + 1}, top_k=10) == [[]]


def test_update_replaces_metadata(pg_store, seeded) -> None:
    '''更新元数据后回查：JSONB 内容被整体覆盖。'''
    pg_store.update(seeded[1], payload={'doc_id': 'doc_updated'})
    result = pg_store.get(seeded[1])
    assert result is not None
    assert result.metadata['doc_id'] == 'doc_updated'


def test_update_missing_id_is_noop(pg_store) -> None:
    '''记录当前行为：不存在的 id 更新不报错，也不会新增行。'''
    pg_store.update(BASE_ID + 9999, payload={'doc_id': 'missing'})
    assert pg_store.get(BASE_ID + 9999) is None


def test_delete_removes_row(pg_store) -> None:
    '''删除后按同一主键回查应为 None。'''
    ids = [BASE_ID + 500]
    pg_store.insert([_vector(1)], [_payload(9)], ids)
    pg_store.delete(ids[0])
    assert pg_store.get(ids[0]) is None

