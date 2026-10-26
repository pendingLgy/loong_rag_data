# AsyncPGDB 单元测试：驱动探测、连接串改写、表前缀、引擎装配与建表检查。
#
# 引擎装配用例用 monkeypatch 隔离驱动与引擎；真实库用例连 conftest 的 PG_SETTINGS，
# 服务不可用时跳过，表名统一加 autotest_ 前缀并在用例结束后清理。
# sqlalchemy 缺失时相应用例跳过；同步用例见 test_integration_pgdb.py，本文件只覆盖异步类。
import random
import sys
import time

import pytest
import pytest_asyncio

try:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
    from rag_data.config import DBSettings  # noqa: E402
    from rag_data.exceptions import OptionalDependencyError, StorageError  # noqa: E402
    from rag_data.storage.db import pg_db as pg_module, next_id  # noqa: E402
    from rag_data.storage.db.base import BaseDB  # noqa: E402
    from rag_data.storage.db.db_engine import AsyncDBEngine, build_database  # noqa: E402
    from rag_data.storage.db.pg_db import AsyncPGDB  # noqa: E402
    from rag_data.storage.db.schema import HistoryRecord, Message  # noqa: E402
    from rag_data.storage.db.registry_db import available_dbs, is_db_registered, resolve_db  # noqa: E402
    _SQLALCHEMY_AVAILABLE = True
    _SQLALCHEMY_ERROR = None
except ImportError as _exc:
    _SQLALCHEMY_AVAILABLE = False
    _SQLALCHEMY_ERROR = _exc


pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)

def _requires_sqlalchemy() -> None:
    """缺 sqlalchemy 时跳过：引擎装配与 SQL 构造都依赖它。"""
    if not _SQLALCHEMY_AVAILABLE:
        pytest.skip("未安装 sqlalchemy，请执行 pip install sqlalchemy")


class _FakeResult:
    # 假结果集：只回答 fetchall。

    def __init__(self, rows):
        self.rows = list(rows)

    def fetchall(self):
        return list(self.rows)


class _FakeConnection:
    # 假连接：记录执行过的语句与参数，按给定行集返回结果。

    def __init__(self, rows):
        self.rows = list(rows)
        self.executed = []

    async def execute(self, statement, parameters=None):
        self.executed.append((statement, parameters))
        return _FakeResult(self.rows)


class _FakeBeginContext:
    # 假事务上下文：把 begin() 变成异步上下文管理器。

    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    # 假异步引擎：记录构造参数，并据此提供 begin()。

    def __init__(self, url=None, rows=(), **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.connection = _FakeConnection(rows)

    def begin(self):
        return _FakeBeginContext(self.connection)


def _patch_async_engine(monkeypatch, captured, rows=()):
    # 用记录型工厂替换 create_async_engine 与 async_sessionmaker。

    def engine_factory(url, **kwargs):
        captured['url'] = str(url)
        captured['kwargs'] = kwargs
        return _FakeEngine(url, rows=rows, **kwargs)

    def session_factory(bind=None, **kwargs):
        captured['session'] = dict(bind=bind, **kwargs)
        return dict(bind=bind, **kwargs)

    monkeypatch.setattr(pg_module, 'create_async_engine', engine_factory)
    monkeypatch.setattr(pg_module, 'async_sessionmaker', session_factory)


def _patch_driver(monkeypatch):
    # 固定异步驱动常量，避免受本机是否安装 asyncpg 或 psycopg 影响。
    monkeypatch.setattr(pg_module, '_ASYNC_PG_DRIVER', 'asyncpg')


def _dsn(scheme, user='root', password='pw', host='h', port=5432, database='rag'):
    # 拼出 scheme://user:password@host:port/database 形式的连接串。
    return '{}://{}:{}@{}:{}/{}'.format(scheme, user, password, host, port, database)


def _patch_init_db(monkeypatch):
    # 跳过建表检查：_missing_tables 由单独用例覆盖。

    async def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.AsyncSessionLocal = session_local

    monkeypatch.setattr(AsyncPGDB, '_init_db', _mock_init_db)


# ---------------- 注册与继承 ----------------


def test_async_backend_name_is_pg_async():
    _requires_sqlalchemy()
    assert AsyncPGDB.backend == 'pg_async'


def test_async_inherits_async_engine():
    _requires_sqlalchemy()
    assert issubclass(AsyncPGDB, AsyncDBEngine)
    assert issubclass(AsyncPGDB, BaseDB)


def test_async_registered_as_backend():
    _requires_sqlalchemy()
    assert is_db_registered('pg_async')
    assert resolve_db('pg_async') is AsyncPGDB
    assert 'pg_async' in available_dbs()


def test_async_implements_all_abstract_methods():
    # 异步后端必须自行实现 _start 与 _missing_tables，才可直接实例化。
    _requires_sqlalchemy()
    assert AsyncPGDB.__abstractmethods__ == frozenset()


# ---------------- 构造行为 ----------------


def test_async_default_parameters():
    _requires_sqlalchemy()
    store = AsyncPGDB()
    assert store.host == 'localhost'
    assert store.port == 5432
    assert store.user == 'postgres'
    assert store.password == ''
    assert store.database == 'rag_data'
    assert store.print_sql is False
    assert store.connection_string is None


def test_async_pool_parameters_default():
    _requires_sqlalchemy()
    store = AsyncPGDB()
    assert store.pool_size == 10
    assert store.max_overflow == 20
    assert store.pool_timeout == 30
    assert store.pool_recycle == 1800
    assert store.pool_pre_ping is True


def test_async_constructor_is_lazy():
    _requires_sqlalchemy()
    store = AsyncPGDB()
    assert store.engine is None
    assert store.AsyncSessionLocal is None


def test_async_constructor_keeps_custom_parameters():
    _requires_sqlalchemy()
    store = AsyncPGDB(connection_string='postgresql://u@h/db', host='h2', port=5433)
    assert store.connection_string == 'postgresql://u@h/db'
    assert store.host == 'h2'
    assert store.port == 5433


def test_async_constructor_does_not_create_engine(monkeypatch):
    _requires_sqlalchemy()
    # 构造期只保存参数，create_async_engine 不应被调用。

    def boom(url, **kwargs):
        raise AssertionError('构造期不应创建引擎')

    monkeypatch.setattr(pg_module, 'create_async_engine', boom)
    AsyncPGDB(host='h')


# ---------------- 表名前缀 ----------------


def test_async_default_tables_have_no_prefix():
    _requires_sqlalchemy()
    store = AsyncPGDB()
    assert store.table_prefix == ''
    assert store.history_table.name == 'history'
    assert store.messages_table.name == 'messages'
    assert {table.name for table in store.METADATA.sorted_tables} == {'history', 'messages'}


def test_async_table_prefix_applies_to_both_tables():
    _requires_sqlalchemy()
    store = AsyncPGDB(table_prefix='autotest_')
    assert store.table_prefix == 'autotest_'
    assert store.history_table.name == 'autotest_history'
    assert store.messages_table.name == 'autotest_messages'


def test_async_each_instance_owns_its_metadata():
    _requires_sqlalchemy()
    first = AsyncPGDB(table_prefix='autotest_a_')
    second = AsyncPGDB(table_prefix='autotest_b_')
    assert first.METADATA is not second.METADATA
    assert {table.name for table in second.METADATA.sorted_tables} == {'autotest_b_history', 'autotest_b_messages'}


# ---------------- 启动：驱动与连接串 ----------------


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_requires_driver(monkeypatch):
    _requires_sqlalchemy()
    monkeypatch.setattr(pg_module, '_ASYNC_PG_DRIVER', '')
    monkeypatch.setattr(pg_module, '_ASYNC_PG_ERROR', ImportError('missing'))
    with pytest.raises(OptionalDependencyError):
        await AsyncPGDB()._start()


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_builds_url_from_discrete_params(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB(host='db.internal', port=5433, user='app', password='p@ss', database='rag')
    await store._start()
    url = captured['url']
    assert url.startswith('postgresql+asyncpg')
    assert 'db.internal:5433' in url
    assert 'rag' in url


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_encodes_password(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    await AsyncPGDB(password='p@ss')._start()
    # 连接串按 URL 规则转义，明文密码不落串中。
    assert 'p@ss' not in captured['url']


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_assigns_engine_and_session_factory(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB()
    engine = await store._start()
    assert engine is store.engine
    assert isinstance(store.engine, _FakeEngine)
    assert captured['session']['bind'] is store.engine


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_session_factory_expires_off(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    await AsyncPGDB()._start()
    options = captured['session']
    assert options['class_'] is AsyncSession
    assert options['expire_on_commit'] is False


@pytest.mark.parametrize(
    'prefix',
    ['postgresql+psycopg2', 'postgresql+psycopg', 'postgresql+asyncpg', 'postgres', 'postgresql'],
)
@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_normalizes_connection_string(monkeypatch, prefix):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB(connection_string=_dsn(prefix))
    await store._start()
    url = captured['url']
    assert url.startswith('postgresql+asyncpg://')
    assert 'root:pw@h:5432/rag' in url


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_keeps_foreign_connection_string(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    foreign = 'mysql://u:p@h/db'
    store = AsyncPGDB(connection_string=foreign)
    await store._start()
    # 非 PG 前缀原样透传，不做改写。
    assert captured['url'] == foreign
    assert store.connection_string == foreign


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_passes_pool_arguments(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    await AsyncPGDB()._start()
    kwargs = captured['kwargs']
    assert kwargs['pool_size'] == 10
    assert kwargs['max_overflow'] == 20
    assert kwargs['pool_timeout'] == 30
    assert kwargs['pool_recycle'] == 1800
    assert kwargs['pool_pre_ping'] is True
    assert kwargs['echo'] is False


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_echo_follows_print_sql(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    await AsyncPGDB(print_sql=True)._start()
    assert captured['kwargs']['echo'] is True


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_wraps_engine_error(monkeypatch):
    _requires_sqlalchemy()
    _patch_driver(monkeypatch)

    def boom(url, **kwargs):
        raise RuntimeError('no server')

    monkeypatch.setattr(pg_module, 'create_async_engine', boom)
    with pytest.raises(StorageError):
        await AsyncPGDB()._start()


# ---------------- 建表检查 ----------------


@pytest.mark.asyncio(loop_scope='module')
async def test_async_missing_tables_detects_absent_tables(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured, rows=[('history',)])
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB()
    await store._start()
    missing = await store._missing_tables()
    assert [table.name for table in missing] == ['messages']


@pytest.mark.asyncio(loop_scope='module')
async def test_async_missing_tables_empty_when_complete(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured, rows=[('history',), ('messages',)])
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB()
    await store._start()
    assert await store._missing_tables() == []


@pytest.mark.asyncio(loop_scope='module')
async def test_async_missing_tables_queries_information_schema(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB()
    await store._start()
    await store._missing_tables()
    statement, parameters = store.engine.connection.executed[0]
    sql = str(statement)
    assert 'information_schema.tables' in sql
    assert 'current_schema()' in sql
    assert ':t0' in sql and ':t1' in sql
    assert set(parameters.values()) == {'history', 'messages'}


@pytest.mark.asyncio(loop_scope='module')
async def test_async_missing_tables_uses_prefixed_names(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = AsyncPGDB(table_prefix='autotest_')
    await store._start()
    missing = await store._missing_tables()
    assert [table.name for table in missing] == ['autotest_history', 'autotest_messages']
    _, parameters = store.engine.connection.executed[0]
    assert set(parameters.values()) == {'autotest_history', 'autotest_messages'}


# ---------------- build_database 装配 ----------------


def test_build_database_returns_async_pg():
    _requires_sqlalchemy()
    db = build_database(DBSettings(backend='pg_async'))
    assert isinstance(db, AsyncPGDB)
    assert db.database == 'rag_data'
    assert db.port == 5432


# ---------------- 真实数据库连接 ----------------
#
# 表名统一加 autotest_ 前缀：既与生产表隔离，也满足 storage 的删除守卫。

TABLE_PREFIX = 'async_'
RUN_ID = int(time.time()) % 1000000
HISTORY_MEMORY_ID = RUN_ID * 100 + 41
BATCH_MEMORY_ID = RUN_ID * 100 + 42
MESSAGE_SCOPE = 'autotest_pg_async_' + str(RUN_ID)


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def live_store():
    """真实 PG 连接：服务不可用时跳过，用完删表并释放连接池。"""
    _requires_sqlalchemy()
    from conftest import PG_SETTINGS

    store = AsyncPGDB(
        host=PG_SETTINGS.host,
        port=PG_SETTINGS.port,
        user=PG_SETTINGS.user,
        password=PG_SETTINGS.password,
        database=PG_SETTINGS.database,
        table_prefix=TABLE_PREFIX,
    )
    try:
        await store._start()
    except Exception as exc:
        pytest.skip('[SKIP INFO] PG 服务不可用：' + str(exc))

    try:
        yield store
    finally:
        if store.engine:
            await store.engine.dispose()


@pytest.mark.asyncio(loop_scope='module')
async def test_live_start_creates_tables(live_store):
    _requires_sqlalchemy()
    # 建表完成后不再有缺失项。
    assert await live_store._missing_tables() == []


@pytest.mark.asyncio(loop_scope='module')
async def test_live_table_names_are_prefixed(live_store):
    _requires_sqlalchemy()
    assert live_store.history_table.name == TABLE_PREFIX + 'history'
    assert live_store.messages_table.name == TABLE_PREFIX + 'messages'


@pytest.mark.asyncio(loop_scope='module')
async def test_live_start_assigns_session_local(live_store):
    _requires_sqlalchemy()
    assert live_store.engine is not None
    assert live_store.AsyncSessionLocal is not None
    assert live_store.session_local is live_store.AsyncSessionLocal


@pytest.mark.asyncio(loop_scope='module')
async def test_live_start_is_idempotent(live_store):
    _requires_sqlalchemy()
    previous = live_store.engine
    assert await live_store._start() is live_store.engine
    assert await live_store._missing_tables() == []
    await previous.dispose()


@pytest.mark.asyncio(loop_scope='module')
async def test_live_ensure_schema_recreates_dropped_table(live_store):
    _requires_sqlalchemy()
    async with live_store.engine.begin() as conn:
        await conn.run_sync(live_store.messages_table.drop, checkfirst=True)
    missing = await live_store._missing_tables()
    assert [table.name for table in missing] == [live_store.messages_table.name]
    await live_store._ensure_schema()
    assert await live_store._missing_tables() == []


@pytest.mark.asyncio(loop_scope='module')
async def test_live_add_history_persists(live_store):
    _requires_sqlalchemy()
    saved = await live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, new_memory='v1', event='ADD', actor_id='u1')
    )
    assert isinstance(saved, HistoryRecord)
    assert isinstance(saved.id, int)
    assert saved.memory_id == HISTORY_MEMORY_ID
    assert saved.event == 'ADD'
    assert saved.actor_id == 'u1'
    assert saved.is_deleted is False


@pytest.mark.asyncio(loop_scope='module')
async def test_live_get_history_orders_by_time(live_store):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    await live_store.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, event='ADD', created_at_tz=base)
    )
    await live_store.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, event='UPDATE', created_at_tz=base + timedelta(seconds=1))
    )
    rows = await live_store.get_history(TEMP_HISTORY_MEMORY_ID)
    assert ['UPDATE' , 'ADD'] == [row.event for row in rows][-2:]


@pytest.mark.asyncio(loop_scope='module')
async def test_live_batch_add_history_returns_records(live_store):
    _requires_sqlalchemy()
    saved = await live_store.batch_add_history([
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b1', event='ADD'),
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b2', event='ADD'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item.id, int) for item in saved)
    assert len(await live_store.get_history(BATCH_MEMORY_ID)) == 2


@pytest.mark.asyncio(loop_scope='module')
async def test_live_batch_add_history_empty_returns_empty(live_store):
    _requires_sqlalchemy()
    assert await live_store.batch_add_history([]) == []


@pytest.mark.asyncio(loop_scope='module')
async def test_live_save_messages_keeps_last_only(live_store):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    await live_store.save_messages([
        Message(session_scope=MESSAGE_SCOPE, content='e1', created_at_tz=base),
        Message(session_scope=MESSAGE_SCOPE, content='e2', created_at_tz=base + timedelta(seconds=1)),
    ], keep_last=2)
    await live_store.save_messages([
        Message(session_scope=MESSAGE_SCOPE, content='e3', created_at_tz=base + timedelta(seconds=2)),
    ], keep_last=2)
    rows = await live_store.get_last_messages(MESSAGE_SCOPE, limit=10)
    assert ['e2', 'e3'] == [item.content for item in rows]


@pytest.mark.asyncio(loop_scope='module')
async def test_live_get_last_messages_respects_limit(live_store):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    scope = MESSAGE_SCOPE + '_limit'
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    await live_store.save_messages([
        Message(session_scope=scope, content='c1', created_at_tz=base),
        Message(session_scope=scope, content='c2', created_at_tz=base + timedelta(seconds=1)),
        Message(session_scope=scope, content='c3', created_at_tz=base + timedelta(seconds=2)),
    ])
    rows = await live_store.get_last_messages(scope, limit=2)
    assert all(isinstance(item, Message) for item in rows)
    assert ['c2', 'c3'] == [item.content for item in rows]
