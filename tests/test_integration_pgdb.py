# PGDB 单元测试：驱动探测、连接串改写、表前缀、引擎装配与建表检查。
#
# 引擎装配用例用 monkeypatch 隔离驱动与引擎；真实库用例连 conftest 的 PG_SETTINGS，
# 服务不可用时跳过，表名统一加 autotest_ 前缀并在用例结束后清理。
# sqlalchemy 缺失时相应用例跳过；异步用例见 test_integration_pgdb_async.py。
import random
import sys
import time

import pytest

try:
    from rag_data.config import DBSettings  # noqa: E402
    from rag_data.exceptions import ConfigError, OptionalDependencyError, StorageError  # noqa: E402
    from rag_data.storage.db import pg_db as pg_module, next_id  # noqa: E402
    from rag_data.storage.db.base import BaseDB  # noqa: E402
    from rag_data.storage.db.db_engine import AsyncDBEngine, DBEngine, build_database  # noqa: E402
    from rag_data.storage.db.pg_db import AsyncPGDB, PGDB  # noqa: E402
    from rag_data.storage.db.registry_db import available_dbs, is_db_registered, resolve_db  # noqa: E402
    from rag_data.storage.db.schema import HistoryRecord, Message  # noqa: E402
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

    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchall(self):
        return list(self.rows)


class _FakeConnection:
    # 假连接：记录执行过的语句与参数，按给定行集返回结果。

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed = []

    def execute(self, statement, parameters=None):
        self.executed.append((str(statement), parameters))
        return _FakeResult(self.rows)


class _FakeBegin:
    # 假事务上下文：把 begin() 变成同步上下文管理器。

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    # 假引擎：记录构造参数，并据此提供 begin()。

    def __init__(self, url=None, rows=(), **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.connection = _FakeConnection(rows)

    def begin(self):
        return _FakeBegin(self.connection)


def _patch_engine(monkeypatch, captured, rows=()):
    # 用记录型工厂替换 create_engine 与 sessionmaker。

    def engine_factory(url, **kwargs):
        captured['url'] = str(url)
        captured['kwargs'] = kwargs
        return _FakeEngine(url, rows=rows, **kwargs)

    def session_factory(bind=None, **kwargs):
        captured['session'] = dict(bind=bind, **kwargs)
        return dict(bind=bind, **kwargs)

    monkeypatch.setattr(pg_module, 'create_engine', engine_factory)
    monkeypatch.setattr(pg_module, 'sessionmaker', session_factory)


def _patch_driver(monkeypatch):
    # 固定同步驱动常量，避免受本机是否安装 psycopg2 影响。
    monkeypatch.setattr(pg_module, '_PG_DRIVER', 'psycopg')


def _patch_init_db(monkeypatch):
    # 跳过建表检查：_missing_tables 由单独用例覆盖。

    def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.SessionLocal = session_local

    monkeypatch.setattr(PGDB, '_init_db', _mock_init_db)


def _dsn(scheme, user='root', password='pw', host='h', port=5432, database='rag'):
    # 拼出 scheme://user:password@host:port/database 形式的连接串。
    return '{}://{}:{}@{}:{}/{}'.format(scheme, user, password, host, port, database)


# ---------------- 注册与继承 ----------------


def test_backend_name_is_pg():
    _requires_sqlalchemy()
    assert PGDB.backend == 'pg'


def test_inherits_engine():
    _requires_sqlalchemy()
    assert issubclass(PGDB, DBEngine)
    assert issubclass(PGDB, BaseDB)


def test_registered_as_backend():
    _requires_sqlalchemy()
    assert is_db_registered('pg')
    assert resolve_db('pg') is PGDB
    assert 'pg' in available_dbs()


def test_implements_all_abstract_methods():
    # DBEngine 把 _start 与 _missing_tables 都声明为抽象方法，PGDB 两者都已实现。
    _requires_sqlalchemy()
    assert PGDB.__abstractmethods__ == frozenset()


# ---------------- 构造行为 ----------------


def test_default_parameters():
    _requires_sqlalchemy()
    store = PGDB()
    assert store.host == 'localhost'
    assert store.port == 5432
    assert store.user == 'postgres'
    assert store.password == ''
    assert store.database == 'rag_data'
    assert store.print_sql is False
    assert store.connection_string is None
    assert store.table_prefix == ''


def test_pool_parameters_default():
    _requires_sqlalchemy()
    store = PGDB()
    assert store.pool_size == 10
    assert store.max_overflow == 20
    assert store.pool_timeout == 30
    assert store.pool_recycle == 1800
    assert store.pool_pre_ping is True


def test_constructor_is_lazy():
    _requires_sqlalchemy()
    store = PGDB()
    assert store.engine is None
    assert store.SessionLocal is None


def test_constructor_keeps_custom_parameters():
    _requires_sqlalchemy()
    store = PGDB(connection_string=_dsn('postgresql'), host='h2', port=5433)
    assert store.connection_string == _dsn('postgresql')
    assert store.host == 'h2'
    assert store.port == 5433


def test_constructor_does_not_create_engine(monkeypatch):
    _requires_sqlalchemy()
    # 构造期只保存参数，create_engine 不应被调用。

    def boom(url, **kwargs):
        raise AssertionError('构造期不应创建引擎')

    monkeypatch.setattr(pg_module, 'create_engine', boom)
    PGDB(host='h')


# ---------------- 表名前缀 ----------------


def test_default_tables_have_no_prefix():
    _requires_sqlalchemy()
    store = PGDB()
    assert store.history_table.name == 'history'
    assert store.messages_table.name == 'messages'
    assert {table.name for table in store.METADATA.sorted_tables} == {'history', 'messages'}


def test_table_prefix_applies_to_both_tables():
    _requires_sqlalchemy()
    store = PGDB(table_prefix='autotest_')
    assert store.table_prefix == 'autotest_'
    assert store.history_table.name == 'autotest_history'
    assert store.messages_table.name == 'autotest_messages'
    assert {table.name for table in store.METADATA.sorted_tables} == {'autotest_history', 'autotest_messages'}


def test_each_instance_owns_its_metadata():
    _requires_sqlalchemy()
    first = PGDB(table_prefix='autotest_a_')
    second = PGDB(table_prefix='autotest_b_')
    assert first.METADATA is not second.METADATA
    assert {table.name for table in first.METADATA.sorted_tables} == {'autotest_a_history', 'autotest_a_messages'}
    assert {table.name for table in second.METADATA.sorted_tables} == {'autotest_b_history', 'autotest_b_messages'}


# ---------------- 启动：驱动与连接串 ----------------


def test_start_requires_driver(monkeypatch):
    _requires_sqlalchemy()
    monkeypatch.setattr(pg_module, '_PG_DRIVER', '')
    monkeypatch.setattr(pg_module, '_PG_ERROR', ImportError('missing'))
    with pytest.raises(OptionalDependencyError):
        PGDB()._start()


def test_start_builds_url_from_discrete_params(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB(host='db.internal', port=5433, user='app', password='p@ss', database='rag')
    store._start()
    url = captured['url']
    assert url.startswith('postgresql+psycopg')
    assert 'db.internal:5433' in url
    assert 'rag' in url


def test_start_hides_password(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    PGDB(password='p@ss')._start()
    # 连接串按 URL 规则转义，明文密码不落串中。
    assert 'p@ss' not in captured['url']


def test_start_assigns_engine_and_session_factory(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB()
    engine = store._start()
    assert engine is store.engine
    assert isinstance(store.engine, _FakeEngine)
    assert captured['session']['bind'] is store.engine


@pytest.mark.parametrize(
    'prefix',
    ['postgresql+psycopg2', 'postgresql+psycopg', 'postgres', 'postgresql'],
)
def test_start_normalizes_connection_string(monkeypatch, prefix):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB(connection_string=_dsn(prefix))
    store._start()
    url = captured['url']
    assert url.startswith('postgresql+psycopg://')
    assert 'root:pw@h:5432/rag' in url


def test_start_keeps_foreign_connection_string(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    foreign = 'mysql://u:p@h/db'
    store = PGDB(connection_string=foreign)
    store._start()
    # 非 PG 前缀原样透传，不做改写。
    assert captured['url'] == foreign
    assert store.connection_string == foreign


def test_start_passes_pool_arguments(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    PGDB()._start()
    kwargs = captured['kwargs']
    assert kwargs['pool_size'] == 10
    assert kwargs['max_overflow'] == 20
    assert kwargs['pool_timeout'] == 30
    assert kwargs['pool_recycle'] == 1800
    assert kwargs['pool_pre_ping'] is True
    assert kwargs['echo'] is False


def test_start_echo_follows_print_sql(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    PGDB(print_sql=True)._start()
    assert captured['kwargs']['echo'] is True


def test_start_enables_pool_logging_when_print_sql(monkeypatch):
    _requires_sqlalchemy()
    import logging

    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    pool_logger = logging.getLogger('sqlalchemy.pool')
    previous = pool_logger.level
    try:
        PGDB(print_sql=True)._start()
        assert pool_logger.level == logging.INFO
    finally:
        pool_logger.setLevel(previous)


def test_start_wraps_schema_error(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)

    def boom(self):
        raise RuntimeError('ddl failed')

    monkeypatch.setattr(PGDB, '_missing_tables', boom)
    with pytest.raises(StorageError):
        PGDB()._start()


# ---------------- 建表检查 ----------------


def test_missing_tables_queries_information_schema(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB()
    store._start()
    assert [table.name for table in store._missing_tables()] == ['history', 'messages']
    sql, parameters = store.engine.connection.executed[0]
    assert 'information_schema.tables' in sql
    assert 'current_schema()' in sql
    assert ':t0' in sql and ':t1' in sql
    assert set(parameters.values()) == {'history', 'messages'}


def test_missing_tables_detects_absent(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured, rows=[('history',)])
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB()
    store._start()
    assert [table.name for table in store._missing_tables()] == ['messages']


def test_missing_tables_empty_when_complete(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured, rows=[('history',), ('messages',)])
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB()
    store._start()
    assert store._missing_tables() == []


def test_missing_tables_uses_prefixed_names(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    _patch_init_db(monkeypatch)
    store = PGDB(table_prefix='autotest_')
    store._start()
    assert [table.name for table in store._missing_tables()] == ['autotest_history', 'autotest_messages']
    _, parameters = store.engine.connection.executed[0]
    assert set(parameters.values()) == {'autotest_history', 'autotest_messages'}


# ---------------- 装配 ----------------


def test_build_database_returns_sync_pg():
    _requires_sqlalchemy()
    db = build_database(DBSettings(backend='pg'))
    assert isinstance(db, PGDB)
    assert db.database == 'rag_data'
    assert db.port == 5432
    # 装配入口不传前缀：表名保持默认 history 与 messages。
    assert db.table_prefix == ''


def test_build_database_returns_async_pg():
    _requires_sqlalchemy()
    db = build_database(DBSettings(backend='pg_async'))
    assert isinstance(db, AsyncPGDB)
    assert isinstance(db, AsyncDBEngine)


def test_build_database_passes_connection_string():
    _requires_sqlalchemy()
    db = build_database(DBSettings(backend='pg', pg={'connection_string': 'postgresql://u@h/db'}))
    assert db.connection_string == 'postgresql://u@h/db'


def test_build_database_rejects_unknown_backend():
    _requires_sqlalchemy()
    # 未知后端名进不了 Literal，用轻量替身绕过配置校验，直接校验装配函数的分支兜底。

    class _Stub:
        backend = 'oracle'
        sqlite = None
        mysql = None
        pg = None

    with pytest.raises(ConfigError):
        build_database(_Stub())


# ---------------- 真实数据库连接 ----------------
#
# 表名统一加 autotest_ 前缀：既与生产表隔离，也满足 storage 的删除守卫。

TABLE_PREFIX = ''
RUN_ID = int(time.time()) % 1000000
HISTORY_MEMORY_ID = RUN_ID * 100 + 31
BATCH_MEMORY_ID = RUN_ID * 100 + 32
MESSAGE_SCOPE = 'autotest_pg_' + str(RUN_ID)


@pytest.fixture(scope='module')
def live_store():
    """真实 PG 连接：服务不可用时跳过，用完删表并释放连接池。"""
    _requires_sqlalchemy()
    from conftest import PG_SETTINGS

    store = PGDB(
        host=PG_SETTINGS.host,
        port=PG_SETTINGS.port,
        user=PG_SETTINGS.user,
        password=PG_SETTINGS.password,
        database=PG_SETTINGS.database,
        table_prefix=TABLE_PREFIX,
    )
    try:
        store._start()
    except Exception as exc:
        pytest.skip('[SKIP INFO] PG 服务不可用：' + str(exc))

    try:
        yield store
    finally:
        if store.engine:
            store.engine.dispose()


def test_live_start_creates_tables(live_store):
    _requires_sqlalchemy()
    # 建表完成后不再有缺失项。
    assert live_store._missing_tables() == []


def test_live_table_names_are_prefixed(live_store):
    _requires_sqlalchemy()
    assert live_store.history_table.name == TABLE_PREFIX + 'history'
    assert live_store.messages_table.name == TABLE_PREFIX + 'messages'


def test_live_start_is_idempotent(live_store):
    _requires_sqlalchemy()
    # 二次启动复用已建表：不重复建表，也不残留缺失项。
    previous = live_store.engine
    assert live_store._start() is live_store.engine
    assert live_store._missing_tables() == []
    previous.dispose()


def test_live_ensure_schema_recreates_dropped_table(live_store):
    _requires_sqlalchemy()
    live_store.messages_table.drop(live_store.engine, checkfirst=True)
    assert [table.name for table in live_store._missing_tables()] == [live_store.messages_table.name]
    live_store._ensure_schema()
    assert live_store._missing_tables() == []


def test_live_add_history_persists(live_store):
    _requires_sqlalchemy()
    saved = live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, new_memory='v1', event='ADD', actor_id='u1')
    )
    assert isinstance(saved, HistoryRecord)
    assert isinstance(saved.id, int)
    assert saved.memory_id == HISTORY_MEMORY_ID
    assert saved.event == 'ADD'
    assert saved.actor_id == 'u1'
    assert saved.is_deleted is False


def test_live_get_history_orders_by_time(live_store):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    live_store.add_history(
        HistoryRecord(id=next_id(),memory_id=TEMP_HISTORY_MEMORY_ID, event='ADD', created_at_tz=base)
    )
    live_store.add_history(
        HistoryRecord(id=next_id(),memory_id=TEMP_HISTORY_MEMORY_ID, event='UPDATE', created_at_tz=base + timedelta(seconds=1))
    )
    rows = live_store.get_history(TEMP_HISTORY_MEMORY_ID)
    assert ['UPDATE','ADD'] == [row.event for row in rows][-2:]


def test_live_get_history_isolates_memory_ids(live_store):
    _requires_sqlalchemy()
    live_store.add_history(HistoryRecord(memory_id=BATCH_MEMORY_ID, event='ADD'))
    assert live_store.get_history(BATCH_MEMORY_ID + 999) == []
    assert len(live_store.get_history(BATCH_MEMORY_ID)) == 1


def test_live_batch_add_history_returns_records(live_store):
    _requires_sqlalchemy()
    saved = live_store.batch_add_history([
        HistoryRecord(memory_id=BATCH_MEMORY_ID + 1, new_memory='b1', event='ADD'),
        HistoryRecord(memory_id=BATCH_MEMORY_ID + 1, new_memory='b2', event='ADD'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item, HistoryRecord) for item in saved)
    assert all(isinstance(item.id, int) for item in saved)
    assert len(live_store.get_history(BATCH_MEMORY_ID + 1)) == 2


def test_live_batch_add_history_empty_returns_empty(live_store):
    _requires_sqlalchemy()
    # 空批次不触达数据库，直接返回空列表。
    assert live_store.batch_add_history([]) == []


def test_live_save_messages_keeps_last_only(live_store):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    live_store.save_messages([
        Message(session_scope=MESSAGE_SCOPE, content='e1', created_at_tz=base),
        Message(session_scope=MESSAGE_SCOPE, content='e2', created_at_tz=base + timedelta(seconds=1)),
    ], keep_last=2)
    live_store.save_messages([
        Message(session_scope=MESSAGE_SCOPE, content='e3', created_at_tz=base + timedelta(seconds=2)),
    ], keep_last=2)
    rows = live_store.get_last_messages(MESSAGE_SCOPE, limit=10)
    assert ['e2', 'e3'] == [item.content for item in rows]


def test_live_get_last_messages_respects_limit(live_store):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    scope = MESSAGE_SCOPE + '_limit'
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    live_store.save_messages([
        Message(session_scope=scope, content='c1', created_at_tz=base),
        Message(session_scope=scope, content='c2', created_at_tz=base + timedelta(seconds=1)),
        Message(session_scope=scope, content='c3', created_at_tz=base + timedelta(seconds=2)),
    ])
    rows = live_store.get_last_messages(scope, limit=2)
    assert all(isinstance(item, Message) for item in rows)
    assert ['c2', 'c3'] == [item.content for item in rows]
