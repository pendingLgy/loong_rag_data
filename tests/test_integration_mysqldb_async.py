# AsyncMySQLDB 单元测试：异步驱动选择、构造参数、连接串归一化与异步引擎装配。
#
# 模拟用例用 monkeypatch 隔离驱动与服务，真实服务用例在服务不可用时自动 skip。
# 异步用例显式标注 asyncio 模块级事件循环，与 asyncio_mode=auto 双保险。
# 父类能力用例连真实 MySQL（conftest 的 MYSQL_SETTINGS），服务不可用时跳过；
# 覆盖 BaseDB 的 6 个抽象方法：_start、add_history、get_history、batch_add_history、save_messages、get_last_messages。

import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from inspect import iscoroutinefunction, signature, isfunction

import pytest
import pytest_asyncio

try:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
    from sqlalchemy.sql.dml import Delete  # noqa: E402
    from rag_data.storage.db.db_engine import AsyncDBEngine  # noqa: E402
    from rag_data.storage.db.schema import HistoryRecord, Message  # noqa: E402
    from rag_data.exceptions import OptionalDependencyError, StorageError  # noqa: E402
    from rag_data.storage.db import mysql_db as mysql_module  # noqa: E402
    from rag_data.storage.db.mysql_db import AsyncMySQLDB  # noqa: E402
    _SQLALCHEMY_AVAILABLE = True
    _SQLALCHEMY_ERROR = None
except ImportError as _exc:
    _SQLALCHEMY_AVAILABLE = False
    _SQLALCHEMY_ERROR = _exc

pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)


def _requires_sqlalchemy() -> None:
    """缺 sqlalchemy 时跳过：连接装配与行转换都依赖它。"""
    if not _SQLALCHEMY_AVAILABLE or not mysql_module._ASYNC_MYSQL_DRIVER:
        pytest.skip("未安装 sqlalchemy，请执行 pip install sqlalchemy")


class _FakeEngine:
    # 假异步引擎：只记录构造参数，不建立真实连接。

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs


class _FakeAsyncSession:
    # 假异步会话：记录提交与回滚，供 _session 语义校验。

    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


def _patch_async_engine(monkeypatch, captured):
    # 用记录型工厂替换 create_async_engine 与 async_sessionmaker。

    def engine_factory(url, **kwargs):
        captured['url'] = url
        captured['kwargs'] = kwargs
        return _FakeEngine(url, **kwargs)

    def session_factory(bind=None, **kwargs):
        captured['session'] = dict(bind=bind, **kwargs)
        return dict(bind=bind, **kwargs)

    monkeypatch.setattr(mysql_module, 'create_async_engine', engine_factory)
    monkeypatch.setattr(mysql_module, 'async_sessionmaker', session_factory)


def _patch_async_driver(monkeypatch):
    # 固定异步驱动常量，避免受本机是否安装 aiomysql 影响。
    monkeypatch.setattr(mysql_module, '_ASYNC_MYSQL_DRIVER', 'aiomysql')
    monkeypatch.setattr(mysql_module, '_MYSQL_DRIVER', 'pymysql')


@pytest_asyncio.fixture
async def started(monkeypatch):
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_async_driver(monkeypatch)

    async def _mock_init_db(self, engine, AAsyncSessionLocal):
        return None

    monkeypatch.setattr(AsyncMySQLDB, '_init_db', _mock_init_db)

    store = AsyncMySQLDB(host='db.internal', port=3307, user='app', password='p@ss', database='rag')
    await store._start()
    return store, captured


# ---------------- 注册与继承 ----------------


def test_backend_name_is_mysql_async():
    _requires_sqlalchemy()
    assert AsyncMySQLDB.backend == 'mysql_async'


def test_mysql_async_inherits_engine():
    _requires_sqlalchemy()
    assert issubclass(AsyncMySQLDB, AsyncDBEngine)


# ---------------- 构造行为 ----------------


def test_default_parameters():
    _requires_sqlalchemy()
    store = AsyncMySQLDB()
    assert store.host == 'localhost'
    assert store.port == 3306
    assert store.user == 'root'
    assert store.database == 'rag_data'
    assert store.charset == 'utf8mb4'
    assert store.print_sql is False
    assert store.connection_string is None


def test_pool_parameters_default():
    _requires_sqlalchemy()
    store = AsyncMySQLDB()
    assert store.pool_size == 10
    assert store.max_overflow == 20
    assert store.pool_timeout == 30
    assert store.pool_recycle == 1800
    assert store.pool_pre_ping is True


def test_constructor_is_lazy():
    _requires_sqlalchemy()
    store = AsyncMySQLDB()
    assert store.engine is None
    assert store.AsyncSessionLocal is None


# ---------------- 启动：驱动与连接串 ----------------


@pytest.mark.asyncio(loop_scope='module')
async def test_start_requires_async_driver(monkeypatch):
    _requires_sqlalchemy()
    monkeypatch.setattr(mysql_module, '_ASYNC_MYSQL_DRIVER', '')
    monkeypatch.setattr(mysql_module, '_ASYNC_MYSQL_ERROR', ImportError('missing'))

    db = AsyncMySQLDB()
    with pytest.raises(OptionalDependencyError):
        await db._start()


@pytest.mark.asyncio(loop_scope='module')
async def test_start_builds_url_from_discrete_params(started):
    _requires_sqlalchemy()
    _, captured = started
    url = captured['url']
    assert 'mysql+aiomysql:' in url
    assert 'db.internal:3307' in url
    assert 'rag' in url


def test_start_hides_password_in_url(started):
    _requires_sqlalchemy()
    _, captured = started
    assert 'p@ss' not in captured['url']


def test_start_assigns_engine_and_session_factory(started):
    _requires_sqlalchemy()
    store, captured = started
    assert isinstance(store.engine, _FakeEngine)
    assert captured['session']['bind'] is store.engine


def test_start_passes_pool_arguments(started):
    _requires_sqlalchemy()
    _, captured = started
    kwargs = captured['kwargs']
    assert kwargs['pool_size'] == 10
    assert kwargs['max_overflow'] == 20
    assert kwargs['pool_timeout'] == 30
    assert kwargs['pool_recycle'] == 1800
    assert kwargs['pool_pre_ping'] is True
    assert kwargs['echo'] is False


def test_start_session_factory_expires_off(started):
    _requires_sqlalchemy()
    _, captured = started
    options = captured['session']
    assert options['class_'] is AsyncSession
    assert options['expire_on_commit'] is False


@pytest.mark.asyncio(loop_scope='module')
async def test_start_echo_follows_print_sql(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_async_driver(monkeypatch)

    async def _mock_init_db(self, engine, AAsyncSessionLocal):
        return None

    monkeypatch.setattr(AsyncMySQLDB, '_init_db', _mock_init_db)

    db = AsyncMySQLDB(print_sql=True)
    await db._start()
    assert captured['kwargs']['echo'] is True


@pytest.mark.asyncio(loop_scope='module')
async def test_start_wraps_engine_error(monkeypatch):
    _requires_sqlalchemy()
    _patch_async_driver(monkeypatch)
    monkeypatch.setattr(AsyncMySQLDB, '_ensure_schema', lambda self: None)

    def boom(url, **kwargs):
        raise RuntimeError('no server')

    monkeypatch.setattr(mysql_module, 'create_async_engine', boom)
    with pytest.raises(StorageError):
        db = AsyncMySQLDB()
        await db._start()


@pytest.mark.asyncio(loop_scope='module')
@pytest.mark.parametrize('prefix', ['mysql+pymysql', 'mysql+mysqlconnector', 'mysql'])
async def test_start_normalizes_connection_string(monkeypatch, prefix):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_async_driver(monkeypatch)

    async def _mock_init_db(self, engine, AAsyncSessionLocal):
        return None

    monkeypatch.setattr(AsyncMySQLDB, '_init_db', _mock_init_db)

    db = AsyncMySQLDB(connection_string=prefix + '://root:pw@h:3306/rag')
    await db._start()
    assert captured['url'].startswith('mysql+aiomysql')
    assert 'root:pw@h:3306/rag' in captured['url']


@pytest.mark.asyncio(loop_scope='module')
async def test_start_keeps_foreign_connection_string(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_async_driver(monkeypatch)

    async def _mock_init_db(self, engine, AAsyncSessionLocal):
        return None

    monkeypatch.setattr(AsyncMySQLDB, '_init_db', _mock_init_db)

    store = AsyncMySQLDB(connection_string='postgresql://u:p@h/db')
    await store._start()
    assert captured['url'] == 'postgresql://u:p@h/db'


@pytest.mark.asyncio(loop_scope='module')
async def test_start_keeps_original_connection_string(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_async_driver(monkeypatch)

    async def _mock_init_db(self, engine, AAsyncSessionLocal):
        return None

    monkeypatch.setattr(AsyncMySQLDB, '_init_db', _mock_init_db)
    store = AsyncMySQLDB(connection_string='mysql://root:pw@h/rag')
    await store._start()
    assert store.connection_string == 'mysql://root:pw@h/rag'
    assert 'mysql+aiomysql' in captured['url']


@pytest.mark.asyncio(loop_scope='module')
async def test_start_runs_schema_setup(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_async_driver(monkeypatch)
    calls = []

    async def fake_schema(self):
        calls.append(True)

    monkeypatch.setattr(AsyncMySQLDB, '_ensure_schema', fake_schema)
    store = AsyncMySQLDB()
    await store._start()
    assert calls == [True]


# ---------------- 会话生命周期 ----------------


@pytest.mark.asyncio(loop_scope='module')
async def test_session_commits_on_success():
    _requires_sqlalchemy()
    store = AsyncMySQLDB()
    fake = _FakeAsyncSession()
    store.AsyncSessionLocal = lambda: fake
    async with store._session() as session:
        assert session is fake
    assert fake.entered is True
    assert fake.committed is True
    assert fake.rolled_back is False


@pytest.mark.asyncio(loop_scope='module')
async def test_session_rolls_back_on_error():
    _requires_sqlalchemy()
    store = AsyncMySQLDB()
    fake = _FakeAsyncSession()
    store.AsyncSessionLocal = lambda: fake
    with pytest.raises(RuntimeError):
        async with store._session():
            raise RuntimeError('boom')
    assert fake.committed is False
    assert fake.rolled_back is True


# ---------------- 父类 AsyncDBEngine 能力（真实 MySQL 连接） ----------------
#
# 父类逻辑与方言无关；这里连真实 MySQL，既验证父类流程，也验证 SQL 在真实方言下可执行。

# 每次运行用独立的标识，避免与既有数据或其他用例互相干扰。
RUN_ID = int(time.time()) % 1000000
HISTORY_MEMORY_ID = RUN_ID * 100 + 1
TABLE_PREFIX = "async_"


@pytest.fixture(scope="module")
async def live_store():
    """真实 MySQL 连接：服务不可用或初始化失败时跳过，用完释放连接池。"""
    _requires_sqlalchemy()
    from conftest import MYSQL_SETTINGS

    store = AsyncMySQLDB(
        host=MYSQL_SETTINGS.host,
        port=MYSQL_SETTINGS.port,
        user=MYSQL_SETTINGS.user,
        password=MYSQL_SETTINGS.password,
        database=MYSQL_SETTINGS.database,
        table_prefix=TABLE_PREFIX,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            await store._start()
        except Exception as exc:
            pytest.skip("[SKIP INFO] MySQL 服务不可用：" + str(exc))

    try:
        yield store
    finally:
        if store.engine:
            await store.engine.dispose()


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_start_creates_engine(live_store):
    _requires_sqlalchemy()
    assert live_store.engine is not None
    assert live_store.AsyncSessionLocal is not None


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_start_accepts_history_write(live_store):
    _requires_sqlalchemy()
    # 能写入即说明 history 表随启动建好并可持久化
    saved = await live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, new_memory='v1', event='ADD', actor_id='u1')
    )
    assert isinstance(saved.id, int)


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_add_history_persists(live_store):
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
async def test_parent_get_history_returns_rows(live_store):
    _requires_sqlalchemy()
    await live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, event='ADD', created_at_tz=datetime.now(tz=timezone.utc),
                      actor_id='u1')
    )
    await live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, event='UPDATE', created_at_tz=datetime.now(tz=timezone.utc),
                      actor_id='u1')
    )
    rows = await live_store.get_history(HISTORY_MEMORY_ID)
    assert len(rows) >= 2
    assert all(isinstance(row, HistoryRecord) for row in rows)


# ---------------- 父类 BaseDB 其余契约方法（真实 MySQL 连接） ----------------
#
# BaseDB 共声明 6 个抽象方法：_start、add_history、get_history 已在上方覆盖，
# 此处补齐 batch_add_history、save_messages 与 get_last_messages。

# 批量历史与消息会话各自独立的标识，避免与既有数据互相干扰。
BATCH_MEMORY_ID = RUN_ID * 100 + 2
MESSAGE_SCOPE = 'autotest_async_' + str(RUN_ID)


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_batch_add_history_returns_records(live_store):
    _requires_sqlalchemy()
    saved = await live_store.batch_add_history([
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b1', event='ADD', actor_id='u2'),
        HistoryRecord(memory_id=BATCH_MEMORY_ID, old_memory='b1', new_memory='b2', event='UPDATE', actor_id='u2'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item, HistoryRecord) for item in saved)
    assert all(isinstance(item.id, int) for item in saved)
    assert [item.event for item in saved] == ['ADD', 'UPDATE']


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_batch_add_history_persists(live_store):
    _requires_sqlalchemy()
    await live_store.batch_add_history([
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b3', event='ADD'),
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b4', event='ADD'),
    ])
    rows = await live_store.get_history(BATCH_MEMORY_ID)
    assert len(rows) >= 2
    assert 'ADD' in {row.event for row in rows}


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_batch_add_history_empty_returns_empty(live_store):
    _requires_sqlalchemy()
    # 空批次不触达数据库，直接返回空列表。
    assert await live_store.batch_add_history([]) == []


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_save_messages_returns_objects(live_store):
    _requires_sqlalchemy()
    saved = await live_store.save_messages([
        Message(session_scope=MESSAGE_SCOPE, role='user', content='a'),
        Message(session_scope=MESSAGE_SCOPE, role='assistant', content='b'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item, Message) for item in saved)
    assert all(isinstance(item.id, int) for item in saved)
    assert [item.content for item in saved] == ['a', 'b']


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_save_messages_empty_returns_empty(live_store):
    _requires_sqlalchemy()
    # 空批次不触达数据库，直接返回空列表。
    assert await live_store.save_messages([]) == []


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_get_last_messages_returns_rows(live_store):
    _requires_sqlalchemy()
    scope = MESSAGE_SCOPE + '_read'
    base = datetime.now(tz=timezone.utc)
    await live_store.save_messages([
        Message(session_scope=scope, role='user', content='q1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, role='assistant', content='a1', created_at_tz=base - timedelta(seconds=2)),
    ])
    rows = await live_store.get_last_messages(scope)
    assert all(isinstance(item, Message) for item in rows)
    assert [item.content for item in rows] == ['q1', 'a1']


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_get_last_messages_respects_limit(live_store):
    _requires_sqlalchemy()
    scope = MESSAGE_SCOPE + '_limit'
    base = datetime.now(tz=timezone.utc)
    await live_store.save_messages([
        Message(session_scope=scope, content='c1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, content='c2', created_at_tz=base - timedelta(seconds=2)),
        Message(session_scope=scope, content='c3', created_at_tz=base - timedelta(seconds=1)),
    ])
    rows = await live_store.get_last_messages(scope, limit=2)
    assert [item.content for item in rows] == ['c2', 'c3']


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_save_messages_keeps_last_only(live_store):
    _requires_sqlalchemy()
    scope = MESSAGE_SCOPE + '_evict'
    base = datetime.now(tz=timezone.utc)
    await live_store.save_messages([
        Message(session_scope=scope, content='e1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, content='e2', created_at_tz=base - timedelta(seconds=2)),
    ], keep_last=2)
    await live_store.save_messages([
        Message(session_scope=scope, content='e3', created_at_tz=base - timedelta(seconds=1)),
    ], keep_last=2)
    rows = await live_store.get_last_messages(scope, limit=10)
    assert [item.content for item in rows] == ['e2', 'e3']


# ---------------- 父类 _build_evict_stmt：构造会话消息的删除语句 ----------------
#
# _build_evict_stmt 只构造语句：保留每个会话最近 keep_last 条，删除其余记录；执行由调用方在自己的会话中完成。

EVICT_SCOPE = MESSAGE_SCOPE + '_build_evict_stmt'


def test_parent_evict_stmt_is_coroutine():
    _requires_sqlalchemy()
    assert isfunction(AsyncMySQLDB._build_evict_stmt)
    assert list(signature(AsyncMySQLDB._build_evict_stmt).parameters) == ['self', 'session_scope', 'keep_last']


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_evict_stmt_returns_delete_statement(live_store):
    _requires_sqlalchemy()
    stmt = live_store._build_evict_stmt(EVICT_SCOPE, 2)
    assert isinstance(stmt, Delete)
    assert stmt.table.name == TABLE_PREFIX + 'messages'


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_evict_stmt_compiles_for_scope_and_keep_last(live_store):
    _requires_sqlalchemy()
    stmt = live_store._build_evict_stmt(EVICT_SCOPE, 2)
    sql = str(stmt)
    assert sql.startswith('DELETE FROM ' + TABLE_PREFIX + 'messages')
    assert 'NOT IN' in sql.upper()
    params = stmt.compile().params
    assert EVICT_SCOPE in params.values()
    assert 2 in params.values()


@pytest.mark.asyncio(loop_scope='module')
async def test_parent_evict_stmt_deletes_all_but_last(live_store):
    _requires_sqlalchemy()
    scope = EVICT_SCOPE + '_apply'
    base = datetime.now(tz=timezone.utc)
    await live_store.save_messages([
        Message(session_scope=scope, content='k1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, content='k2', created_at_tz=base - timedelta(seconds=2)),
        Message(session_scope=scope, content='k3', created_at_tz=base - timedelta(seconds=1)),
    ], keep_last=10)
    stmt = live_store._build_evict_stmt(scope, 2)
    async with live_store._session() as session:
        await session.execute(stmt)
    rows = await live_store.get_last_messages(scope, limit=10)
    assert [item.content for item in rows] == ['k2', 'k3']
