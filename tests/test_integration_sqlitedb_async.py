# AsyncSQLiteDB 单元测试：驱动探测、路径校验、连接串构造、表前缀与异步 CRUD。
#
# 引擎装配用例用 monkeypatch 隔离驱动与引擎，异步 CRUD 用例在临时目录建真实库文件；
# sqlalchemy 或 aiosqlite 缺失时相应用例跳过。
# 同步用例见 test_integration_sqlitedb.py，本文件只覆盖异步类。

import logging
import random
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio


try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from rag_data.config import DBSettings
    from rag_data.exceptions import ConfigError, OptionalDependencyError, StorageError
    from rag_data.storage.db import sqlite_db as sqlite_module, next_id
    from rag_data.storage.db.base import BaseDB
    from rag_data.storage.db.db_engine import AsyncDBEngine
    from rag_data.storage.db.registry_db import available_dbs, is_db_registered, resolve_db
    from rag_data.storage.db.schema import HistoryRecord, Message, get_history_table, get_messages_table
    from rag_data.storage.db.sqlite_db import AsyncSQLiteDB

    _SQLALCHEMY_AVAILABLE = True
    _SQLALCHEMY_ERROR = None
except ImportError as _exc:
    _SQLALCHEMY_AVAILABLE = False
    _SQLALCHEMY_ERROR = _exc


pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）',
                allow_module_level=True)


def _requires_sqlalchemy() -> None:
    """缺 sqlalchemy 时跳过：引擎装配与行转换都依赖它。"""
    if not _SQLALCHEMY_AVAILABLE:
        pytest.skip("未安装 sqlalchemy，请执行 pip install sqlalchemy")


def _requires_async_driver() -> None:
    """缺 aiosqlite 时跳过：异步 SQLite 必须有该驱动。"""
    _requires_sqlalchemy()
    if not sqlite_module._ASYNC_SQLITE_DRIVER:
        pytest.skip("未安装 aiosqlite，请执行 pip install aiosqlite")


class _FakeEngine:
    # 假异步引擎：只记录构造参数，不建立真实连接。

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs


def _patch_async_engine(monkeypatch, captured):
    # 用记录型工厂替换 create_async_engine 与 async_sessionmaker。

    def engine_factory(url, **kwargs):
        captured['url'] = url
        captured['kwargs'] = kwargs
        return _FakeEngine(url, **kwargs)

    def session_factory(bind=None, **kwargs):
        captured['session'] = dict(bind=bind, **kwargs)
        return dict(bind=bind, **kwargs)

    monkeypatch.setattr(sqlite_module, 'create_async_engine', engine_factory)
    monkeypatch.setattr(sqlite_module, 'async_sessionmaker', session_factory)


def _patch_driver(monkeypatch):
    # 固定异步驱动常量，避免受本机是否安装 aiosqlite 影响。
    monkeypatch.setattr(sqlite_module, '_ASYNC_SQLITE_DRIVER', 'aiosqlite')


@pytest_asyncio.fixture
async def started(monkeypatch, tmp_path):
    # 构造并启动 AsyncSQLiteDB：引擎、会话工厂与建表全部替换。
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    init_calls = []

    async def _mock_init_db(self, engine, session_local):
        init_calls.append(engine)
        self.engine = engine
        self.AsyncSessionLocal = session_local

    monkeypatch.setattr(AsyncSQLiteDB, '_init_db', _mock_init_db)
    store = AsyncSQLiteDB(db_path=str(tmp_path / 'rag.db'))
    await store._start()
    return store, captured, init_calls


# ---------------- 注册与继承 ----------------


def test_async_backend_name_is_sqlite_async():
    _requires_sqlalchemy()
    assert AsyncSQLiteDB.backend == 'sqlite_async'


def test_async_inherits_async_engine():
    _requires_sqlalchemy()
    assert issubclass(AsyncSQLiteDB, AsyncDBEngine)
    assert issubclass(AsyncSQLiteDB, BaseDB)


def test_async_registered_as_backend():
    _requires_sqlalchemy()
    assert is_db_registered('sqlite_async')
    assert resolve_db('sqlite_async') is AsyncSQLiteDB
    assert 'sqlite_async' in available_dbs()


def test_async_implements_all_abstract_methods():
    # 异步后端必须自行实现 _start 与 _missing_tables，才可直接实例化。
    _requires_sqlalchemy()
    assert AsyncSQLiteDB.__abstractmethods__ == frozenset()


def test_missing_tables_is_implemented_here():
    # 异步实现自行提供 _missing_tables：SQLite 没有 information_schema，走通用 inspect。
    _requires_sqlalchemy()
    assert '_missing_tables' in AsyncSQLiteDB.__dict__


# ---------------- 构造行为 ----------------


def test_async_constructor_keeps_parameters():
    _requires_sqlalchemy()
    store = AsyncSQLiteDB(db_path='data/rag.db', print_sql=True)
    assert store.db_path == 'data/rag.db'
    assert store.print_sql is True
    assert store.connection_string is None


def test_async_constructor_is_lazy():
    _requires_sqlalchemy()
    store = AsyncSQLiteDB(db_path='data/rag.db')
    assert store.engine is None
    assert store.AsyncSessionLocal is None


def test_async_constructor_does_not_create_engine(monkeypatch):
    _requires_sqlalchemy()

    # 构造期只保存参数，create_async_engine 不应被调用。

    def boom(url, **kwargs):
        raise AssertionError('构造期不应创建引擎')

    monkeypatch.setattr(sqlite_module, 'create_async_engine', boom)
    AsyncSQLiteDB(db_path='data/rag.db')


def test_async_constructor_builds_default_tables():
    _requires_sqlalchemy()
    # 基类按空前缀建出两张表：AsyncSQLiteDB 未传 table_prefix。
    store = AsyncSQLiteDB(db_path='rag.db')
    assert store.table_prefix == ''
    assert store.history_table.name == 'history'
    assert store.messages_table.name == 'messages'
    assert {table.name for table in store.METADATA.sorted_tables} == {'history', 'messages'}


def test_async_table_factories_apply_prefix():
    _requires_sqlalchemy()
    from sqlalchemy import MetaData

    metadata = MetaData()
    assert get_history_table(metadata, 'autotest_').name == 'autotest_history'
    assert get_messages_table(metadata, 'autotest_').name == 'autotest_messages'


# ---------------- 启动：驱动与连接串 ----------------


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_requires_driver(monkeypatch):
    _requires_sqlalchemy()
    monkeypatch.setattr(sqlite_module, '_ASYNC_SQLITE_DRIVER', '')
    monkeypatch.setattr(sqlite_module, '_ASYNC_SQLITE_ERROR', ImportError('missing'))
    with pytest.raises(OptionalDependencyError):
        await AsyncSQLiteDB(db_path='rag.db')._start()


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_requires_db_path(monkeypatch):
    _requires_sqlalchemy()
    _patch_driver(monkeypatch)
    with pytest.raises(ConfigError):
        await AsyncSQLiteDB(db_path='')._start()


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_builds_url(started):
    _requires_sqlalchemy()
    _, captured, _ = started
    url = captured['url']
    assert 'sqlite+aiosqlite' in str(url)
    assert 'rag.db' in str(url)


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_creates_parent_directory(monkeypatch, tmp_path):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)

    async def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.AsyncSessionLocal = session_local

    monkeypatch.setattr(AsyncSQLiteDB, '_init_db', _mock_init_db)
    target = tmp_path / 'nested' / 'deep' / 'rag.db'
    await AsyncSQLiteDB(db_path=str(target))._start()
    assert target.parent.is_dir()


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_assigns_engine_and_session_factory(started):
    _requires_sqlalchemy()
    store, captured, init_calls = started
    assert isinstance(store.engine, _FakeEngine)
    assert init_calls == [store.engine]
    assert captured['session']['bind'] is store.engine


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_session_factory_expires_off(started):
    _requires_sqlalchemy()
    _, captured, _ = started
    options = captured['session']
    assert options['class_'] is AsyncSession
    assert options['expire_on_commit'] is False


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_echo_follows_print_sql(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)

    async def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.AsyncSessionLocal = session_local

    monkeypatch.setattr(AsyncSQLiteDB, '_init_db', _mock_init_db)
    await AsyncSQLiteDB(db_path='rag.db', print_sql=True)._start()
    assert captured['kwargs']['echo'] is True


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_keeps_session_local(started):
    _requires_sqlalchemy()
    # _start 保留小写 session_local 供调试查看，实际会话走 AsyncSessionLocal。
    store, captured, _ = started
    assert store.session_local.get('class_') == captured['session']['class_']


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_enables_pool_logging_when_print_sql(monkeypatch):
    _requires_sqlalchemy()
    captured = {}
    _patch_async_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)

    async def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.AsyncSessionLocal = session_local

    monkeypatch.setattr(AsyncSQLiteDB, '_init_db', _mock_init_db)
    pool_logger = logging.getLogger('sqlalchemy.pool')
    previous = pool_logger.level
    try:
        await AsyncSQLiteDB(db_path='rag.db', print_sql=True)._start()
        assert pool_logger.level == logging.INFO
    finally:
        pool_logger.setLevel(previous)


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_returns_engine(started):
    _requires_sqlalchemy()
    store, _, _ = started
    engine = await store._start()
    assert engine is store.engine


@pytest.mark.asyncio(loop_scope='module')
async def test_async_start_wraps_engine_error(monkeypatch):
    _requires_sqlalchemy()
    _patch_driver(monkeypatch)

    def boom(url, **kwargs):
        raise RuntimeError('no driver')

    monkeypatch.setattr(sqlite_module, 'create_async_engine', boom)
    with pytest.raises(StorageError):
        await AsyncSQLiteDB(db_path='rag.db')._start()


# ---------------- 真实库文件（缺 aiosqlite 时跳过） ----------------
TABLE_PREFIX = "async_"


@pytest_asyncio.fixture(scope="module",loop_scope="module")
async def live_db():
    """真实库文件：缺驱动时跳过，用完释放连接池。"""

    from conftest import SQLITE_SETTINGS

    _requires_async_driver()

    DB_PATH = Path(SQLITE_SETTINGS.path)
    db_file = (DB_PATH / 'live.db').resolve()

    store = AsyncSQLiteDB(db_path=str(db_file), table_prefix=TABLE_PREFIX)
    await store._start()
    try:
        yield store
    finally:
        if store.engine:
            await store.engine.dispose()


@pytest.mark.asyncio(loop_scope='module')
async def test_live_start_creates_tables(live_db):
    _requires_async_driver()
    # 建表完成后不再有缺失项。
    assert await live_db._missing_tables() == []


@pytest.mark.asyncio(loop_scope='module')
async def test_live_table_names(live_db):
    _requires_async_driver()
    # 真实库里的表名与声明一致，未加前缀。
    assert live_db.history_table.name == TABLE_PREFIX + 'history'
    assert live_db.messages_table.name == TABLE_PREFIX + 'messages'


@pytest.mark.asyncio(loop_scope='module')
async def test_live_start_is_idempotent(live_db):
    _requires_async_driver()
    # 二次启动复用已建表：不重复建表，也不残留缺失项。
    previous = live_db.engine
    assert await live_db._start() is live_db.engine
    assert await live_db._missing_tables() == []


@pytest.mark.asyncio(loop_scope='module')
async def test_live_add_history_persists(live_db):
    _requires_async_driver()

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    saved = await live_db.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='v1', event='ADD', actor_id='u1')
    )
    assert isinstance(saved, HistoryRecord)
    assert isinstance(saved.id, int)
    assert saved.memory_id == TEMP_HISTORY_MEMORY_ID
    assert saved.event == 'ADD'
    assert saved.actor_id == 'u1'
    assert saved.is_deleted is False


@pytest.mark.asyncio(loop_scope='module')
async def test_live_get_history_orders_by_time(live_db):
    _requires_async_driver()
    from datetime import datetime, timedelta, timezone

    base = datetime.now(timezone.utc)
    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    await live_db.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='v1', event='ADD',
                      created_at_tz=base)
    )
    await live_db.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, old_memory='v1', new_memory='v2', event='UPDATE',
                      created_at_tz=base + timedelta(seconds=1))
    )
    rows = await live_db.get_history(TEMP_HISTORY_MEMORY_ID)
    assert [row.event for row in rows] == ['UPDATE', 'ADD']
    assert all(isinstance(row, HistoryRecord) for row in rows)


@pytest.mark.asyncio(loop_scope='module')
async def test_live_get_history_isolates_memory_ids(live_db):
    _requires_async_driver()

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    await live_db.add_history(HistoryRecord(memory_id=TEMP_HISTORY_MEMORY_ID, event='ADD'))
    assert await live_db.get_history(14) == []
    assert len(await live_db.get_history(TEMP_HISTORY_MEMORY_ID)) == 1


@pytest.mark.asyncio(loop_scope='module')
async def test_live_batch_add_history_returns_records(live_db):
    _requires_async_driver()

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    saved = await live_db.batch_add_history([
        HistoryRecord(memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='b1', event='ADD'),
        HistoryRecord(memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='b2', event='ADD'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item.id, int) for item in saved)
    assert len(await live_db.get_history(TEMP_HISTORY_MEMORY_ID)) == 2


@pytest.mark.asyncio(loop_scope='module')
async def test_live_batch_add_history_empty_returns_empty(live_db):
    _requires_async_driver()
    # 空批次不触达数据库，直接返回空列表。
    assert await live_db.batch_add_history([]) == []


@pytest.mark.asyncio(loop_scope='module')
async def test_live_save_messages_keeps_last_only(live_db):
    _requires_async_driver()
    from datetime import datetime, timedelta, timezone

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    session_cope = 's1_' + str(TEMP_HISTORY_MEMORY_ID)
    base = datetime.now(tz=timezone.utc)

    await live_db.save_messages([
        Message(id=next_id(), session_scope=session_cope, content='e1', created_at_tz=base),
        Message(id=next_id(), session_scope=session_cope, content='e2', created_at_tz=base + timedelta(seconds=1)),
    ], keep_last=2)
    await live_db.save_messages([
        Message(id=next_id(), session_scope=session_cope, content='e3', created_at_tz=base + timedelta(seconds=2)),
    ], keep_last=2)
    rows = await live_db.get_last_messages(session_cope, limit=10)
    assert [item.content for item in rows] == ['e2', 'e3']


@pytest.mark.asyncio(loop_scope='module')
async def test_live_get_last_messages_respects_limit(live_db):
    _requires_async_driver()
    from datetime import datetime, timedelta, timezone

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    session_cope = 's2_' + str(TEMP_HISTORY_MEMORY_ID)
    base = datetime.now(tz=timezone.utc)

    await live_db.save_messages([
        Message(id=next_id(), session_scope=session_cope, content='c1', created_at_tz=base),
        Message(id=next_id(), session_scope=session_cope, content='c2', created_at_tz=base + timedelta(seconds=1)),
        Message(id=next_id(), session_scope=session_cope, content='c3', created_at_tz=base + timedelta(seconds=2)),
    ])
    rows = await live_db.get_last_messages(session_cope, limit=2)
    assert all(isinstance(item, Message) for item in rows)
    assert [item.content for item in rows] == ['c2', 'c3']
