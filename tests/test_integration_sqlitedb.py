# SQLiteDB 单元测试：路径校验、连接串构造、表前缀、建表与同步 CRUD。
#
# 引擎装配用例用 monkeypatch 隔离引擎；真实库用例在临时目录建库文件，验证建表与 CRUD。
# sqlalchemy 缺失时相应用例跳过；异步用例见 test_integration_sqlitedb_async.py。

import logging
import random
import sys
import time
from inspect import signature
from pathlib import Path

import pytest

from conftest import SQLITE_SETTINGS

try:
    from sqlalchemy import MetaData  # noqa: E402
    from rag_data.config import DBSettings  # noqa: E402
    from rag_data.exceptions import ConfigError, StorageError  # noqa: E402
    from rag_data.storage.db import sqlite_db as sqlite_module, next_id  # noqa: E402
    from rag_data.storage.db.base import BaseDB  # noqa: E402
    from rag_data.storage.db.db_engine import AsyncDBEngine, DBEngine  # noqa: E402
    from rag_data.storage.db.registry_db import available_dbs, is_db_registered, resolve_db  # noqa: E402
    from rag_data.storage.db.schema import (
        HistoryRecord,
        Message,
        get_history_table,
        get_messages_table,
    )  # noqa: E402
    from rag_data.storage.db.sqlite_db import AsyncSQLiteDB, SQLiteDB  # noqa: E402

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


class _FakeEngine:
    # 假引擎：只记录构造参数，不建立真实连接。

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs


def _patch_engine(monkeypatch, captured):
    # 用记录型工厂替换 create_engine 与 sessionmaker。

    def engine_factory(url, **kwargs):
        captured['url'] = str(url)
        captured['kwargs'] = kwargs
        return _FakeEngine(url, **kwargs)

    def session_factory(bind=None, **kwargs):
        captured['session'] = dict(bind=bind, **kwargs)
        return dict(bind=bind, **kwargs)

    monkeypatch.setattr(sqlite_module, 'create_engine', engine_factory)
    monkeypatch.setattr(sqlite_module, 'sessionmaker', session_factory)


@pytest.fixture()
def started(monkeypatch, tmp_path):
    # 构造并启动 SQLiteDB：引擎、会话工厂与建表全部替换。
    captured = {}
    _patch_engine(monkeypatch, captured)
    init_calls = []

    def _mock_init_db(self, engine, session_local):
        init_calls.append(engine)
        self.engine = engine
        self.SessionLocal = session_local

    monkeypatch.setattr(SQLiteDB, '_init_db', _mock_init_db)
    store = SQLiteDB(db_path=str(tmp_path / 'rag.db'))
    store._start()
    return store, captured, init_calls


# ---------------- 注册与继承 ----------------


def test_backend_name_is_sqlite():
    _requires_sqlalchemy()
    assert SQLiteDB.backend == 'sqlite'


def test_inherits_engine():
    _requires_sqlalchemy()
    assert issubclass(SQLiteDB, DBEngine)
    assert issubclass(SQLiteDB, BaseDB)


def test_registered_as_backend():
    _requires_sqlalchemy()
    assert is_db_registered('sqlite')
    assert resolve_db('sqlite') is SQLiteDB
    assert 'sqlite' in available_dbs()


def test_implements_all_abstract_methods():
    # DBEngine 把 _start 与 _missing_tables 都声明为抽象方法，同步实现须逐一实现。
    # 缺任意一个都会让 SQLiteDB 保持抽象、无法实例化。
    _requires_sqlalchemy()
    assert SQLiteDB.__abstractmethods__ == frozenset()


def test_missing_tables_is_implemented_here():
    # DBEngine 的 _missing_tables 是抽象方法，同步实现必须在本类中给出实现，
    # 否则 SQLiteDB 保持抽象、连实例化都会失败。
    _requires_sqlalchemy()
    assert '_missing_tables' in SQLiteDB.__dict__


# ---------------- 构造行为 ----------------


def test_constructor_keeps_parameters():
    _requires_sqlalchemy()
    store = SQLiteDB(db_path='data/rag.db', print_sql=True)
    assert store.db_path == 'data/rag.db'
    assert store.print_sql is True
    assert store.connection_string is None


def test_constructor_is_lazy():
    _requires_sqlalchemy()
    store = SQLiteDB(db_path='rag.db')
    assert store.engine is None
    assert store.SessionLocal is None


def test_constructor_does_not_create_engine(monkeypatch):
    _requires_sqlalchemy()

    # 构造期只保存参数，create_engine 不应被调用。

    def boom(url, **kwargs):
        raise AssertionError('构造期不应创建引擎')

    monkeypatch.setattr(sqlite_module, 'create_engine', boom)
    SQLiteDB(db_path='rag.db')


def test_constructor_builds_default_tables():
    _requires_sqlalchemy()
    # 基类按空前缀建出两张表：SQLiteDB 未传 table_prefix。
    store = SQLiteDB(db_path='rag.db')
    assert store.table_prefix == ''
    assert store.history_table.name == 'history'
    assert store.messages_table.name == 'messages'
    assert {table.name for table in store.METADATA.sorted_tables} == {'history', 'messages'}


def test_tables_share_declared_metadata():
    _requires_sqlalchemy()
    store = SQLiteDB(db_path='rag.db')
    assert store.history_table.metadata is store.METADATA
    assert store.messages_table.metadata is store.METADATA


# ---------------- 表名前缀（schema 工厂） ----------------


def test_history_table_default_name():
    _requires_sqlalchemy()
    assert get_history_table(MetaData()).name == 'history'


def test_messages_table_default_name():
    _requires_sqlalchemy()
    assert get_messages_table(MetaData()).name == 'messages'


def test_table_factories_apply_prefix():
    _requires_sqlalchemy()
    metadata = MetaData()
    assert get_history_table(metadata, 'autotest_').name == 'autotest_history'
    assert get_messages_table(metadata, 'autotest_').name == 'autotest_messages'
    assert {table.name for table in metadata.sorted_tables} == {'autotest_history', 'autotest_messages'}


# ---------------- 启动：路径与连接串 ----------------


def test_start_requires_db_path(monkeypatch):
    _requires_sqlalchemy()
    with pytest.raises(ConfigError):
        SQLiteDB(db_path='')._start()


def test_start_builds_url(started):
    _requires_sqlalchemy()
    _, captured, _ = started
    url = captured['url']
    assert url.startswith('sqlite:')
    assert 'rag.db' in url


def test_start_disables_thread_check(started):
    _requires_sqlalchemy()
    _, captured, _ = started
    assert captured['kwargs']['connect_args'] == {'check_same_thread': False, "timeout": 30.0}


def test_start_creates_parent_directory(monkeypatch, tmp_path):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)

    def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.SessionLocal = session_local

    monkeypatch.setattr(SQLiteDB, '_init_db', _mock_init_db)
    target = tmp_path / 'nested' / 'deep' / 'rag.db'
    SQLiteDB(db_path=str(target))._start()
    assert target.parent.is_dir()


def test_start_assigns_engine_and_session_factory(started):
    _requires_sqlalchemy()
    store, captured, init_calls = started
    assert isinstance(store.engine, _FakeEngine)
    assert init_calls == [store.engine]
    assert captured['session']['bind'] is store.engine


def test_start_echo_follows_print_sql(monkeypatch, tmp_path):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)

    def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.SessionLocal = session_local

    monkeypatch.setattr(SQLiteDB, '_init_db', _mock_init_db)
    SQLiteDB(db_path=str(tmp_path / 'rag.db'), print_sql=True)._start()
    assert captured['kwargs']['echo'] is True


def test_start_returns_engine(started):
    _requires_sqlalchemy()
    store, _, _ = started
    assert store._start() is store.engine


def test_start_wraps_schema_error(monkeypatch, tmp_path):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)

    def boom(self):
        raise RuntimeError('ddl failed')

    monkeypatch.setattr(SQLiteDB, '_missing_tables', boom)
    with pytest.raises(StorageError):
        SQLiteDB(db_path=str(tmp_path / 'rag.db'))._start()


def test_start_enables_pool_logging_when_print_sql(monkeypatch, tmp_path):
    _requires_sqlalchemy()
    captured = {}
    _patch_engine(monkeypatch, captured)

    def _mock_init_db(self, engine, session_local):
        self.engine = engine
        self.SessionLocal = session_local

    monkeypatch.setattr(SQLiteDB, '_init_db', _mock_init_db)
    pool_logger = logging.getLogger('sqlalchemy.pool')
    previous = pool_logger.level
    try:
        SQLiteDB(db_path=str(tmp_path / 'rag.db'), print_sql=True)._start()
        assert pool_logger.level == logging.INFO
    finally:
        pool_logger.setLevel(previous)


# ---------------- 真实库文件 ----------------


@pytest.fixture(scope="module")
def live_db():
    """真实库文件：在临时目录建库并建表，用完释放连接池。"""
    DB_PATH = Path(SQLITE_SETTINGS.path)
    db_file = (DB_PATH / 'live.db').resolve()
    # db_file.parent.mkdir(parents=True, exist_ok=True)
    _requires_sqlalchemy()
    store = SQLiteDB(db_path=str(db_file))
    store._start()
    try:
        yield store
    finally:
        if store.engine:
            store.engine.dispose()


def test_live_start_creates_tables(live_db):
    _requires_sqlalchemy()
    # 建表完成后不再有缺失项。
    assert live_db._missing_tables() == []


def test_live_table_names(live_db):
    _requires_sqlalchemy()
    # 真实库里的表名与声明一致，未加前缀。
    assert live_db.history_table.name == 'history'
    assert live_db.messages_table.name == 'messages'


def test_live_add_history_persists(live_db):
    _requires_sqlalchemy()
    saved = live_db.add_history(
        HistoryRecord(memory_id=21, new_memory='v1', event='ADD', actor_id='u1')
    )
    assert isinstance(saved, HistoryRecord)
    assert isinstance(saved.id, int)
    assert saved.memory_id == 21
    assert saved.event == 'ADD'
    assert saved.actor_id == 'u1'
    assert saved.is_deleted is False


def test_live_get_history_orders_by_time(live_db):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    base = datetime.now(tz=timezone.utc)
    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)
    live_db.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='v1', event='ADD', created_at_tz=base)
    )
    live_db.add_history(
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, old_memory='v1', new_memory='v2', event='UPDATE',
                      created_at_tz=base + timedelta(seconds=1))
    )
    rows = live_db.get_history(TEMP_HISTORY_MEMORY_ID)
    assert [row.event for row in rows] == ['UPDATE', 'ADD']
    assert all(isinstance(row, HistoryRecord) for row in rows)


def test_live_get_history_isolates_memory_ids(live_db):
    _requires_sqlalchemy()
    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    live_db.add_history(HistoryRecord(memory_id=TEMP_HISTORY_MEMORY_ID, event='ADD'))
    assert live_db.get_history(24) == []
    assert len(live_db.get_history(TEMP_HISTORY_MEMORY_ID)) == 1


def test_live_batch_add_history_returns_records(live_db):
    _requires_sqlalchemy()

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    saved = live_db.batch_add_history([
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='b1', event='ADD'),
        HistoryRecord(id=next_id(), memory_id=TEMP_HISTORY_MEMORY_ID, new_memory='b2', event='ADD'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item.id, int) for item in saved)
    assert len(live_db.get_history(TEMP_HISTORY_MEMORY_ID)) == 2


def test_live_batch_add_history_empty_returns_empty(live_db):
    _requires_sqlalchemy()
    # 空批次不触达数据库，直接返回空列表。
    assert live_db.batch_add_history([]) == []


def test_live_save_messages_keeps_last_only(live_db):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    TEMP_HISTORY_MEMORY_ID = (divmod(int(time.time()), 60)[1] * 1000 + 900000) * 10000 + random.randint(0, 9999)

    session_cope = 's1_' + str(TEMP_HISTORY_MEMORY_ID)
    base = datetime.now(tz=timezone.utc)
    live_db.save_messages([
        Message(id=next_id(),session_scope=session_cope, content='e1', created_at_tz=base),
        Message(id=next_id(),session_scope=session_cope, content='e2', created_at_tz=base + timedelta(seconds=1)),
    ], keep_last=2)
    live_db.save_messages([
        Message(id=next_id(),session_scope=session_cope, content='e3', created_at_tz=base + timedelta(seconds=2)),
    ], keep_last=2)
    rows = live_db.get_last_messages(session_cope, limit=10)
    assert [item.content for item in rows] == ['e2', 'e3']


def test_live_save_messages_empty_returns_empty(live_db):
    _requires_sqlalchemy()
    assert live_db.save_messages([]) == []


def test_live_get_last_messages_respects_limit(live_db):
    _requires_sqlalchemy()
    from datetime import datetime, timedelta, timezone

    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    live_db.save_messages([
        Message(session_scope='s2', content='c1', created_at_tz=base),
        Message(session_scope='s2', content='c2', created_at_tz=base + timedelta(seconds=1)),
        Message(session_scope='s2', content='c3', created_at_tz=base + timedelta(seconds=2)),
    ])
    rows = live_db.get_last_messages('s2', limit=2)
    assert all(isinstance(item, Message) for item in rows)
    assert [item.content for item in rows] == ['c2', 'c3']


# ---------------- 装配与签名 ----------------


def test_constructor_signature():
    _requires_sqlalchemy()
    params = signature(SQLiteDB.__init__).parameters
    assert params['print_sql'].default is False






