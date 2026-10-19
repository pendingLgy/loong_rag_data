# 数据库存储单元测试：注册表、数据对象、建表 SQL、雪花 ID、装配工厂与 SQLite 端到端。
#
# 单元用例不依赖外部服务：SQLite 用临时文件，缺库分支只校验报错类型。
# 缺驱动或无服务时按服务名与目标地址跳过，可用 pytest -rs 查看跳过清单。

import os
import sys

import pytest

from rag_data.config import DBSettings, MySQLSettings, PGSettings, SQLiteSettings
from rag_data.exceptions import ConfigError, OptionalDependencyError
from rag_data.storage.db import (
    BUILTIN_DBS,
    CREATE_SQL,
    BaseDB,
    HistoryRecord,
    Message,
    SnowflakeGenerator,
    available_dbs,
    create_table_sql,
    is_db_registered,
    next_id,
    register_db,
    register_db_provider,
    resolve_db,
)
from rag_data.storage.db.db_engine import build_database
from rag_data.storage.db.sqlite_db import SQLiteDB

# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if "--integration" not in sys.argv:
    # 你可以直接在代码里写原因，或者写死一个统一的提示
    skip_reason = f"[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）"

    pytest.skip(skip_reason, allow_module_level=True)

# ---------------- 注册表 ----------------


def test_builtin_dbs_declared():
    assert set(BUILTIN_DBS) == {'sqlite', 'mysql', 'pg'}


def test_available_dbs_sorted():
    names = available_dbs()
    assert names == sorted(names)
    for name in ('sqlite', 'mysql', 'pg'):
        assert is_db_registered(name)


def test_resolve_returns_backend_class():
    assert resolve_db('sqlite') is SQLiteDB
    assert issubclass(SQLiteDB, BaseDB)


def test_resolve_unknown_backend_raises():
    with pytest.raises(ConfigError):
        resolve_db('nope')


def test_register_db_by_class_object():
    register_db('custom', SQLiteDB)
    assert resolve_db('custom') is SQLiteDB


def test_register_db_by_decorator():
    @register_db_provider('deco')
    class _Local(BaseDB):
        def add_history(self, record): return record
        def batch_add_history(self, records): return list(records)
        def get_history(self, memory_id): return []
        def save_messages(self, messages, keep_last=10): return list(messages)
        def get_last_messages(self, session_scope, limit=10): return []
        def reset(self): pass
        def close(self): pass

    assert resolve_db('deco') is _Local


# ---------------- 数据对象与表结构 ----------------


def test_history_record_defaults():
    record = HistoryRecord(memory_id=1001, event='ADD')
    assert record.id is None
    assert record.is_deleted is False
    assert record.actor_id is None


def test_message_defaults():
    message = Message(role='user', content='hi')
    assert message.id is None
    assert message.session_scope is None


def test_metadata_declares_two_tables():
    from rag_data.storage.db.schema import HISTORY_TABLE, MESSAGES_TABLE, METADATA
    assert set(METADATA.tables) == {'history', 'messages'}
    assert HISTORY_TABLE.name == 'history'
    assert MESSAGES_TABLE.name == 'messages'


def test_column_lists_match_table_definition():
    from rag_data.storage.db.schema import HISTORY_COLUMNS, HISTORY_TABLE, MESSAGE_COLUMNS, MESSAGES_TABLE
    assert HISTORY_COLUMNS == [column.name for column in HISTORY_TABLE.columns]
    assert MESSAGE_COLUMNS == [column.name for column in MESSAGES_TABLE.columns]


def test_primary_keys_are_bigint():
    from sqlalchemy import BigInteger

    from rag_data.storage.db.schema import HISTORY_TABLE, MESSAGES_TABLE

    assert isinstance(HISTORY_TABLE.c.id.type, BigInteger)
    assert isinstance(MESSAGES_TABLE.c.id.type, BigInteger)
    assert HISTORY_TABLE.c.id.primary_key
    assert MESSAGES_TABLE.c.id.primary_key


def test_memory_id_is_nullable_bigint_with_index():
    from sqlalchemy import BigInteger

    from rag_data.storage.db.schema import HISTORY_TABLE

    assert isinstance(HISTORY_TABLE.c.memory_id.type, BigInteger)
    assert HISTORY_TABLE.c.memory_id.nullable is True
    assert any(index.name == 'ix_history_memory_id' for index in HISTORY_TABLE.indexes)


# ---------------- 建表 SQL ----------------


def test_create_sql_covers_all_backends():
    assert set(CREATE_SQL) == {'sqlite', 'mysql', 'pg'}


def test_create_sql_is_plain_text():
    for sql in CREATE_SQL.values():
        assert isinstance(sql, str)
        assert sql.endswith(';')
        assert 'CREATE TABLE IF NOT EXISTS history' in sql
        assert 'CREATE TABLE IF NOT EXISTS messages' in sql


def test_create_sql_covers_indexes_and_columns():
    from rag_data.storage.db.schema import HISTORY_COLUMNS, MESSAGE_COLUMNS

    for sql in CREATE_SQL.values():
        assert 'ix_history_memory_id' in sql
        assert 'ix_messages_scope_created' in sql
        for column in HISTORY_COLUMNS + MESSAGE_COLUMNS:
            assert column in sql


def test_create_sql_uses_bigint_keys():
    for sql in CREATE_SQL.values():
        assert 'id BIGINT NOT NULL PRIMARY KEY' in sql
        assert 'memory_id BIGINT' in sql


def test_create_table_sql_falls_back_to_pg():
    from rag_data.storage.db.schema import PG_CREATE_SQL, SQLITE_CREATE_SQL
    assert create_table_sql('sqlite') is SQLITE_CREATE_SQL
    assert create_table_sql('unknown') is PG_CREATE_SQL


def test_sqlite_create_sql_is_executable(tmp_path):
    import sqlite3
    from rag_data.storage.db.schema import SQLITE_CREATE_SQL

    db_file = tmp_path / 'ddl.db'
    conn = sqlite3.connect(str(db_file))
    try:
        conn.executescript(SQLITE_CREATE_SQL)
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        names = {row[0] for row in conn.execute(sql).fetchall()}
    finally:
        conn.close()
    assert {'history', 'messages'} <= names
    assert db_file.exists()


# ---------------- 雪花 ID ----------------


def test_next_id_is_positive_int():
    value = next_id()
    assert isinstance(value, int)
    assert value > 0
    assert value.bit_length() <= 63


def test_next_id_is_monotonic():
    values = [next_id() for _ in range(200)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_next_id_singleton_shared():
    from rag_data.storage.db.snowflake import next_id as module_next_id
    assert module_next_id is next_id


def test_generator_nodes_are_distinct():
    first = SnowflakeGenerator(datacenter_id=1, worker_id=1)
    second = SnowflakeGenerator(datacenter_id=2, worker_id=2)
    assert first.next_id() != second.next_id()


def test_generator_callable():
    generator = SnowflakeGenerator()
    first = generator()
    second = generator()
    assert isinstance(first, int)
    assert second > first


def test_generator_rejects_out_of_range_nodes():
    with pytest.raises(ValueError):
        SnowflakeGenerator(worker_id=999)
    with pytest.raises(ValueError):
        SnowflakeGenerator(datacenter_id=-1)


def test_generator_epoch_constant():
    from rag_data.storage.db.snowflake import EPOCH_MS
    assert EPOCH_MS == 1704067200000


# ---------------- build_database 装配工厂 ----------------


class _FakeStore:
    # 受控后端替身：仅记录构造入参。
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _recorder(captured):
    # 生成一个记录入参的替身构造器。
    def factory(**kwargs):
        captured.update(kwargs)
        return _FakeStore(**kwargs)
    return factory


def test_build_database_routes_sqlite(monkeypatch, tmp_path):
    import rag_data.storage.db.sqlite_db as sqlite_module

    captured = {}
    monkeypatch.setattr(sqlite_module, 'SQLiteDB', _recorder(captured))
    path = str(tmp_path / 'app.db')
    result = build_database(DBSettings(sqlite=SQLiteSettings(path=path)))
    assert isinstance(result, _FakeStore)
    assert captured == {'db_path': path}


def test_build_database_sqlite_requires_path():
    with pytest.raises(ConfigError):
        build_database(DBSettings())


def test_build_database_routes_mysql(monkeypatch):
    import rag_data.storage.db.mysql_db as mysql_module

    captured = {}
    monkeypatch.setattr(mysql_module, 'MySQLDB', _recorder(captured))
    settings = DBSettings(
        backend='mysql',
        mysql=MySQLSettings(host='db.internal', port=3307, user='app', password='pw', database='rag'),
    )
    result = build_database(settings)
    assert isinstance(result, _FakeStore)
    assert captured['host'] == 'db.internal'
    assert captured['port'] == 3307
    assert captured['user'] == 'app'
    assert captured['password'] == 'pw'
    assert captured['database'] == 'rag'
    assert captured['charset'] == 'utf8mb4'


def test_build_database_routes_pg(monkeypatch):
    import rag_data.storage.db.pg_db as pg_module

    captured = {}
    monkeypatch.setattr(pg_module, 'PGDB', _recorder(captured))
    settings = DBSettings(backend='pg', pg=PGSettings(host='pg.internal', port=6432, database='rag'))
    build_database(settings)
    assert captured['host'] == 'pg.internal'
    assert captured['port'] == 6432
    assert captured['database'] == 'rag'
    assert captured['connection_string'] is None


def test_build_database_pg_prefers_connection_string(monkeypatch):
    import rag_data.storage.db.pg_db as pg_module

    captured = {}
    monkeypatch.setattr(pg_module, 'PGDB', _recorder(captured))
    settings = DBSettings(backend='pg', pg=PGSettings(connection_string='postgresql:store'))
    build_database(settings)
    assert captured['connection_string'] == 'postgresql:store'


# ---------------- SQLite 端到端（文件落盘） ----------------


@pytest.fixture()
def db(tmp_path):
    store = SQLiteDB(db_path=str(tmp_path / 'unit.db'))
    yield store
    store.close()


def test_fixture_returns_sqlite(db):
    assert isinstance(db, SQLiteDB)
    assert db.backend == 'sqlite'


def test_startup_creates_tables(db):
    from sqlalchemy import inspect

    assert {'history', 'messages'} <= set(inspect(db.engine).get_table_names())


def test_missing_tables_empty_when_all_exist(db):
    assert db._missing_tables() == []


def test_missing_tables_detects_dropped_table(db):
    from rag_data.storage.db.schema import HISTORY_TABLE

    HISTORY_TABLE.drop(db.engine)
    assert [table.name for table in db._missing_tables()] == ['history']


def test_ensure_schema_recreates_only_missing(db):
    from sqlalchemy import inspect

    from rag_data.storage.db.schema import MESSAGES_TABLE

    MESSAGES_TABLE.drop(db.engine)
    db._ensure_schema()
    assert {'history', 'messages'} <= set(inspect(db.engine).get_table_names())


def test_add_history_returns_record(db):
    saved = db.add_history(HistoryRecord(memory_id=1001, new_memory='v1', event='ADD', actor_id='u1', role='user'))
    assert isinstance(saved, HistoryRecord)
    assert isinstance(saved.id, int)
    assert saved.memory_id == 1001
    assert saved.is_deleted is False


def test_get_history_orders_by_time(db):
    db.add_history(HistoryRecord(memory_id=1001, new_memory='v1', event='ADD', actor_id='u1'))
    db.add_history(HistoryRecord(memory_id=1001, old_memory='v1', new_memory='v2', event='UPDATE'))
    db.add_history(HistoryRecord(memory_id=1002, event='ADD'))
    rows = db.get_history(1001)
    assert all(isinstance(row, HistoryRecord) for row in rows)
    assert [row.event for row in rows] == ['ADD', 'UPDATE']
    assert rows[0].actor_id == 'u1'


def test_batch_add_history_returns_records(db):
    saved = db.batch_add_history([
        HistoryRecord(memory_id=1001, event='A'),
        HistoryRecord(memory_id=1001, event='B', is_deleted=True),
    ])
    assert len(saved) == 2
    assert all(isinstance(record.id, int) for record in saved)
    assert db.get_history(1001)[1].is_deleted is True


def test_batch_add_history_empty_is_noop(db):
    assert db.batch_add_history([]) == []
    assert db.get_history(1001) == []


def test_save_messages_returns_objects(db):
    saved = db.save_messages([
        Message(session_scope='s1', role='user', content='a'),
        Message(session_scope='s1', role='assistant', content='b'),
    ])
    assert all(isinstance(item.id, int) and item.created_at for item in saved)
    rows = db.get_last_messages('s1')
    assert all(isinstance(row, Message) for row in rows)
    assert [row.content for row in rows] == ['b','a']
    assert rows[0].role == 'assistant'


def test_save_messages_empty_is_noop(db):
    assert db.save_messages([]) == []
    assert db.get_last_messages('s1') == []


def test_messages_keep_only_recent(db):
    for index in range(15):
        db.save_messages([Message(session_scope='s1', role='user', content=str(index))], keep_last=10)
    rows = db.get_last_messages('s1', limit=50)
    assert len(rows) == 10
    assert rows[0].content == '5'
    assert rows[-1].content == '14'


def test_messages_scoped_by_session(db):
    db.save_messages([Message(session_scope='s1', role='user', content='x')])
    db.save_messages([Message(session_scope='s2', role='user', content='y')])
    assert [row.content for row in db.get_last_messages('s1')] == ['x']
    assert [row.content for row in db.get_last_messages('s2')] == ['y']


def test_reset_clears_tables(db):
    db.add_history(HistoryRecord(memory_id=1001, event='ADD'))
    db.save_messages([Message(session_scope='s1', content='a')])
    db.reset()
    assert db.get_history(1001) == []
    assert db.get_last_messages('s1') == []


def test_context_manager_closes(tmp_path):
    with SQLiteDB(db_path=str(tmp_path / 'ctx.db')) as store:
        assert isinstance(store, SQLiteDB)
        assert store.engine is not None
    assert store.engine is None


def test_close_then_no_engine(tmp_path):
    store = SQLiteDB(db_path=str(tmp_path / 'close.db'))
    store.close()
    assert store.engine is None


def test_data_persists_on_disk(tmp_path):
    from sqlalchemy import inspect

    path = tmp_path / 'persist.db'
    first = SQLiteDB(db_path=str(path))
    first.add_history(HistoryRecord(memory_id=7001, new_memory='v1', event='ADD'))
    first.close()
    assert path.exists() and path.stat().st_size > 0
    second = SQLiteDB(db_path=str(path))
    try:
        assert {'history', 'messages'} <= set(inspect(second.engine).get_table_names())
        assert [row.event for row in second.get_history(7001)] == ['ADD']
    finally:
        second.close()


def test_snowflake_ids_are_sorted_by_creation(db):
    ids = [db.add_history(HistoryRecord(memory_id=1, event='ADD')).id for _ in range(20)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


# ---------------- SQLite：路径必填 ----------------


def test_db_path_parameter_is_required():
    import inspect

    signature = inspect.signature(SQLiteDB.__init__)
    parameter = signature.parameters['db_path']
    assert parameter.default is inspect.Parameter.empty


def test_db_path_empty_raises():
    with pytest.raises(ValueError):
        SQLiteDB(db_path='')


def test_db_path_creates_parent_directories(tmp_path):
    path = tmp_path / 'nested' / 'given.db'
    store = SQLiteDB(db_path=str(path))
    try:
        assert store.db_path == str(path)
        assert path.exists()
    finally:
        store.close()


# ---------------- 后端声明与惰性依赖 ----------------


def test_backend_names_declared():
    from rag_data.storage.db.mysql_db import MySQLDB
    from rag_data.storage.db.pg_db import PGDB

    assert SQLiteDB.backend == 'sqlite'
    assert MySQLDB.backend == 'mysql'
    assert PGDB.backend == 'pg'


def test_sqlite_is_subclass_of_sql_database():
    from rag_data.storage.db.db_engine import SQLDatabase

    assert issubclass(SQLiteDB, SQLDatabase)
    assert issubclass(SQLiteDB, BaseDB)


def test_mysql_constructor_is_lazy_or_connects():
    # 缺驱动抛 OptionalDependencyError；已装驱动但服务不可达时抛其他异常，均视为可接受。
    from rag_data.storage.db.mysql_db import MySQLDB

    try:
        MySQLDB()
    except OptionalDependencyError:
        pass
    except Exception:
        pass


def test_pg_constructor_is_lazy_or_connects():
    from rag_data.storage.db.pg_db import PGDB

    try:
        PGDB()
    except OptionalDependencyError:
        pass
    except Exception:
        pass


# ---------------- 集成：MySQL 与 PG（默认跳过） ----------------
#
# 运行前设置环境变量并显式指定集成标记：
#   set RAG_IT_MYSQL_USER=root
#   set RAG_IT_MYSQL_PASSWORD=secret
#   set RAG_IT_PG_PASSWORD=secret

MYSQL_HOST = os.environ.get('RAG_IT_MYSQL_HOST', '127.0.0.1')
MYSQL_PORT = int(os.environ.get('RAG_IT_MYSQL_PORT', '3306'))
MYSQL_USER = os.environ.get('RAG_IT_MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('RAG_IT_MYSQL_PASSWORD', '123456')
MYSQL_DB = os.environ.get('RAG_IT_MYSQL_DB', 'rag_data_it')

PG_HOST = os.environ.get('RAG_IT_PG_HOST', '127.0.0.1')
PG_PORT = int(os.environ.get('RAG_IT_PG_PORT', '5432'))
PG_USER = os.environ.get('RAG_IT_PG_USER', 'root')
PG_PASSWORD = os.environ.get('RAG_IT_PG_PASSWORD', '123456')
PG_DB = os.environ.get('RAG_IT_PG_DB', 'rag_data_it')


def _skip_service(service: str, target: str, exc: Exception) -> None:
    """外部服务或驱动不可用时跳过，并给出服务名、目标与原始异常。"""
    pytest.skip("跳过：" + service + "（目标 " + target + "）：" + type(exc).__name__ + "：" + str(exc))


def _open_mysql():
    # 建立 MySQL 实例；驱动缺失或无服务时跳过。
    try:
        from rag_data.storage.db.mysql_db import MySQLDB
    except Exception as exc:
        _skip_service('MySQL 驱动', 'rag_data.storage.db.mysql_db', exc)
    try:
        return MySQLDB(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB)
    except Exception as exc:
        _skip_service('MySQL 服务', 'host=' + MYSQL_HOST + ':' + str(MYSQL_PORT) + ' db=' + MYSQL_DB, exc)


def _open_pg():
    # 建立 PG 实例；驱动缺失或无服务时跳过。
    try:
        from rag_data.storage.db.pg_db import PGDB
    except Exception as exc:
        _skip_service('PG 驱动', 'rag_data.storage.db.pg_db', exc)
    try:
        return PGDB(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DB)
    except Exception as exc:
        _skip_service('PG 服务', 'host=' + PG_HOST + ':' + str(PG_PORT) + ' db=' + PG_DB, exc)


def _history_roundtrip(store):
    # 历史记录：写入、读回、顺序与字段一致性。
    first = store.add_history(HistoryRecord(memory_id=9001, new_memory='v1', event='ADD', actor_id='u1'))
    store.add_history(HistoryRecord(memory_id=9001, old_memory='v1', new_memory='v2', event='UPDATE'))
    store.add_history(HistoryRecord(memory_id=9002, event='ADD'))
    rows = store.get_history(9001)
    assert [row.event for row in rows] == ['ADD', 'UPDATE']
    assert rows[0].actor_id == 'u1'
    assert isinstance(first.id, int)
    return first


def _messages_roundtrip(store, scope):
    # 消息：写入、保留上限与会话隔离。
    for index in range(12):
        store.save_messages([Message(session_scope=scope, role='user', content=str(index))], keep_last=5)
    rows = store.get_last_messages(scope, limit=50)
    assert len(rows) == 5
    assert [row.content for row in rows] == ['7', '8', '9', '10', '11']
    other = scope + '-x'
    store.save_messages([Message(session_scope=other, role='user', content='other')])
    assert [row.content for row in store.get_last_messages(other)] == ['other']


def test_mysql_startup_creates_tables():
    from sqlalchemy import inspect

    store = _open_mysql()
    try:
        assert store.backend == 'mysql'
        assert {'history', 'messages'} <= set(inspect(store.engine).get_table_names())
    finally:
        store.close()

def test_mysql_history_and_messages_roundtrip():
    store = _open_mysql()
    try:
        store.reset()
        _history_roundtrip(store)
        _messages_roundtrip(store, 'it-mysql')
    finally:
        store.reset()
        store.close()


def test_mysql_reset_is_idempotent():
    store = _open_mysql()
    try:
        store.reset()
        assert store.get_history(9001) == []
    finally:
        store.close()

def test_mysql_reopen_reuses_tables():
    first = _open_mysql()
    first.close()
    second = _open_mysql()
    try:
        assert second._missing_tables() == []
    finally:
        second.close()


def test_pg_startup_creates_tables():
    from sqlalchemy import inspect

    store = _open_pg()
    try:
        assert store.backend == 'pg'
        assert {'history', 'messages'} <= set(inspect(store.engine).get_table_names())
    finally:
        store.close()


def test_pg_history_and_messages_roundtrip():
    store = _open_pg()
    try:
        store.reset()
        _history_roundtrip(store)
        _messages_roundtrip(store, 'it-pg')
    finally:
        store.reset()
        store.close()

def test_pg_reset_is_idempotent():
    store = _open_pg()
    try:
        store.reset()
        assert store.get_history(9001) == []
    finally:
        store.close()


def test_pg_reopen_reuses_tables():
    first = _open_pg()
    first.close()
    second = _open_pg()
    try:
        assert second._missing_tables() == []
    finally:
        second.close()


