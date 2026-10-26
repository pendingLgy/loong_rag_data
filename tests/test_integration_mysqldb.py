# MySQLDB 单元测试：驱动选择、构造参数、连接串归一化与同步引擎装配。
#
# 模拟用例用 monkeypatch 隔离驱动与服务，真实服务用例在服务不可用时自动 skip。
# 父类能力用例连真实 MySQL（conftest 的 MYSQL_SETTINGS），服务不可用时跳过；
# 覆盖 BaseDB 的 6 个抽象方法：_start、add_history、get_history、batch_add_history、save_messages、get_last_messages。

import sys
import time
from datetime import datetime, timedelta, timezone
from inspect import iscoroutinefunction, signature

import pytest

try:
    from sqlalchemy import inspect  # noqa: E402
    from sqlalchemy.sql.dml import Delete  # noqa: E402
    from rag_data.exceptions import OptionalDependencyError, StorageError  # noqa: E402
    from rag_data.storage.db import mysql_db as mysql_module  # noqa: E402
    from rag_data.storage.db.base import BaseDB  # noqa: E402
    from rag_data.storage.db.db_engine import DBEngine  # noqa: E402
    from rag_data.storage.db.mysql_db import MySQLDB  # noqa: E402
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
    """缺 sqlalchemy 或 MySQL 驱动时跳过：连接装配与行转换都依赖它。"""
    if not _SQLALCHEMY_AVAILABLE or not mysql_module._MYSQL_DRIVER:
        pytest.skip("未安装 sqlalchemy 或 MySQL 驱动，请执行 pip install sqlalchemy pymysql")


class _FakeEngine:
    # 假引擎：只记录构造参数，不建立真实连接。

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs


class _FakeSession:
    # 替身会话：只记录提交、回滚与关闭，供 _session 语义校验。

    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _patch_engine(monkeypatch, captured):
    # 用记录型工厂替换 create_engine 与 sessionmaker。

    def engine_factory(url, **kwargs):
        captured['url'] = url
        captured['kwargs'] = kwargs
        return _FakeEngine(url, **kwargs)

    def session_factory(bind=None, **kwargs):
        captured['session'] = dict(bind=bind, **kwargs)
        return dict(bind=bind, **kwargs)

    monkeypatch.setattr(mysql_module, 'create_engine', engine_factory)
    monkeypatch.setattr(mysql_module, 'sessionmaker', session_factory)


def _patch_driver(monkeypatch):
    # 固定驱动常量，避免受本机是否安装 pymysql 影响。
    monkeypatch.setattr(mysql_module, '_MYSQL_DRIVER', 'pymysql')
    monkeypatch.setattr(mysql_module, '_ASYNC_MYSQL_DRIVER', 'aiomysql')


@pytest.fixture()
def started(monkeypatch):
    # 构造并启动 MySQLDB：引擎、会话工厂与建表全部替换。
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    schema_calls = []
    monkeypatch.setattr(MySQLDB, '_ensure_schema', lambda self: schema_calls.append(True))
    store = MySQLDB(host='db.internal', port=3307, user='app', password='p@ss', database='rag')
    store._start()
    return store, captured, schema_calls


# ---------------- 注册与继承 ----------------


def test_backend_name_is_mysql():
    _requires_sqlalchemy()
    assert MySQLDB.backend == 'mysql'


def test_mysql_inherits_engine():
    _requires_sqlalchemy()
    assert issubclass(MySQLDB, DBEngine)
    assert issubclass(MySQLDB, BaseDB)


def test_registered_as_mysql_backend():
    _requires_sqlalchemy()
    assert is_db_registered('mysql')
    assert resolve_db('mysql') is MySQLDB


# ---------------- 构造行为 ----------------


def test_default_parameters():
    _requires_sqlalchemy()
    store = MySQLDB()
    assert store.host == 'localhost'
    assert store.port == 3306
    assert store.user == 'root'
    assert store.password == ''
    assert store.database == 'rag_data'
    assert store.charset == 'utf8mb4'
    assert store.print_sql is False
    assert store.connection_string is None


def test_pool_parameters_default():
    _requires_sqlalchemy()
    store = MySQLDB()
    assert store.pool_size == 10
    assert store.max_overflow == 20
    assert store.pool_timeout == 30
    assert store.pool_recycle == 1800
    assert store.pool_pre_ping is True


def test_constructor_keeps_custom_parameters():
    _requires_sqlalchemy()
    store = MySQLDB(connection_string='mysql://u@h/db', host='h2', port=3308, database='other')
    assert store.connection_string == 'mysql://u@h/db'
    assert store.host == 'h2'
    assert store.port == 3308
    assert store.database == 'other'


def test_constructor_is_lazy():
    _requires_sqlalchemy()
    store = MySQLDB()
    assert store.engine is None
    assert store.SessionLocal is None


def test_constructor_does_not_open_connection(monkeypatch):
    _requires_sqlalchemy()
    # 构造期只保存参数，create_engine 不应被调用。

    def boom(url, **kwargs):
        raise AssertionError('构造期不应创建引擎')

    monkeypatch.setattr(mysql_module, 'create_engine', boom)
    MySQLDB(host='h')


# ---------------- 启动：驱动与连接串 ----------------


def test_start_requires_installed_driver(monkeypatch):
    _requires_sqlalchemy()
    monkeypatch.setattr(mysql_module, '_MYSQL_DRIVER', '')
    monkeypatch.setattr(mysql_module, '_MYSQL_ERROR', ImportError('missing'))
    with pytest.raises(OptionalDependencyError):
        MySQLDB()._start()


def test_start_builds_url_from_discrete_params(started):
    _requires_sqlalchemy()
    _, captured, _ = started
    url = captured['url']
    assert 'mysql+pymysql:' in url
    assert 'db.internal:3307' in url
    assert 'rag' in url


def test_start_hides_password_in_url(started):
    _requires_sqlalchemy()
    # 连接串按 hide_password 渲染，密码不落明文。
    _, captured, _ = started
    assert 'p@ss' not in captured['url']


def test_start_assigns_engine(started):
    _requires_sqlalchemy()
    store, _, _ = started
    assert isinstance(store.engine, _FakeEngine)


def test_start_binds_session_factory(started):
    _requires_sqlalchemy()
    store, captured, _ = started
    assert captured['session']['bind'] is store.engine


def test_start_passes_pool_arguments(started):
    _requires_sqlalchemy()
    _, captured, _ = started
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
    monkeypatch.setattr(MySQLDB, '_ensure_schema', lambda self: None)
    MySQLDB(print_sql=True)._start()
    assert captured['kwargs']['echo'] is True


def test_start_ensures_schema_once(started):
    _requires_sqlalchemy()
    _, _, schema_calls = started
    assert schema_calls == [True]


def test_start_stores_rendered_url(started):
    _requires_sqlalchemy()
    store, captured, _ = started
    assert store.connection_string == captured['url']


@pytest.mark.parametrize('prefix', ['mysql+pymysql', 'mysql+mysqlconnector', 'mysql'])
def test_start_normalizes_connection_string(monkeypatch, prefix):
    _requires_sqlalchemy()
    # 三种同步前缀都归一成驱动形式，与驱动无关的部分保持一致。
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    monkeypatch.setattr(MySQLDB, '_ensure_schema', lambda self: None)
    MySQLDB(connection_string=prefix + '://root:pw@h:3306/rag')._start()
    assert captured['url'].startswith('mysql+')
    assert 'root:pw@h:3306/rag' in captured['url']


def test_start_keeps_foreign_connection_string(monkeypatch):
    _requires_sqlalchemy()
    # 非 mysql 前缀原样透传。
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)
    monkeypatch.setattr(MySQLDB, '_ensure_schema', lambda self: None)
    MySQLDB(connection_string='postgresql://u:p@h/db')._start()
    assert captured['url'] == 'postgresql://u:p@h/db'


def test_start_uses_sync_driver_name(monkeypatch):
    _requires_sqlalchemy()
    # 已知缺陷：同步 _start 用 _ASYNC_MYSQL_DRIVER 拼前缀，会把 mysql:// 改成 mysql+aiomysql://。
    captured = {}
    _patch_engine(monkeypatch, captured)
    _patch_driver(monkeypatch)

    def _mock_init_db(self, engine, sessionLocal):
        return None

    monkeypatch.setattr(MySQLDB, '_init_db', _mock_init_db)
    MySQLDB(connection_string='mysql://root:pw@h/rag')._start()
    assert 'mysql+pymysql:' in captured['url']


# ---------------- 会话生命周期 ----------------


def test_session_commits_on_success():
    _requires_sqlalchemy()
    store = MySQLDB()
    fake = _FakeSession()
    store.SessionLocal = lambda: fake
    with store._session() as session:
        assert session is fake
    assert fake.committed is True
    assert fake.rolled_back is False
    assert fake.closed is True


def test_session_rolls_back_on_error():
    _requires_sqlalchemy()
    store = MySQLDB()
    fake = _FakeSession()
    store.SessionLocal = lambda: fake
    with pytest.raises(RuntimeError):
        with store._session():
            raise RuntimeError('boom')
    assert fake.committed is False
    assert fake.rolled_back is True
    assert fake.closed is True


# ---------------- 父类 BaseDB 抽象契约 ----------------
#
# BaseDB 是关系库最上层接口：只声明契约与自动登记，不含任何方言实现。
# 以下用例同样不连数据库，全部通过临时子类验证抽象约束与登记行为。

# BaseDB 声明的全部抽象方法，子类必须逐一实现。
BASE_ABSTRACT_METHODS = (
    '_start',
    'add_history',
    'batch_add_history',
    'get_history',
    'save_messages',
    'get_last_messages',
)


def _build_base(class_name, backend=None, skip=None):
    # 按需剔除一个抽象方法，构造用于验证契约的 BaseDB 子类。
    namespace = {}
    for method in BASE_ABSTRACT_METHODS:
        if method != skip:
            namespace[method] = lambda self, *args, **kwargs: None
    if backend is not None:
        namespace['backend'] = backend
    return type(class_name, (BaseDB,), namespace)


def test_base_db_is_abstract():
    _requires_sqlalchemy()
    with pytest.raises(TypeError):
        BaseDB()


def test_base_db_backend_default_is_none():
    _requires_sqlalchemy()
    assert BaseDB.backend is None


def test_base_db_declares_expected_abstract_methods():
    _requires_sqlalchemy()
    assert set(BaseDB.__abstractmethods__) == set(BASE_ABSTRACT_METHODS)


def test_base_db_declares_no_dialect_detail():
    # 抽象层不应出现任何连接参数，方言细节留给子类。
    _requires_sqlalchemy()
    assert not hasattr(BaseDB, 'engine')


@pytest.mark.parametrize('missing', BASE_ABSTRACT_METHODS)
def test_subclass_missing_one_method_stays_abstract(missing):
    _requires_sqlalchemy()
    probe = _build_base('_ProbeMissing', skip=missing)
    with pytest.raises(TypeError):
        probe()


def test_concrete_subclass_is_instantiable():
    _requires_sqlalchemy()
    probe = _build_base('_ProbeConcrete')
    assert isinstance(probe(), BaseDB)


# ---------------- 父类 BaseDB 自动登记 ----------------


def test_subclass_without_backend_is_not_registered():
    _requires_sqlalchemy()
    before = set(available_dbs())
    _build_base('_ProbeNoBackend')
    assert set(available_dbs()) == before


def test_subclass_with_backend_is_auto_registered():
    _requires_sqlalchemy()
    probe = _build_base('_ProbeRegistered', backend='autotest_probe_base')
    assert is_db_registered('autotest_probe_base')
    assert resolve_db('autotest_probe_base') is probe


def test_auto_registered_backend_appears_in_available():
    _requires_sqlalchemy()
    _build_base('_ProbeListed', backend='autotest_probe_listed')
    assert 'autotest_probe_listed' in available_dbs()


def test_empty_backend_does_not_register():
    # backend 为空串时视为未声明，不进入注册表。
    _requires_sqlalchemy()
    before = set(available_dbs())
    probe = _build_base('_ProbeEmpty', backend='')
    assert probe.backend == ''
    assert set(available_dbs()) == before


def test_inherited_backend_is_not_registered_again():
    # 只看子类自身命名空间：继承来的 backend 不重复登记。
    _requires_sqlalchemy()
    parent = _build_base('_ProbeParent', backend='autotest_probe_parent')
    child = type('_ProbeChild', (parent,), {})
    assert child.backend == 'autotest_probe_parent'
    assert resolve_db('autotest_probe_parent') is parent


# ---------------- 父类 BaseDB 方法签名 ----------------


def test_save_messages_keep_last_default():
    _requires_sqlalchemy()
    assert signature(BaseDB.save_messages).parameters['keep_last'].default == 10


def test_get_last_messages_limit_default():
    _requires_sqlalchemy()
    assert signature(BaseDB.get_last_messages).parameters['limit'].default == 10


def test_history_methods_take_expected_parameters():
    _requires_sqlalchemy()
    assert list(signature(BaseDB.add_history).parameters) == ['self', 'record']
    assert list(signature(BaseDB.batch_add_history).parameters) == ['self', 'records']
    assert list(signature(BaseDB.get_history).parameters) == ['self', 'memory_id']


def test_message_methods_take_expected_parameters():
    _requires_sqlalchemy()
    assert list(signature(BaseDB.save_messages).parameters) == ['self', 'messages', 'keep_last']
    assert list(signature(BaseDB.get_last_messages).parameters) == ['self', 'session_scope', 'limit']


def test_mysql_db_clears_all_abstract_methods():
    # 内置实现必须实现全部抽象方法，可直接实例化。
    _requires_sqlalchemy()
    assert issubclass(MySQLDB, BaseDB)
    assert MySQLDB.__abstractmethods__ == frozenset()


# ============ 真实 MySQL 连接 ============
#
# 服务不可用时整体跳过；连接使用 conftest 的 MYSQL_SETTINGS 分区配置。

# 每次运行用独立的标识，避免与既有数据或其他用例互相干扰。


RUN_ID = int(time.time()) % 1000000
HISTORY_MEMORY_ID = RUN_ID * 100 + 3
BATCH_MEMORY_ID = RUN_ID * 100 + 4
MESSAGE_SCOPE = 'autotest_sync_' + str(RUN_ID)


@pytest.fixture(scope='module')
def live_store():
    """真实 MySQL 连接：服务不可用或初始化失败时跳过，用完释放连接池。"""
    _requires_sqlalchemy()
    from conftest import MYSQL_SETTINGS

    store = MySQLDB(
        host=MYSQL_SETTINGS.host,
        port=MYSQL_SETTINGS.port,
        user=MYSQL_SETTINGS.user,
        password=MYSQL_SETTINGS.password,
        database=MYSQL_SETTINGS.database,
    )
    try:
        store._start()
    except Exception as exc:
        pytest.skip('[SKIP INFO] MySQL 服务不可用：' + str(exc))

    try:
        yield store
    finally:
        if store.engine:
            store.engine.dispose()


# ---------------- 真实服务（服务不可用时跳过） ----------------


def test_mysql_service_creates_tables():
    _requires_sqlalchemy()
    from conftest import MYSQL_SETTINGS

    store = MySQLDB(
        host=MYSQL_SETTINGS.host,
        port=MYSQL_SETTINGS.port,
        user=MYSQL_SETTINGS.user,
        password=MYSQL_SETTINGS.password,
        database=MYSQL_SETTINGS.database,
    )
    try:
        store._start()
    except Exception as exc:
        pytest.skip('[SKIP INFO] MySQL 服务不可用：' + str(exc))

    try:
        assert {'history', 'messages'} <= set(inspect(store.engine).get_table_names())
    finally:
        store.engine.dispose()


def test_mysql_service_start_is_idempotent():
    _requires_sqlalchemy()
    from conftest import MYSQL_SETTINGS

    store = MySQLDB(
        host=MYSQL_SETTINGS.host,
        port=MYSQL_SETTINGS.port,
        user=MYSQL_SETTINGS.user,
        password=MYSQL_SETTINGS.password,
        database=MYSQL_SETTINGS.database,
    )
    try:
        store._start()
    except Exception as exc:
        pytest.skip('[SKIP INFO] MySQL 服务不可用：' + str(exc))

    try:
        # 已建好的表不再重复创建，启动可重复执行。
        assert store._missing_tables() == []
    finally:
        store.engine.dispose()


# ---------------- 父类 DBEngine 能力（真实 MySQL 连接） ----------------
#
# 父类逻辑与方言无关；这里连真实 MySQL，既验证父类流程，也验证 SQL 在真实方言下可执行。


def test_parent_start_creates_engine(live_store):
    _requires_sqlalchemy()
    assert live_store.engine is not None
    assert live_store.SessionLocal is not None


def test_parent_start_creates_tables(live_store):
    _requires_sqlalchemy()
    assert {'history', 'messages'} <= set(inspect(live_store.engine).get_table_names())


def test_parent_ensure_schema_wraps_error(monkeypatch):
    _requires_sqlalchemy()
    # 建表检查失败时统一转成 StorageError，便于上层按存储错误处理。

    def boom(self):
        raise RuntimeError('ddl failed')

    monkeypatch.setattr(MySQLDB, '_missing_tables', boom)
    with pytest.raises(StorageError):
        MySQLDB()._ensure_schema()


def test_parent_add_history_persists(live_store):
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


def test_parent_get_history_returns_rows(live_store):
    _requires_sqlalchemy()
    live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, event='ADD', actor_id='u1')
    )
    live_store.add_history(
        HistoryRecord(memory_id=HISTORY_MEMORY_ID, event='UPDATE', actor_id='u1')
    )
    rows = live_store.get_history(HISTORY_MEMORY_ID)
    assert len(rows) >= 2
    assert all(isinstance(row, HistoryRecord) for row in rows)


# ---------------- 父类 BaseDB 其余契约方法（真实 MySQL 连接） ----------------
#
# BaseDB 共声明 6 个抽象方法：_start、add_history、get_history 已在上方覆盖，
# 此处补齐 batch_add_history、save_messages 与 get_last_messages。


def test_parent_batch_add_history_returns_records(live_store):
    _requires_sqlalchemy()
    saved = live_store.batch_add_history([
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b1', event='ADD', actor_id='u2'),
        HistoryRecord(memory_id=BATCH_MEMORY_ID, old_memory='b1', new_memory='b2', event='UPDATE', actor_id='u2'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item, HistoryRecord) for item in saved)
    assert all(isinstance(item.id, int) for item in saved)
    assert [item.event for item in saved] == ['ADD', 'UPDATE']


def test_parent_batch_add_history_persists(live_store):
    _requires_sqlalchemy()
    live_store.batch_add_history([
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b3', event='ADD'),
        HistoryRecord(memory_id=BATCH_MEMORY_ID, new_memory='b4', event='ADD'),
    ])
    rows = live_store.get_history(BATCH_MEMORY_ID)
    assert len(rows) >= 2
    assert 'ADD' in {row.event for row in rows}


def test_parent_batch_add_history_empty_returns_empty(live_store):
    _requires_sqlalchemy()
    # 空批次不触达数据库，直接返回空列表。
    assert live_store.batch_add_history([]) == []


def test_parent_save_messages_returns_objects(live_store):
    _requires_sqlalchemy()
    saved = live_store.save_messages([
        Message(session_scope=MESSAGE_SCOPE, role='user', content='a'),
        Message(session_scope=MESSAGE_SCOPE, role='assistant', content='b'),
    ])
    assert len(saved) == 2
    assert all(isinstance(item, Message) for item in saved)
    assert all(isinstance(item.id, int) for item in saved)
    assert [item.content for item in saved] == ['a', 'b']


def test_parent_save_messages_empty_returns_empty(live_store):
    _requires_sqlalchemy()
    # 空批次不触达数据库，直接返回空列表。
    assert live_store.save_messages([]) == []


def test_parent_save_messages_keeps_last_only(live_store):
    _requires_sqlalchemy()
    scope = MESSAGE_SCOPE + '_evict'
    base = datetime.now(tz=timezone.utc)
    live_store.save_messages([
        Message(session_scope=scope, content='e1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, content='e2', created_at_tz=base - timedelta(seconds=2)),
    ], keep_last=2)
    live_store.save_messages([
        Message(session_scope=scope, content='e3', created_at_tz=base - timedelta(seconds=1)),
    ], keep_last=2)
    rows = live_store.get_last_messages(scope, limit=10)
    assert [item.content for item in rows] == ['e2', 'e3']


def test_parent_get_last_messages_returns_rows(live_store):
    _requires_sqlalchemy()
    scope = MESSAGE_SCOPE + '_read'
    base = datetime.now(tz=timezone.utc)
    live_store.save_messages([
        Message(session_scope=scope, role='user', content='q1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, role='assistant', content='a1', created_at_tz=base - timedelta(seconds=2)),
    ])
    rows = live_store.get_last_messages(scope)
    assert all(isinstance(item, Message) for item in rows)
    assert [item.content for item in rows] == ['q1', 'a1']


def test_parent_get_last_messages_respects_limit(live_store):
    _requires_sqlalchemy()
    scope = MESSAGE_SCOPE + '_limit'
    base = datetime.now(tz=timezone.utc)
    live_store.save_messages([
        Message(session_scope=scope, content='c1', created_at_tz=base - timedelta(seconds=3)),
        Message(session_scope=scope, content='c2', created_at_tz=base - timedelta(seconds=2)),
        Message(session_scope=scope, content='c3', created_at_tz=base - timedelta(seconds=1)),
    ])
    rows = live_store.get_last_messages(scope, limit=2)
    assert [item.content for item in rows] == ['c2', 'c3']


# ---------------- 父类 _build_evict_stmt：构造会话消息的删除语句 ----------------
#
# _build_evict_stmt 只构造语句：保留每个会话最近 keep_last 条，删除其余记录；执行由调用方在自己的会话中完成。


EVICT_SCOPE = MESSAGE_SCOPE + '_build_evict_stmt'


def test_parent_evict_stmt_is_not_coroutine():
    _requires_sqlalchemy()
    assert not iscoroutinefunction(DBEngine._build_evict_stmt)
    assert list(signature(DBEngine._build_evict_stmt).parameters) == ['self', 'session_scope', 'keep_last']


def test_parent_evict_stmt_returns_delete_statement(live_store):
    _requires_sqlalchemy()
    stmt = live_store._build_evict_stmt(EVICT_SCOPE, 2)
    assert isinstance(stmt, Delete)
    assert stmt.table.name == 'messages'


def test_parent_evict_stmt_compiles_for_scope_and_keep_last(live_store):
    _requires_sqlalchemy()
    stmt = live_store._build_evict_stmt(EVICT_SCOPE, 2)
    sql = str(stmt)
    assert sql.startswith('DELETE FROM messages')
    assert 'NOT IN' in sql.upper()
    params = stmt.compile().params
    assert EVICT_SCOPE in params.values()
    assert 2 in params.values()
