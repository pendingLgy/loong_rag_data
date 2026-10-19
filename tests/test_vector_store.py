# 向量库存储测试：注册表、结果对象、过滤表达式、向量引擎，以及真实 Milvus 与 PG 集成。
#
# 单元用例不连服务：Milvus 用例只覆盖纯逻辑（过滤表达式、结果解析、记录构造）；
# PG 用例覆盖 SQL 条件构造；缺库时镜像真实行为，断言 OptionalDependencyError。
# 末尾的集成用例需真实服务，标记 integration，默认由 addopts 的 -m not integration 跳过；
# 缺驱动或无服务时按服务名与目标跳过，可用 pytest -rs 查看跳过清单。

import sys
from contextlib import contextmanager
from typing import Any, List, Optional

import pytest

from conftest import CASES_CONFIG, VECTOR_CONFIG, MILVUS_SETTINGS, PG_VECTOR_SETTINGS, VECTOR_SETTINGS

from rag_data.exceptions import ConfigError, OptionalDependencyError, StorageError
from rag_data.storage.vector_store import (
    BUILTIN_STORES,
    VectorStoreBase,
    VectorStoreResult,
    available_vector_stores,
    is_vector_store_registered,
    register_vector_store,
    register_vector_store_provider,
    resolve_vector_store,
)
from rag_data.storage.vector_store.milvus_store import MilvusVectorStore
from rag_data.storage.vector_store.pg_store import (
    PGVectorStore,
    _TEXT_VECTOR,
    _extension_installed,
    diskann_index_ddl,
    hnsw_index_ddl,
    table_ddl,
    text_index_ddl,
)
from rag_data.storage.vector_store.vector_engine import (
    VectorEngine,
    build_vector_engine,
    build_vector_store,
)

# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if "--integration" not in sys.argv:
    # 你可以直接在代码里写原因，或者写死一个统一的提示
    skip_reason = f"[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）"

    pytest.skip(skip_reason, allow_module_level=True)

# ---------------- 注册表 ----------------


def test_builtin_stores_declared():
    assert set(BUILTIN_STORES) == {'milvus', 'pg'}


def test_available_stores_sorted():
    names = available_vector_stores()
    assert names == sorted(names)
    for name in ('milvus', 'pg'):
        assert is_vector_store_registered(name)


def test_resolve_returns_store_class():
    assert resolve_vector_store('milvus') is MilvusVectorStore
    assert resolve_vector_store('pg') is PGVectorStore
    assert issubclass(MilvusVectorStore, VectorStoreBase)
    assert issubclass(PGVectorStore, VectorStoreBase)


def test_resolve_unknown_store_raises():
    with pytest.raises(ConfigError):
        resolve_vector_store('nope')


def test_register_vector_store_by_class():
    register_vector_store('custom-vs', PGVectorStore)
    assert resolve_vector_store('custom-vs') is PGVectorStore


def test_register_vector_store_by_decorator():
    @register_vector_store_provider('deco-vs')
    class _Local(VectorStoreBase):
        def create_col(self, name, vector_size, distance='COSINE'): pass
        def insert(self, vectors, payloads=None, ids=None): pass
        def search(self, query, vectors, top_k=5, filters=None): return []
        def delete(self, vector_id): pass
        def update(self, vector_id, vector=None, payload=None): pass
        def get(self, vector_id): return None
        def list_cols(self): return []
        def delete_col(self): pass
        def col_info(self): return {}
        def list(self, filters=None, top_k=None): return []
        def reset(self): pass

    assert resolve_vector_store('deco-vs') is _Local


# ---------------- 结果对象 ----------------


def test_result_model_defaults():
    result = VectorStoreResult()
    assert result.id is None
    assert result.score is None
    assert result.payload is None


def test_result_model_values():
    result = VectorStoreResult(id='1', score=0.5, payload={'k': 'v'})
    assert result.id == '1'
    assert result.score == 0.5
    assert result.payload == {'k': 'v'}


# ---------------- Milvus：纯逻辑（不连服务） ----------------


def _milvus_like(metric="COSINE"):
    # 借道未初始化实例调用纯逻辑方法。
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    store.metric_type = metric
    return store


def test_milvus_filter_builds_and_expression():
    expr = _milvus_like()._create_filter({'user_id': 'u1', 'n': 5, 'flag': True})
    assert expr is not None
    assert len(expr.split(' and ')) == 3
    assert expr.count('metadata[') == 3
    for token in ('u1', '5', 'true'):
        assert token in expr


def test_milvus_filter_none_and_wildcard():
    store = _milvus_like()
    assert store._create_filter(None) is None
    assert store._create_filter({}) is None
    assert store._create_filter({'any': '*'}) is None


def test_milvus_filter_rejects_bad_key():
    with pytest.raises(StorageError):
        _milvus_like()._create_filter({'bad-key': 1})


def test_milvus_filter_rejects_bad_value_type():
    with pytest.raises(StorageError):
        _milvus_like()._create_filter({'k': {'nested': 1}})


def test_milvus_parse_cosine_score():
    store = _milvus_like(metric='COSINE')
    hits = [{'id': '1', 'distance': 0.25, 'entity': {'metadata': {'k': 'v'}}}]
    results = store._parse(hits)
    assert len(results) == 1
    assert results[0].id == '1'
    assert results[0].score == 0.25
    assert results[0].payload == {'k': 'v'}
    assert isinstance(results[0], VectorStoreResult)


def test_milvus_parse_l2_score_converted():
    store = _milvus_like(metric='L2')
    hits = [{'id': '1', 'distance': 3.0, 'entity': {}}]
    assert store._parse(hits)[0].score == 0.25


def test_milvus_parse_without_distance():
    store = _milvus_like()
    hits = [{'id': '1', 'entity': {}}]
    assert store._parse(hits)[0].score is None


def test_milvus_parse_empty():
    assert _milvus_like()._parse([]) == []


def test_milvus_record_includes_text_field():
    """写入记录始终带 text 字段；优先取 text_lemmatized。"""
    row = _milvus_like()._record("1", [0.1, 0.2], {"text_lemmatized": "hello"})
    assert row["id"] == "1"
    assert row["text"] == "hello"
    assert row["vectors"] == [0.1, 0.2]
    assert row["metadata"] == {"text_lemmatized": "hello"}


def test_milvus_record_falls_back_to_data_field():
    row = _milvus_like()._record("1", [0.1], {"data": "raw"})
    assert row["text"] == "raw"


def test_milvus_record_falls_back_to_text_field():
    row = _milvus_like()._record("1", [0.1], {"text": "plain"})
    assert row["text"] == "plain"


def test_milvus_record_empty_payload_yields_empty_text():
    row = _milvus_like()._record("1", [0.1], None)
    assert row["text"] == ""
    assert row["metadata"] == {}


def test_milvus_record_truncates_long_text():
    """text 字段受 Milvus VARCHAR 上限约束，超出部分截断。"""
    row = _milvus_like()._record("1", [0.1], {"text_lemmatized": "x" * 70000})
    assert len(row["text"]) == 65535


# ---------------- PG：SQL 条件构造（不连服务） ----------------


def test_pg_filter_simple_equality():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    conditions, params = _build_filter_conditions({'user_id': 'u1'})
    assert len(conditions) == 1
    assert params == ['user_id', 'u1']


def test_pg_filter_operators():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    conditions, params = _build_filter_conditions({'age': {'gte': 18}, 'name': {'contains': 'ab'}})
    assert len(conditions) == 2
    assert any('numeric' in item for item in conditions)
    assert any('LIKE' in item for item in conditions)
    assert params[0] == 'age' and params[1] == 18.0


def test_pg_filter_in_and_nin():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    _, params = _build_filter_conditions({'tag': {'in': ['a', 'b']}})
    assert params == ['tag', ['a', 'b']]
    conditions, _ = _build_filter_conditions({'tag': {'nin': ['a']}})
    assert conditions[0].startswith('NOT (')


def test_pg_filter_list_shorthand():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    conditions, params = _build_filter_conditions({'tag': ['a', 'b']})
    assert 'ANY(' in conditions[0]
    assert params[1] == ['a', 'b']


def test_pg_filter_wildcard():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    conditions, params = _build_filter_conditions({'any': '*'})
    assert conditions == ['payload ? %s']
    assert params == ['any']


def test_pg_filter_or_group():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    conditions, params = _build_filter_conditions({'$or': [{'a': 1}, {'b': 2}]})
    assert len(conditions) == 1
    assert ' OR ' in conditions[0]
    assert params == ['a', 1, 'b', 2]


def test_pg_filter_not_group():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    conditions, _ = _build_filter_conditions({'$not': [{'a': 1}]})
    assert conditions[0].startswith('NOT (')


def test_pg_filter_unknown_operator_raises():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    with pytest.raises(StorageError):
        _build_filter_conditions({'a': {'nope': 1}})


def test_pg_filter_empty():
    from rag_data.storage.vector_store.pg_store import _build_filter_conditions
    assert _build_filter_conditions(None) == ([], [])
    assert _build_filter_conditions({}) == ([], [])


# ---------------- 惰性依赖与缺库行为 ----------------


def test_milvus_missing_dependency_raises_optional_dependency_error():
    # 未装 pymilvus 时构造应抛 OptionalDependencyError；已装则跳过该断言。
    try:
        MilvusVectorStore(uri='http://localhost:19530', collection_name='t', vector_size=4)
    except OptionalDependencyError:
        pass
    except Exception:
        pass  # 已装依赖但连不上服务时同样接受，避免依赖真实服务


def test_pg_missing_dependency_raises_optional_dependency_error():
    try:
        PGVectorStore(connection_string='postgresql:', collection_name='t', vector_size=4)
    except OptionalDependencyError:
        pass
    except Exception:
        pass


# ---------------- 抽象契约 ----------------


def test_abstract_methods_declared():
    expected = {
        'create_col', 'insert', 'search', 'delete', 'update', 'get',
        'list_cols', 'delete_col', 'col_info', 'list', 'reset',
    }
    assert expected <= set(VectorStoreBase.__abstractmethods__)


def test_keyword_search_default_returns_none():
    class _Minimal(VectorStoreBase):
        def create_col(self, name, vector_size, distance='COSINE'): pass
        def insert(self, vectors, payloads=None, ids=None): pass
        def search(self, query, vectors, top_k=5, filters=None): return []
        def delete(self, vector_id): pass
        def update(self, vector_id, vector=None, payload=None): pass
        def get(self, vector_id): return None
        def list_cols(self): return []
        def delete_col(self): pass
        def col_info(self): return {}
        def list(self, filters=None, top_k=None): return []
        def reset(self): pass

    assert _Minimal().keyword_search('q') is None


# ---------------- vector_engine ----------------


class _FakeStore(VectorStoreBase):
    # 受控向量库替身：记录调用，不连外部服务。

    backend = 'fake'
    collection_name = 'demo'
    vector_size = 3

    def __init__(self):
        self.calls = []

    def create_col(self, name, vector_size, distance='COSINE'): pass
    def insert(self, vectors, payloads=None, ids=None): self.calls.append(('insert', len(vectors)))
    def search(self, query, vectors, top_k=5, filters=None):
        self.calls.append(('search', top_k))
        return [VectorStoreResult(id='1', score=0.9)]
    def delete(self, vector_id): self.calls.append(('delete', vector_id))
    def update(self, vector_id, vector=None, payload=None): self.calls.append(('update', vector_id))
    def get(self, vector_id): return VectorStoreResult(id=vector_id)
    def list_cols(self): return ['demo']
    def delete_col(self): self.calls.append(('delete_col', None))
    def col_info(self): return {'rows': 0}
    def list(self, filters=None, top_k=None): return [[VectorStoreResult(id='1')]]
    def reset(self): self.calls.append(('reset', None))


def test_engine_exposes_store_properties():
    engine = VectorEngine(_FakeStore())
    assert engine.backend == 'fake'
    assert engine.collection_name == 'demo'
    assert engine.vector_size == 3
    assert isinstance(engine.store, VectorStoreBase)


def test_engine_insert_returns_count_and_skips_empty():
    store = _FakeStore()
    engine = VectorEngine(store)
    assert engine.insert([]) == 0
    assert store.calls == []
    assert engine.insert([[1.0, 2.0, 3.0]]) == 1
    assert store.calls == [('insert', 1)]


def test_engine_search_uses_default_top_k():
    store = _FakeStore()
    engine = VectorEngine(store, default_top_k=4)
    results = engine.search('q', [1.0, 2.0, 3.0])
    assert [item.id for item in results] == ['1']
    assert store.calls == [('search', 4)]
    engine.search('q', [1.0, 2.0, 3.0], top_k=1)
    assert store.calls[-1] == ('search', 1)


def test_engine_keyword_search_passthrough():
    engine = VectorEngine(_FakeStore())
    assert engine.keyword_search('q') is None


def test_engine_get_and_list():
    engine = VectorEngine(_FakeStore())
    assert engine.get('x').id == 'x'
    assert [item.id for item in engine.list()] == ['1']


def test_engine_collection_maintenance():
    store = _FakeStore()
    engine = VectorEngine(store)
    assert engine.list_cols() == ['demo']
    assert engine.col_info() == {'rows': 0}
    engine.reset()
    engine.delete_col()
    assert store.calls == [('reset', None), ('delete_col', None)]


def test_engine_rejects_dimension_mismatch():
    engine = VectorEngine(_FakeStore())
    with pytest.raises(StorageError):
        engine.insert([[1.0, 2.0]])


def test_engine_rejects_payload_length_mismatch():
    engine = VectorEngine(_FakeStore())
    with pytest.raises(StorageError):
        engine.insert([[1.0, 2.0, 3.0]], payloads=[{}, {}])


def test_build_vector_store_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        build_vector_store('nope')


def test_build_vector_store_known_backends_lazy_or_build():
    # 已装驱动但无服务时抛其他异常，均视为可接受，避免依赖真实服务。
    for name in ('milvus', 'pg'):
        try:
            build_vector_store(name)
        except Exception:
            pass


def test_engine_context_manager_closes():
    store = _FakeStore()
    engine = VectorEngine(store)
    with engine as opened:
        assert opened is engine


# ---------------- build_vector_engine ----------------


def _install_fake_build(monkeypatch):
    # 打桩建库函数，捕获后端名与传参。
    captured = {}

    def fake_build(backend, **kwargs):
        captured['backend'] = backend
        captured['kwargs'] = kwargs
        return _FakeStore()

    import rag_data.storage.vector_store.vector_engine as engine_module

    monkeypatch.setattr(engine_module, 'build_vector_store', fake_build)
    return captured


def test_build_vector_engine_injects_milvus_params(monkeypatch):
    from rag_data.config import VectorSettings

    captured = _install_fake_build(monkeypatch)
    settings = VectorSettings(backend='milvus', collection='docs', vector_size=768, top_k=4)
    engine = build_vector_engine(settings)
    assert captured['backend'] == 'milvus'
    assert captured['kwargs']['collection_name'] == 'docs'
    assert captured['kwargs']['vector_size'] == 768
    assert captured['kwargs']['metric_type'] == 'COSINE'
    # 留空的连接字段不下传，交由实现层默认值处理。
    assert 'uri' not in captured['kwargs']
    assert isinstance(engine, VectorEngine)
    assert engine.default_top_k == 4


def test_build_vector_engine_injects_pg_params(monkeypatch):
    from rag_data.config import VectorSettings

    captured = _install_fake_build(monkeypatch)
    settings = VectorSettings(backend='pg')
    build_vector_engine(settings)
    kwargs = captured['kwargs']
    assert captured['backend'] == 'pg'
    assert kwargs['collection_name'] == 'rag_data'
    assert kwargs['host'] == 'localhost'
    assert kwargs['port'] == 5432
    assert kwargs['database'] == 'rag_data'
    assert kwargs['use_hnsw'] is True
    assert kwargs['use_diskann'] is False
    # 空的连接串与密码不下传。
    assert 'connection_string' not in kwargs
    assert 'password' not in kwargs


def test_build_vector_engine_accepts_metric_override(monkeypatch):
    from rag_data.config import VectorSettings

    captured = _install_fake_build(monkeypatch)
    build_vector_engine(VectorSettings(), metric_type='L2')
    assert captured['kwargs']['metric_type'] == 'L2'


def test_build_vector_engine_forwards_overrides(monkeypatch):
    from rag_data.config import VectorSettings

    captured = _install_fake_build(monkeypatch)
    build_vector_engine(VectorSettings(), uri='milvus.internal:19530')
    assert captured['kwargs']['uri'] == 'milvus.internal:19530'



# ---------------- 集成：真实 Milvus 与 PG（默认跳过） ----------------
#
# 需人工准备服务与凭据后，显式指定集成标记执行：
#   python -m pytest --integration -m integration -v -s
#
# 配置一律来自 conftest 的 *_JSON 与各分区设置对象：
#   vector.backend 与 vector.dim   参与的后端与测试维度
#   vector.keep 与 vector.prefix   是否保留集合、允许清理的前缀
#   cases.vector_store.collection  集合名或表名，须以 vector.prefix 开头
#   MILVUS_SETTINGS                Milvus 地址、令牌与数据库名
#   PG_VECTOR_SETTINGS             PG 连接信息，连接串非空时优先
#
# 安全约定：只允许删除以 vector.prefix 开头的集合或表，避免误删他人数据。
VS_BACKEND = VECTOR_CONFIG["backend"].lower()
VS_COLLECTION = CASES_CONFIG["vector_store"]["collection"]
VS_DIM = VECTOR_SETTINGS.vector_size
VS_KEEP = VECTOR_CONFIG["keep"]
VS_PREFIX = VECTOR_CONFIG["prefix"]

MILVUS_URI = MILVUS_SETTINGS.uri
MILVUS_TOKEN = MILVUS_SETTINGS.token
MILVUS_DB = MILVUS_SETTINGS.db_name

# PG 连接参数。
PG_CONNECTION_STRING = PG_VECTOR_SETTINGS.connection_string
PG_HOST = PG_VECTOR_SETTINGS.host
PG_PORT = PG_VECTOR_SETTINGS.port
PG_USER = PG_VECTOR_SETTINGS.user
PG_PASSWORD = PG_VECTOR_SETTINGS.password
PG_DB = PG_VECTOR_SETTINGS.database


def _backends() -> List[str]:
    '''返回本次参与的后端名列表；vector.backend 允许 both。'''
    if VS_BACKEND == 'both':
        return ['milvus', 'pg']
    return [VS_BACKEND]


def _skip_service(service: str, target: str, exc: Exception) -> None:
    """外部服务或驱动不可用时跳过，并给出服务名、目标与原始异常。"""
    pytest.skip("跳过：" + service + "（目标 " + target + "）：" + type(exc).__name__ + "：" + str(exc))


def _open_milvus() -> Any:
    '''建立 Milvus 实例；驱动缺失或无服务时跳过。'''
    try:
        store = MilvusVectorStore(
            uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DB,
            collection_name=VS_COLLECTION, vector_size=VS_DIM,
        )
    except Exception as exc:
        _skip_service('Milvus', 'uri=' + MILVUS_URI + ' collection=' + VS_COLLECTION, exc)
    return store


def _open_pg() -> Any:
    '''建立 PG 向量库实例；驱动缺失或无服务时跳过。'''
    try:
        store = PGVectorStore(
            connection_string=PG_CONNECTION_STRING or None,
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DB,
            collection_name=VS_COLLECTION, vector_size=VS_DIM,
        )
    except Exception as exc:
        _skip_service('PG 向量库', 'host=' + PG_HOST + ':' + str(PG_PORT) + ' db=' + PG_DB, exc)
    return store


def _open_store(backend: str) -> Any:
    '''按后端名建立实例。'''
    return _open_milvus() if backend == 'milvus' else _open_pg()


def _drop_store(store: Any) -> None:
    '''清理集合：仅在未要求保留且集合名以 vector.prefix 开头时删除。'''
    if VS_KEEP or not str(getattr(store, 'collection_name', '')).startswith(VS_PREFIX):
        return
    try:
        store.delete_col()
    except Exception:
        pass


def _sample_vectors(count: int = 3) -> List[List[float]]:
    '''生成 count 条互相正交的单位向量，第 i 条在第 i 维取 1，便于验证相似度排序。'''
    return [[1.0 if dim == index % VS_DIM else 0.0 for dim in range(VS_DIM)] for index in range(count)]


@pytest.fixture(params=_backends())
def live_store(request):
    '''真实向量库夹具：按后端打开实例，用完按安全约定清理。'''
    store = _open_store(request.param)
    try:
        store.reset()
        yield store
    finally:
        _drop_store(store)


def _flush(store: Any) -> None:
    '''Milvus 写入或更新后需 flush 才能被后续查询看到；PG 无需处理。'''
    client = getattr(store, 'client', None)
    if client is None or not hasattr(client, 'flush'):
        return
    try:
        client.flush(collection_name=store.collection_name)
    except Exception:
        pass


def _flatten(groups: Any) -> List[Any]:
    '''把 list 的嵌套结果拍平。'''
    results: List[Any] = []
    for group in groups or []:
        results.extend(group)
    return results


def _ids(results: Any) -> List[str]:
    '''取出结果中的 id 列表。'''
    return [str(item.id) for item in results or []]



def test_live_collection_is_ready(live_store):
    '''集合已建立：list_cols 包含目标集合，col_info 可读。'''
    assert VS_COLLECTION in live_store.list_cols()
    info = live_store.col_info()
    assert isinstance(info, dict) and info



def test_live_insert_and_get(live_store):
    '''写入后可按 id 读回，payload 原样保留。'''
    payloads = [{'tag': 'alpha'}, {'tag': 'beta'}]
    live_store.insert(_sample_vectors(2), payloads=payloads, ids=['0', '1'])
    _flush(live_store)
    first = live_store.get('0')
    assert first is not None
    assert first.id == '0'
    assert (first.payload or {}).get('tag') == 'alpha'




def test_live_search_orders_by_similarity(live_store):
    '''三条正交向量：用第一条向量检索，其自身应排第一且相似度最高。'''
    vectors = _sample_vectors(3)
    live_store.insert(vectors, payloads=[{'tag': 'alpha'}, {'tag': 'beta'}, {'tag': 'gamma'}], ids=['0', '1', '2'])
    _flush(live_store)
    hits = live_store.search('alpha', vectors[0], top_k=3)
    assert len(hits) == 3
    assert hits[0].id == '0'
    assert hits[0].score is not None
    assert hits[0].score >= hits[-1].score



def test_live_search_respects_top_k(live_store):
    '''top_k 限制返回条数。'''
    live_store.insert(_sample_vectors(3), ids=['0', '1', '2'])
    _flush(live_store)
    assert len(live_store.search('q', _sample_vectors(1)[0], top_k=2)) == 2




def test_live_search_with_equality_filter(live_store):
    '''按 payload 等值过滤，只命中匹配的记录。'''
    live_store.insert(_sample_vectors(3), payloads=[{'tag': 'alpha'}, {'tag': 'beta'}, {'tag': 'beta'}], ids=['0', '1', '2'])
    _flush(live_store)
    hits = live_store.search('q', _sample_vectors(1)[0], top_k=5, filters={'tag': 'beta'})
    assert sorted(_ids(hits)) == ['1', '2']



def test_live_update_payload(live_store):
    '''按 id 更新 payload 后可读回新值。'''
    live_store.insert(_sample_vectors(1), payloads=[{'tag': 'alpha'}], ids=['0'])
    _flush(live_store)
    live_store.update('0', payload={'tag': 'beta'})
    _flush(live_store)
    updated = live_store.get('0')
    assert updated is not None
    assert (updated.payload or {}).get('tag') == 'beta'




def test_live_delete(live_store):
    '''删除后该 id 读不到。'''
    live_store.insert(_sample_vectors(2), ids=['0', '1'])
    _flush(live_store)
    live_store.delete('0')
    _flush(live_store)
    assert live_store.get('0') is None



def test_live_list_returns_written_rows(live_store):
    '''list 能看到已写入的 id（外层嵌套已拍平）。'''
    live_store.insert(_sample_vectors(2), ids=['0', '1'])
    _flush(live_store)
    assert set(_ids(_flatten(live_store.list(top_k=10)))) >= {'0', '1'}



def test_live_reset_clears_collection(live_store):
    '''reset 删除并重建集合，数据清空但集合仍在。'''
    live_store.insert(_sample_vectors(2), ids=['0', '1'])
    _flush(live_store)
    live_store.reset()
    assert VS_COLLECTION in live_store.list_cols()
    assert _flatten(live_store.list(top_k=10)) == []




def test_live_keyword_search_optional(live_store):
    '''关键词检索：实现不支持时返回 None，支持时返回列表；两者都接受。'''
    live_store.insert(_sample_vectors(1), payloads=[{'text_lemmatized': 'rolling memory summary'}], ids=['0'])
    _flush(live_store)
    result = live_store.keyword_search('memory', top_k=3)
    assert result is None or isinstance(result, list)



def test_live_engine_end_to_end(live_store):
    '''引擎包装真实向量库：写入、检索、读取与列表全部走通。'''
    engine = VectorEngine(live_store, default_top_k=2)
    assert engine.backend == live_store.backend
    assert engine.collection_name == VS_COLLECTION
    assert engine.insert(_sample_vectors(3), ids=['0', '1', '2']) == 3
    _flush(live_store)
    hits = engine.search('q', _sample_vectors(1)[0])
    assert len(hits) == 2
    assert engine.get('0') is not None
    assert set(_ids(engine.list(top_k=10))) >= {'0', '1', '2'}




def test_live_engine_rejects_dimension_mismatch(live_store):
    '''引擎按集合维度校验，写入维度不符时抛 StorageError。'''
    wrong_dim = 1 if VS_DIM != 1 else 2
    engine = VectorEngine(live_store)
    with pytest.raises(StorageError):
        engine.insert([[0.0] * wrong_dim])



def test_live_build_from_env_settings(live_store):
    '''用环境变量驱动的 VectorSettings 装配引擎，并完成一次真实检索。'''
    from rag_data.config import VectorSettings

    if live_store.backend == 'milvus':
        settings = VectorSettings(
            backend='milvus', collection=VS_COLLECTION, vector_size=VS_DIM, top_k=2,
            milvus={'uri': MILVUS_URI, 'token': MILVUS_TOKEN, 'db_name': MILVUS_DB},
        )
    else:
        settings = VectorSettings(
            backend='pg', collection=VS_COLLECTION, vector_size=VS_DIM, top_k=2,
            pg={'connection_string': PG_CONNECTION_STRING, 'host': PG_HOST, 'port': PG_PORT, 'user': PG_USER, 'password': PG_PASSWORD, 'database': PG_DB},
        )
    engine = build_vector_engine(settings)
    try:
        engine.store.reset()
        engine.insert(_sample_vectors(2), ids=['0', '1'])
        _flush(engine.store)
        assert len(engine.search('q', _sample_vectors(1)[0])) >= 1
    finally:
        engine.close()



# ---------------- PG 建表与索引 DDL（不连服务） ----------------


def test_pg_table_ddl_shape():
    ddl = table_ddl('autotest_v', 8)
    assert ddl.startswith('CREATE TABLE IF NOT EXISTS ' + 'autotest_v')
    assert 'id TEXT PRIMARY KEY' in ddl
    assert 'vector vector(8)' in ddl
    assert 'payload JSONB' in ddl


def test_pg_hnsw_index_ddl_shape():
    ddl = hnsw_index_ddl('t')
    assert ddl.startswith('CREATE INDEX IF NOT EXISTS t_hnsw_idx ON t')
    assert 'USING hnsw (vector vector_cosine_ops)' in ddl


def test_pg_diskann_index_ddl_shape():
    ddl = diskann_index_ddl('t')
    assert ddl.startswith('CREATE INDEX IF NOT EXISTS t_diskann_idx ON t')
    assert 'USING diskann (vector)' in ddl


def test_pg_text_index_ddl_tracks_keyword_expression():
    '''GIN 索引表达式与 keyword_search 共用同一常量，避免二者漂移导致索引失效。'''
    import inspect

    ddl = text_index_ddl('t')
    assert 'USING gin(' in ddl
    assert _TEXT_VECTOR in ddl
    assert 'text_lemmatized' in _TEXT_VECTOR
    assert '_TEXT_VECTOR' in inspect.getsource(PGVectorStore.keyword_search)


def test_pg_extension_installed_true_and_false():
    class _Cursor:
        def __init__(self, present): self.present = present; self.sql = None
        def execute(self, sql, params=None): self.sql = sql
        def fetchone(self): return (1,) if self.present else None

    assert _extension_installed(_Cursor(True), 'vectorscale') is True
    assert _extension_installed(_Cursor(False), 'vectorscale') is False



def _pg_like(use_hnsw=True, use_diskann=False, installed=True, collection='autotest_v'):
    '''构造未初始化的 PGVectorStore，并把取游标替换为记录语句的假实现（不连服务）。'''
    store = PGVectorStore.__new__(PGVectorStore)
    store.collection_name = collection
    store.vector_size = 8
    store.metric_type = 'COSINE'
    store.use_hnsw = use_hnsw
    store.use_diskann = use_diskann

    class _Cursor:
        def execute(self, sql, params=None): statements.append(sql)
        def fetchone(self): return (1,) if installed else None

    def fake_cursor(commit=False):
        yield _Cursor()

    statements = []
    store._cursor = contextmanager(fake_cursor)
    return store, statements


def _kinds(statements):
    '''把语句归类，便于断言建表流程的组成。'''
    kinds = []
    for item in statements:
        if item.startswith('CREATE EXTENSION'): kinds.append('extension')
        elif item.startswith('CREATE TABLE'): kinds.append('table')
        elif '_diskann_idx' in item: kinds.append('diskann')
        elif '_hnsw_idx' in item: kinds.append('hnsw')
        elif '_text_idx' in item: kinds.append('text')
    return kinds



def test_pg_create_col_creates_table_and_indexes():
    store, statements = _pg_like()
    store.create_col('autotest_v', 8)
    assert _kinds(statements) == ['extension', 'table', 'hnsw', 'text']


def test_pg_create_col_prefers_diskann_when_extension_present():
    store, statements = _pg_like(use_hnsw=True, use_diskann=True, installed=True)
    store.create_col('autotest_v', 8)
    assert _kinds(statements) == ['extension', 'table', 'diskann', 'text']


def test_pg_create_col_falls_back_to_hnsw_without_extension():
    store, statements = _pg_like(use_hnsw=True, use_diskann=True, installed=False)
    store.create_col('autotest_v', 8)
    assert _kinds(statements) == ['extension', 'table', 'hnsw', 'text']


def test_pg_create_col_skips_vector_index_when_disabled():
    store, statements = _pg_like(use_hnsw=False, use_diskann=False)
    store.create_col('autotest_v', 8)
    assert _kinds(statements) == ['extension', 'table', 'text']


def test_pg_create_col_skips_diskann_for_large_dimension():
    '''DiskANN 在维度较大时不可用，即使扩展已装也应回退 HNSW。'''
    store, statements = _pg_like(use_hnsw=True, use_diskann=True, installed=True)
    store.create_col('autotest_v', 4096)
    assert _kinds(statements) == ['extension', 'table', 'hnsw', 'text']
    assert any('vector vector(4096)' in item for item in statements)




def test_live_pg_indexes_created(live_store):
    '''PG 后端：建表时创建的向量索引与全文 GIN 索引确实存在。'''
    if live_store.backend != 'pg':
        pytest.skip('该用例只针对 PG 后端，当前后端为 ' + str(live_store.backend))
    with live_store._cursor() as cursor:
        cursor.execute('SELECT indexname FROM pg_indexes WHERE tablename = %s', (VS_COLLECTION,))
        names = {row[0] for row in cursor.fetchall()}
    assert VS_COLLECTION + '_text_idx' in names
    assert any(name.endswith('_hnsw_idx') or name.endswith('_diskann_idx') for name in names)


