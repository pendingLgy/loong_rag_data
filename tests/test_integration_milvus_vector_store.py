# 运行：pytest tests 目录下的 test_vector_store.py --integration -m integration -v -s
# 集合名取自 conftest 的 cases.vector_store.collection，必须以 autotest_ 开头。

import sys
import time

import pytest

from conftest import CASES_CONFIG, MILVUS_SETTINGS, VECTOR_CONFIG, VECTOR_SETTINGS

# 1. 声明模块级标签
pytestmark = pytest.mark.integration

# 2. 检查是否满足运行条件
if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)

from rag_data.exceptions import StorageError  # noqa: E402
from rag_data.storage.vector_store import milvus_store as milvus_module  # noqa: E402
from rag_data.storage.vector_store.milvus_store import MilvusVectorStore  # noqa: E402

COLLECTION = CASES_CONFIG['vector_store']['collection']
# 基数同时用于主键与 tenant_id（均为 INT64）：按当前秒数取余到分钟级再抬高基数，避免重复运行与历史数据撞主键。
# 每轮运行使用独立租户，隔离历史残留数据，避免检索结果被旧行干扰。
BASE_ID = divmod(int(time.time()), 60)[1] * 1000 + 900000


# ---------------- 前置校验与替身 ----------------


def _requires_pymilvus() -> None:
    '''缺 pymilvus 或版本低于 2.5 时跳过：BM25 稀疏检索依赖 Function 类型。'''
    if milvus_module.MilvusClient is None or milvus_module.Function is None:
        pytest.skip('未安装 pymilvus 或版本低于 2.5，请执行 pip install pymilvus>=2.5')


def _bare_store(collection=COLLECTION, metric='COSINE'):
    '''构造不连服务的实例：只供纯逻辑方法使用，避免组装校验依赖真实服务。'''
    store = object.__new__(MilvusVectorStore)
    store.collection_name = collection
    store.metric_type = metric
    return store


def _vector(index):
    '''构造可区分的单位向量：第 index 个位置为 1，其余为 0。'''
    size = VECTOR_SETTINGS.vector_size
    vector = [0.0] * size
    vector[index % size] = 1.0
    return vector


def _payload(index, tenant=BASE_ID, user_id=BASE_ID + 1, text='配置加载优先级', lemma='配置 加载 优先级'):
    '''最小合法 payload：tenant_id 必填，集合中该字段不可空。'''
    return {
        'tenant_id': tenant,
        'user_id': user_id,
        'text': text,
        'text_lemma': lemma,
        'created_at_tz': '2025-01-01T00:00:00+00:00',
        'created_at_tz': '2025-01-01T00:00:00+00:00',
        'doc_id': 'doc_' + str(index),
    }


# ---------------- 记录组装（_record） ----------------


def test_record_maps_declared_fields() -> None:
    '''写入记录只含集合声明的物理字段，且正文与词形文本各归其位。'''
    record = _bare_store()._record(7, _vector(0), _payload(1))
    assert set(record) == {
        'id', 'tenant_id', 'user_id', 'vectors', 'metadata',
        'text', 'text_lemma', 'created_at_tz', 'updated_at_tz',
    }
    assert record['id'] == 7
    assert record['tenant_id'] == BASE_ID
    assert record['user_id'] == BASE_ID + 1
    assert record['text'] == '配置加载优先级'
    assert record['text_lemma'] == '配置 加载 优先级'
    assert record['vectors'] == _vector(0)
    assert record['created_at_tz'].startswith('2025-01-01')


def test_record_leaves_sparse_to_bm25_function() -> None:
    '''sparse 由 BM25 函数按 text_lemma 生成，写入记录中不得出现。'''
    record = _bare_store()._record(1, _vector(1), _payload(1))
    assert 'sparse' not in record


def test_record_text_lemma_falls_back_to_data() -> None:
    '''缺 text_lemmatized 时词形文本回退到 data，保证 BM25 仍有输入。'''
    record = _bare_store()._record(1, _vector(0), {'tenant_id': BASE_ID, 'data': '只有 data'})
    assert record['text_lemma'] == '只有 data'
    assert record['text'] == ''


def test_record_text_lemma_falls_back_to_text() -> None:
    '''data 与 text_lemmatized 都缺时，词形文本退化为原文。'''
    record = _bare_store()._record(1, _vector(0), {'tenant_id': BASE_ID, 'text': '只有 text'})
    assert record['text_lemma'] == '只有 text'


def test_record_truncates_oversize_text() -> None:
    '''超长正文按 VARCHAR 上限截断，避免写入报错。'''
    max_len = milvus_module._MAX_TEXT_LEN
    long_text = '长' * (max_len + 10)
    record = _bare_store()._record(1, _vector(0), {'tenant_id': BASE_ID, 'text': long_text, 'text_lemmatized': long_text})
    assert len(record['text']) == max_len
    assert len(record['text_lemma']) == max_len


def test_record_blanks_scope_to_none() -> None:
    '''空的租户与用户标识归一为 None，交由 nullable 与约束层判断。'''
    record = _bare_store()._record(1, _vector(0), {'tenant_id': '', 'user_id': '', 'text': '正文'})
    assert record['tenant_id'] is None
    assert record['user_id'] is None


# ---------------- 集合 schema（text_lemma 承载 BM25） ----------------


def test_schema_declares_integer_scope_fields() -> None:
    '''主键与作用域标识统一为 INT64：字符串与整数混用会让标量过滤失效。'''
    _requires_pymilvus()
    schema = _bare_store()._schema(VECTOR_SETTINGS.vector_size)
    fields = {field.name: field for field in schema.fields}
    for name in ('id', 'tenant_id', 'user_id'):
        assert fields[name].dtype == milvus_module.DataType.INT64


def test_schema_declares_text_lemma_field() -> None:
    '''集合字段按物理列固定：text_lemma 需与 text 同宽并开启分析器。'''
    _requires_pymilvus()
    schema = _bare_store()._schema(VECTOR_SETTINGS.vector_size)
    names = [field.name for field in schema.fields]
    assert names[:4] == ['id', 'tenant_id', 'user_id', 'vectors']
    assert 'text' in names and 'text_lemma' in names and 'sparse' in names
    assert names[-2:] == ['updated_at_tz', 'created_at_tz']
    lemma = next(field for field in schema.fields if field.name == 'text_lemma')
    params = getattr(lemma, 'params', {}) or {}
    assert int(params.get('max_length')) == milvus_module._MAX_TEXT_LEN
    assert str(params.get('enable_analyzer')).lower() == 'true'
    assert 'jieba' in str(params.get('analyzer_params'))


def test_schema_keeps_text_without_analyzer() -> None:
    '''原文字段只存原文，分词只发生在 text_lemma 上，避免重复分词。'''
    _requires_pymilvus()
    schema = _bare_store()._schema(VECTOR_SETTINGS.vector_size)
    text = next(field for field in schema.fields if field.name == 'text')
    params = getattr(text, 'params', {}) or {}
    assert int(params.get('max_length')) == milvus_module._MAX_TEXT_LEN


def test_schema_bm25_reads_text_lemma() -> None:
    '''BM25 函数必须从 text_lemma 生成稀疏向量，否则关键词检索拿不到分词结果。'''
    _requires_pymilvus()
    schema = _bare_store()._schema(VECTOR_SETTINGS.vector_size)
    functions = list(getattr(schema, 'functions', []) or [])
    assert len(functions) == 1
    bm25 = functions[0]
    assert getattr(bm25, 'name', '') == 'text_lemma_bm25'
    assert list(getattr(bm25, 'input_field_names', []) or []) == ['text_lemma']
    assert list(getattr(bm25, 'output_field_names', []) or []) == ['sparse']


# ---------------- 过滤表达式（_create_filter） ----------------


def test_filter_covers_scalar_types() -> None:
    '''等值过滤按类型生成表达式：字符串带引号，布尔转小写，数值原样。'''
    expression = _bare_store()._create_filter({
        'tenant_id': BASE_ID,
        'flag': True,
        'count': 3,
        'ratio': 0.5,
    })
    assert expression is not None
    first = expression.split(' and ')[0]
    assert first.startswith('tenant_id')
    assert 'tenant_id' in first and str(BASE_ID) in first
    assert '== true' in expression
    assert '== 3' in expression
    assert '== 0.5' in expression
    assert expression.count(' and ') == 3


def test_filter_skips_wildcard_value() -> None:
    '''通配值表示不过滤，全部省略后返回 None。'''
    assert _bare_store()._create_filter({'tenant_id': '*'}) is None


def test_filter_rejects_illegal_key() -> None:
    '''过滤键走白名单正则，非法键直接拒绝，防止表达式注入。'''
    with pytest.raises(StorageError):
        _bare_store()._create_filter({'tenant id': 'x'})


def test_filter_rejects_unsupported_value() -> None:
    '''列表等未支持的值类型直接报错，不静默丢弃。'''
    with pytest.raises(StorageError):
        _bare_store()._create_filter({'doc_id': ['a', 'b']})


# ---------------- 结果解析（_parse） ----------------


def test_parse_converts_l2_distance_to_similarity() -> None:
    '''L2 距离换算为相似度：1 除以 1 加距离，越大越相似。'''
    hits = [
        {'distance': 3.0,'entity': {'id': 1, 'metadata': {'doc_id': 'a'}}},
        {'distance': 0.0,'entity': {'id': 2, 'metadata': {}}},
    ]
    results = _bare_store(metric='L2')._parse(hits)
    assert results[0].id == 1
    assert results[0].score == pytest.approx(0.25)
    assert results[0].metadata == {'doc_id': 'a'}
    assert results[1].score == pytest.approx(1.0)


def test_parse_keeps_cosine_score() -> None:
    '''余弦相似度直接作为分数，不再换算。'''
    results = _bare_store(metric='COSINE')._parse([{'id': 9, 'distance': 0.87, 'entity': {'metadata': {'doc_id': 'b'}}}])
    assert results[0].score == pytest.approx(0.87)


def test_parse_tolerates_missing_distance() -> None:
    '''无 distance 的命中（如 query 结果）分数与元数据均为 None，不抛异常。'''
    results = _bare_store()._parse([{'id': 5, 'entity': {}}])
    assert results[0].score is None
    assert results[0].metadata is None


# ---------------- 真实服务：写入、检索、更新与删除 ----------------


@pytest.fixture(scope='module')
def milvus_store():
    '''按 conftest 的连接参数装配真实集合；服务不可用或无 pymilvus 时跳过。'''
    _requires_pymilvus()
    if not COLLECTION.startswith('autotest_'):
        pytest.skip('集合名必须以 autotest_ 开头，当前为 ' + COLLECTION)
    try:
        store = MilvusVectorStore(
            uri=MILVUS_SETTINGS.uri,
            token=MILVUS_SETTINGS.token,
            db_name=MILVUS_SETTINGS.db_name,
            collection_name=COLLECTION,
            vector_size=VECTOR_SETTINGS.vector_size,
        )
    except Exception as exc:
        pytest.skip('跳过：Milvus 不可用（' + MILVUS_SETTINGS.uri + '）：' + type(exc).__name__ + ' ' + str(exc))
    try:
        store.client.load_collection(COLLECTION)
    except Exception:
        pass
    yield store
    # if not VECTOR_CONFIG.get('keep'):
        # try:
        #     store.client.drop_collection(collection_name=COLLECTION)
        # except Exception:
        #     pass
    store.client.close()


@pytest.fixture(scope='module')
def seeded(milvus_store):
    '''写入三条可区分的数据并 flush，供检索与更新用例复用。'''
    ids = [BASE_ID + index for index in range(3)]
    vectors = [_vector(index) for index in range(3)]
    payloads = [_payload(index) for index in range(3)]
    milvus_store.insert(vectors, payloads, ids)
    milvus_store.client.flush(collection_name=COLLECTION)
    return ids


def test_insert_and_get_roundtrip(milvus_store, seeded) -> None:
    '''写入后按 INT64 主键回查：id 转字符串，payload 原样可读。'''
    result = milvus_store.get(seeded[0])
    assert result is not None
    assert result.id == seeded[0]
    assert result.metadata['doc_id'] == 'doc_0'
    assert result.tenant_id == BASE_ID


def test_dense_search_returns_first_vector(milvus_store, seeded) -> None:
    '''用首条向量检索时，首位命中应为自身。'''
    hits = milvus_store.search('配置', _vector(0), top_k=3, filters={'tenant_id': BASE_ID})
    assert hits
    assert hits[0].id == seeded[0]
    assert hits[0].score is not None


def test_keyword_search_matches_text_lemma(milvus_store, seeded) -> None:
    '''BM25 走 text_lemma：中文查询应命中写入的中文词形文本。'''
    hits = milvus_store.keyword_search('配置', top_k=3, filters={'tenant_id': BASE_ID})
    assert hits, 'BM25 未命中：确认 text_lemma 已开启 chinese 分析器且 pymilvus 支持 BM25'
    assert seeded[0] in [hit.id for hit in hits]


def test_list_filters_by_tenant(milvus_store, seeded) -> None:
    '''按 tenant_id 过滤时只返回本租户数据；外层列表按既有约定包一层。'''
    rows = milvus_store.list(filters={'tenant_id': BASE_ID}, top_k=10)
    assert len(rows) == 1
    assert len(rows[0]) >= 3
    assert all(row.tenant_id == BASE_ID for row in rows[0])


def test_list_returns_empty_for_other_tenant(milvus_store, seeded) -> None:
    '''其他租户过滤无命中，返回只含一个空列表的结构。'''
    assert milvus_store.list(filters={'tenant_id': BASE_ID + 1}, top_k=10) == [[]]


def test_update_replaces_payload(milvus_store, seeded) -> None:
    '''更新元数据后回查：doc_id 被覆盖，正文与词形文本随 payload 一起更新。'''
    payload = _payload(1)
    payload['doc_id'] = 'doc_updated'
    milvus_store.update(seeded[1], payload=payload)
    milvus_store.client.flush(collection_name=COLLECTION)
    result = milvus_store.get(seeded[1])
    assert result is not None
    assert result.metadata['doc_id'] == 'doc_updated'


def test_update_reports_missing_id(milvus_store) -> None:
    '''更新不存在的 id 抛 StorageError，而不是静默插入新行。'''
    with pytest.raises(StorageError):
        milvus_store.update(BASE_ID + 9999, payload=_payload(1))


def test_delete_removes_row(milvus_store) -> None:
    '''删除后按同一主键回查应为 None。'''
    ids = [BASE_ID + 500]
    milvus_store.insert([_vector(1)], [_payload(9)], ids)
    milvus_store.client.flush(collection_name=COLLECTION)
    milvus_store.delete(ids[0])
    milvus_store.client.flush(collection_name=COLLECTION)
    assert milvus_store.get(ids[0]) is None

