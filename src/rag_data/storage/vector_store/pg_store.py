# PostgreSQL（pgvector）向量库实现：表即集合，向量列配合 jsonb 元数据。
# 优先使用 psycopg3，缺失时回退 psycopg2；缺库时仅在实例化时报错。

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from rag_data.exceptions import OptionalDependencyError, StorageError
from rag_data.storage.guard import ensure_test_table
from rag_data.storage.vector_store.base import MetricType, VectorStoreBase, VectorStoreResult

try:
    import psycopg
    _PSYCOPG_VERSION = 3
    _PSYCOPG_ERROR: Optional[Exception] = None
except ImportError as _exc:
    psycopg = None  # type: ignore[assignment]
    try:
        import psycopg2  # type: ignore[no-redef]
        _PSYCOPG_VERSION = 2
        _PSYCOPG_ERROR = None
    except ImportError as _exc2:
        psycopg2 = None  # type: ignore[assignment]
        _PSYCOPG_VERSION = 0
        _PSYCOPG_ERROR = _exc2

# -------------------- 具名常量定义 --------------------

# DiskANN 索引仅在维度较小且已安装 vectorscale 扩展时使用。
DISKANN_MAX_DIMENSION = 2000

# 全文检索表达式：直接使用标准 SQL 字符串，保证 GIN 索引与 keyword_search 完美契合命中索引。
_TEXT_VECTOR = "to_tsvector('simple', payload->>'text_lemmatized')"

# 支持的操作符到 SQL 片段与是否数值化的映射。
OPERATOR_SQL_MAP: Dict[str, Any] = {
    "eq": ('payload->>%s = %s', False),
    "ne": ('payload->>%s != %s', False),
    "gt": ('(payload->>%s)::numeric > %s', True),
    "gte": ('(payload->>%s)::numeric >= %s', True),
    "lt": ('(payload->>%s)::numeric < %s', True),
    "lte": ('(payload->>%s)::numeric <= %s', True),
    "in": ('payload->>%s = ANY(%s)', False),
    "nin": ('NOT (payload->>%s = ANY(%s))', False),
    "contains": ('payload->>%s LIKE %s', False),
    "icontains": ('payload->>%s ILIKE %s', False),
}



def _build_filter_conditions(filters: Optional[Dict[str, Any]]) -> Any:
    "把过滤字典翻译为 SQL WHERE 片段与参数列表；支持 $or、$not 与操作符写法。"
    conditions: List[str] = []
    params: List[Any] = []
    if not filters:
        return conditions, params
    for key, value in filters.items():
        if key == "$or":
            groups = []
            for item in value:
                sub_conditions, sub_params = _build_filter_conditions(item)
                if sub_conditions:
                    groups.append('(' + ' AND '.join(sub_conditions) + ')')
                    params.extend(sub_params)
            if groups:
                conditions.append('(' + ' OR '.join(groups) + ')')
            continue
        if key == "$not":
            groups = []
            for item in value:
                sub_conditions, sub_params = _build_filter_conditions(item)
                if sub_conditions:
                    groups.append('(' + ' AND '.join(sub_conditions) + ')')
                    params.extend(sub_params)
            if groups:
                conditions.append('NOT (' + ' OR '.join(groups) + ')')
            continue
        if value == "*":
            conditions.append('payload ? %s')
            params.append(key)
            continue
        if isinstance(value, dict):
            for op, op_value in value.items():
                if op not in OPERATOR_SQL_MAP:
                    raise StorageError('不支持的过滤操作符：' + str(op))
                template, is_numeric = OPERATOR_SQL_MAP[op]
                if op in ('in', 'nin'):
                    if not isinstance(op_value, list):
                        raise StorageError('操作符需要列表值：' + str(op))
                    conditions.append(template)
                    params.extend([key, [str(v) for v in op_value]])
                else:
                    conditions.append(template)
                    params.extend([key, float(op_value) if is_numeric else op_value])
        elif isinstance(value, list):
            conditions.append('payload->>%s = ANY(%s)')
            params.extend([key, [str(v) for v in value]])
        else:
            conditions.append('payload->>%s = %s')
            params.extend([key, json.dumps(value) if isinstance(value, bool) else value])
    return conditions, params


def table_ddl(name: str, vector_size: int) -> str:
    """建表语句：id 主键、vector 向量列、payload 存 jsonb 元数据。"""
    return "CREATE TABLE IF NOT EXISTS " + name + " (id TEXT PRIMARY KEY, vector vector(" + str(vector_size) + "), payload JSONB)"


def hnsw_index_ddl(name: str) -> str:
    """HNSW 向量索引（余弦距离）。"""
    return "CREATE INDEX IF NOT EXISTS " + name + "_hnsw_idx ON " + name + " USING hnsw (vector vector_cosine_ops)"


def diskann_index_ddl(name: str) -> str:
    """DiskANN 向量索引；需 vectorscale 扩展，缺失时不可执行。"""
    return "CREATE INDEX IF NOT EXISTS " + name + "_diskann_idx ON " + name + " USING diskann (vector)"


def text_index_ddl(name: str) -> str:
    """全文检索 GIN 索引；与 keyword_search 共用同一表达式。"""
    return "CREATE INDEX IF NOT EXISTS " + name + "_text_idx ON " + name + " USING gin(" + _TEXT_VECTOR + ")"


def _extension_installed(cursor: Any, extname: str) -> bool:
    """查询扩展是否已安装，用于判断 DiskANN 索引是否可用。"""
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (extname,))
    return cursor.fetchone() is not None


class PGVectorStore(VectorStoreBase):
    "基于 PostgreSQL 与 pgvector 的向量库实现；表即集合。"

    backend = "pg"
    metric_type = "COSINE"

    def __init__(
        self,
        connection_string: Optional[str] = None,
        collection_name: str = "rag_data",
        vector_size: int = 1024,
        metric_type: MetricType = "COSINE",
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        host: str = "localhost",
        port: int = 5432,
        use_hnsw: bool = True,
        use_diskann: bool = False,
    ) -> None:
        if _PSYCOPG_VERSION == 0:
            raise OptionalDependencyError('使用 PG 向量库需要 psycopg 或 psycopg2，请安装 psycopg[binary]>=3.1') from _PSYCOPG_ERROR
        if connection_string:
            self.connection_string = connection_string
        else:
            # self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self.connection_string = (
                "postgresql://{user}:{password}@{host}:{port}/{database}"
            ).format(user=user, password=password, host=host, port=port, database=database)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.metric_type = metric_type
        self.use_hnsw = use_hnsw
        self.use_diskann = use_diskann
        self._conn: Optional[Any] = None
        self.create_col(collection_name, vector_size, metric_type)

    # ---------------- 连接与游标 ----------------

    def _connection(self) -> Any:
        if self._conn is None or getattr(self._conn, "closed", False):
            try:
                if _PSYCOPG_VERSION == 3:
                    self._conn = psycopg.connect(self.connection_string)
                else:
                    self._conn = psycopg2.connect(self.connection_string)
            except Exception as exc:
                raise StorageError('无法连接 PostgreSQL：' + str(exc)) from exc
        return self._conn

    @contextmanager
    def _cursor(self, commit: bool = False) -> Iterator[Any]:
        conn = self._connection()
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _col(self) -> str:
        "返回经校验的安全表名（仅允许字母、数字与下划线）。"
        if not self.collection_name.replace('_', '').isalnum():
            raise StorageError('非法集合名：' + str(self.collection_name))
        return self.collection_name


    # ---------------- 集合管理 ----------------

    def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        """建表并初始化向量索引与全文索引；重复调用安全。"""
        safe_name = self._col()
        with self._cursor(commit=True) as cursor:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cursor.execute(table_ddl(safe_name, vector_size))
            if self.use_diskann and vector_size < DISKANN_MAX_DIMENSION and _extension_installed(cursor, 'vectorscale'):
                cursor.execute(diskann_index_ddl(safe_name))
            elif self.use_hnsw:
                cursor.execute(hnsw_index_ddl(safe_name))
            cursor.execute(text_index_ddl(safe_name))

    def list_cols(self) -> List[str]:
        with self._cursor() as cursor:
            cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()')
            return [row[0] for row in cursor.fetchall()]

    def delete_col(self) -> None:
        target = ensure_test_table(self._col())
        with self._cursor(commit=True) as cursor:
            cursor.execute('DROP TABLE IF EXISTS ' + target)

    def col_info(self) -> Dict[str, Any]:
        count_sql = 'SELECT COUNT(*) FROM ' + self._col()
        size_sql = 'SELECT pg_size_pretty(pg_total_relation_size(to_regclass(%s)))'
        with self._cursor() as cursor:
            cursor.execute(count_sql)
            count = cursor.fetchone()[0]
            cursor.execute(size_sql, (self._col(),))
            size = cursor.fetchone()[0]
        return {"name": self.collection_name, "count": count, "size": size}

    def reset(self) -> None:
        "删除并重建当前集合。"
        self.delete_col()
        self.create_col(self.collection_name, self.vector_size, self.metric_type)

    # ---------------- 写入与检索 ----------------

    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        "批量写入；ids 缺省时用行号字符串补齐。"
        payloads = payloads or [None] * len(vectors)  # type: ignore[list-item]
        ids = [str(rid) for rid in (ids or range(len(vectors)))]
        rows = [(rid, vector, json.dumps(payload)) for rid, vector, payload in zip(ids, vectors, payloads)]
        sql = 'INSERT INTO ' + self._col() + ' (id, vector, payload) VALUES (%s, %s::vector, %s::jsonb)'
        with self._cursor(commit=True) as cursor:
            cursor.executemany(sql, rows)


    def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[VectorStoreResult]:
        conditions, params = _build_filter_conditions(filters)
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        sql = 'SELECT id, vector <=> %s::vector AS distance, payload FROM ' + self._col() + where + ' ORDER BY distance LIMIT %s'
        with self._cursor() as cursor:
            cursor.execute(sql, (vectors, *params, top_k))
            rows = cursor.fetchall()
        return [VectorStoreResult(id=str(row[0]), score=max(0.0, 1.0 - float(row[1])), payload=row[2]) for row in rows]

    def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        "基于 text_lemmatized 的全文检索；失败返回 None。"
        _ts = _TEXT_VECTOR
        conditions, params = _build_filter_conditions(filters)
        extra = ' AND ' + ' AND '.join(conditions) if conditions else ''
        query_expression = query.replace(" ", " | ")
        sql = (
            'SELECT id, ts_rank_cd(' + _ts + ", to_tsquery('simple', %s)) AS score, payload FROM " + self._col()
            + ' WHERE ' + _ts + " @@ to_tsquery('simple', %s)" + extra + ' ORDER BY score DESC LIMIT %s'
        )
        try:
            with self._cursor() as cursor:
                cursor.execute(sql, (query_expression, query_expression, *params, top_k))
                rows = cursor.fetchall()
        except Exception:
            return None
        return [VectorStoreResult(id=str(row[0]), score=float(row[1]), payload=row[2]) for row in rows]


    def delete(self, vector_id: str) -> None:
        with self._cursor(commit=True) as cursor:
            cursor.execute('DELETE FROM ' + self._col() + ' WHERE id = %s', (vector_id,))

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        "按 id 更新；缺省字段保持不变。"
        table = self._col()
        with self._cursor(commit=True) as cursor:
            if vector is not None:
                cursor.execute('UPDATE ' + table + ' SET vector = %s::vector WHERE id = %s', (vector, vector_id))
            if payload is not None:
                cursor.execute('UPDATE ' + table + ' SET payload = %s::jsonb WHERE id = %s', (json.dumps(payload), vector_id))

    def get(self, vector_id: str) -> Optional[VectorStoreResult]:
        with self._cursor() as cursor:
            cursor.execute('SELECT id, payload FROM ' + self._col() + ' WHERE id = %s', (vector_id,))
            row = cursor.fetchone()
        if not row:
            return None
        return VectorStoreResult(id=str(row[0]), score=None, payload=row[1])

    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[List[VectorStoreResult]]:
        conditions, params = _build_filter_conditions(filters)
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        sql = 'SELECT id, payload FROM ' + self._col() + where + ' LIMIT %s'
        with self._cursor() as cursor:
            cursor.execute(sql, (*params, top_k or 100))
            rows = cursor.fetchall()
        results = [VectorStoreResult(id=str(row[0]), score=None, payload=row[1]) for row in rows]
        return [results]

