# PostgreSQL 实现：优先 psycopg3 驱动，缺失时回退 psycopg2。
#
# 缺库时仅在实例化时报错，模块本身可正常导入。

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from rag_data.exceptions import OptionalDependencyError
from rag_data.storage.db.db_engine import SQLDatabase

try:
    import psycopg
    _PG_DRIVER = "psycopg"
    _PG_ERROR: Optional[Exception] = None
except ImportError as _exc:
    psycopg = None  # type: ignore[assignment]
    try:
        import psycopg2  # type: ignore[no-redef]
        _PG_DRIVER = "psycopg2"
        _PG_ERROR = None
    except ImportError as _exc2:
        psycopg2 = None  # type: ignore[assignment]
        _PG_DRIVER = ""
        _PG_ERROR = _exc2


class PGDB(SQLDatabase):
    """基于 PostgreSQL 的实现。"""

    backend = "pg"

    def __init__(self, connection_string: Optional[str] = None, database: str = "postgres", user: str = "postgres", password: str = "", host: str = "localhost", port: int = 5432, logger: Optional[Any] = None) -> None:
        if not _PG_DRIVER:
            raise OptionalDependencyError("使用 PG 数据库需要 psycopg 或 psycopg2") from _PG_ERROR
        self.connection_string = connection_string
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        super().__init__(logger=logger)

    def _open_engine(self) -> Engine:
        "优先使用显式连接串，否则用离散参数构造 URL。"
        if self.connection_string:
            return create_engine(self.connection_string, future=True)
        url = URL.create("postgresql+" + _PG_DRIVER, username=self.user, password=self.password, host=self.host, port=self.port, database=self.database)
        return create_engine(url, future=True)

