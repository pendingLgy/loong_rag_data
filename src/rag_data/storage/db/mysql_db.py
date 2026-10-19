# MySQL 实现：优先 PyMySQL 驱动，缺失时回退 mysql-connector-python。
#
# 缺库时仅在实例化时报错，模块本身可正常导入。

from __future__ import annotations

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from rag_data.exceptions import OptionalDependencyError
from rag_data.storage.db.db_engine import SQLDatabase

try:
    import pymysql
    _MYSQL_DRIVER = "pymysql"
    _MYSQL_ERROR: Optional[Exception] = None
except ImportError as _exc:
    pymysql = None  # type: ignore[assignment]
    try:
        import mysql.connector as mysql_connector
        _MYSQL_DRIVER = "mysqlconnector"
        _MYSQL_ERROR = None
    except ImportError as _exc2:
        mysql_connector = None  # type: ignore[assignment]
        _MYSQL_DRIVER = ""
        _MYSQL_ERROR = _exc2


class MySQLDB(SQLDatabase):
    """基于 MySQL 的实现。"""

    backend = "mysql"

    def __init__(self, host: str = "localhost", port: int = 3306, user: str = "root", password: str = "", database: str = "rag_data", charset: str = "utf8mb4") -> None:
        if not _MYSQL_DRIVER:
            raise OptionalDependencyError("使用 MySQL 数据库需要 pymysql 或 mysql-connector-python") from _MYSQL_ERROR
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        super().__init__()

    def _open_engine(self) -> Engine:
        "用离散参数构造 URL，驱动名按可用库选择。"
        url = URL.create("mysql+" + _MYSQL_DRIVER, username=self.user, password=self.password, host=self.host, port=self.port, database=self.database, query={"charset": self.charset})
        return create_engine(url, future=True)

