# SQLite 实现：用 SQLAlchemy 引擎连接标准库 sqlite3 驱动。
#
# 数据一律落盘：库文件路径必须由构造参数显式指定，不提供内存库模式。

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from rag_data.storage.db.db_engine import SQLDatabase


class SQLiteDB(SQLDatabase):
    """基于 SQLite 的实现；db_path 为必填参数，指向库文件路径。"""

    backend = 'sqlite'

    def __init__(self, db_path: str) -> None:
        if not db_path:
            raise ValueError('db_path 不能为空')
        self.db_path = db_path
        self._ensure_parent(db_path)
        super().__init__()

    @staticmethod
    def _ensure_parent(path: str) -> None:
        """确保库文件所在目录存在。"""
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

    def _open_engine(self) -> Engine:
        """用 URL.create 构造连接串；关闭线程检查以便跨线程复用同一连接。"""
        args = {"check_same_thread": False}
        url = URL.create("sqlite", database=self.db_path)
        return create_engine(url, connect_args=args, future=True)

