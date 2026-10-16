# 关系型数据库通用实现：基于 SQLAlchemy Core，方言差异由引擎适配。

from __future__ import annotations

import threading
from abc import abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import delete, inspect, insert, select
from sqlalchemy.engine import Engine

from rag_data.exceptions import ConfigError, StorageError
from rag_data.storage.db.base import BaseDB
from rag_data.storage.db.schema import HISTORY_TABLE, MESSAGES_TABLE, METADATA, HistoryRecord, Message
from rag_data.storage.db.snowflake import next_id

if TYPE_CHECKING:  # pragma: no cover
    from rag_data.config import DBSettings
    from rag_data.logging.base import LoggerAdapter


class SQLDatabase(BaseDB):
    """基于 SQLAlchemy Core 的通用实现：单引擎加互斥锁，方言差异由子类提供连接串。"""

    def __init__(self, logger: Optional[Any] = None) -> None:
        super().__init__(logger)
        self._lock = threading.Lock()
        self.engine: Optional[Engine] = None
        self.engine = self._open_engine()
        self._ensure_schema()

    @abstractmethod
    def _open_engine(self) -> Engine:
        """创建并返回 SQLAlchemy 引擎。"""

    def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表。"""
        inspector = inspect(self.engine)
        return [table for table in METADATA.sorted_tables if not inspector.has_table(table.name)]

    def _ensure_schema(self) -> None:
        """检查表是否存在，缺失的才创建；全部存在时不做任何写操作。"""
        try:
            missing = self._missing_tables()
            if not missing:
                return
            for table in missing:
                table.create(self.engine, checkfirst=True)
        except Exception as exc:
            raise StorageError("建表失败" + ": " + str(exc)) from exc

    # ---------------- 历史记录 ----------------

    def add_history(self, record: HistoryRecord) -> HistoryRecord:
        "追加一条历史记录，返回补全 id 后的对象。"
        row = self._record_row(record)
        with self._lock, self.engine.begin() as conn:
            conn.execute(insert(HISTORY_TABLE).values(**row))
        return HistoryRecord(**row)

    def batch_add_history(self, records: List[HistoryRecord]) -> List[HistoryRecord]:
        "批量追加历史记录，返回补全 id 后的对象列表。"
        rows = [self._record_row(record) for record in records]
        if not rows:
            return []
        with self._lock, self.engine.begin() as conn:
            conn.execute(insert(HISTORY_TABLE), rows)
        return [HistoryRecord(**row) for row in rows]

    def get_history(self, memory_id: int) -> List[HistoryRecord]:
        "按 memory_id 取历史，按时间升序。"
        table = HISTORY_TABLE
        stmt = (
            select(table)
            .where(table.c.memory_id == memory_id)
            .order_by(table.c.created_at.asc(), table.c.updated_at.asc())
        )
        with self._lock, self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._to_record(dict(row)) for row in rows]

    @staticmethod
    def _record_row(record: HistoryRecord) -> Dict[str, Any]:
        "把历史记录对象转换为可写入的行字典；id 缺省时生成雪花 ID。"
        return {
            "id": record.id or next_id(),
            "memory_id": record.memory_id,
            "old_memory": record.old_memory,
            "new_memory": record.new_memory,
            "event": record.event,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "is_deleted": int(record.is_deleted),
            "actor_id": record.actor_id,
            "role": record.role,
        }

    @staticmethod
    def _to_record(row: Dict[str, Any]) -> HistoryRecord:
        "把数据库行转换为历史记录对象。"
        return HistoryRecord(
            id=row.get("id"),
            memory_id=row.get("memory_id"),
            old_memory=row.get("old_memory"),
            new_memory=row.get("new_memory"),
            event=row.get("event"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            is_deleted=bool(row.get("is_deleted")),
            actor_id=row.get("actor_id"),
            role=row.get("role"),
        )


    # ---------------- 会话消息 ----------------

    def save_messages(self, messages: List[Message], keep_last: int = 10) -> List[Message]:
        "写入消息，并只保留各会话最近的 keep_last 条。"
        rows = [self._message_row(message) for message in messages]
        if not rows:
            return []
        table = MESSAGES_TABLE
        scopes = sorted({row["session_scope"] for row in rows})
        with self._lock, self.engine.begin() as conn:
            conn.execute(insert(table), rows)
            for scope in scopes:
                conn.execute(self._evict_stmt(scope, keep_last))
        return [self._to_message(row) for row in rows]

    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Message]:
        "按时间升序取会话最近 limit 条消息。"
        table = MESSAGES_TABLE
        recent = (
            select(table.c.id, table.c.session_scope, table.c.role, table.c.content, table.c.name, table.c.created_at)
            .where(table.c.session_scope == session_scope)
            .order_by(table.c.created_at.desc(),table.c.id.desc())
            .limit(limit)
            .subquery()
        )
        stmt = select(recent).order_by(recent.c.created_at.asc())
        with self._lock, self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._to_message(dict(row)) for row in rows]

    def _evict_stmt(self, session_scope: str, keep_last: int) -> Any:
        "构造删除语句：保留会话最近 keep_last 条，其余删除。"
        table = MESSAGES_TABLE
        recent = (
            select(table.c.id)
            .where(table.c.session_scope == session_scope)
            .order_by(table.c.created_at.desc())
            .limit(keep_last)
            .subquery()
        )
        return delete(table).where(table.c.session_scope == session_scope, table.c.id.notin_(select(recent.c.id)))

    @staticmethod
    def _message_row(message: Message) -> Dict[str, Any]:
        "把消息对象转换为可写入的行字典。"
        return {
            "id": message.id or next_id(),
            "session_scope": message.session_scope,
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "created_at": message.created_at or datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _to_message(row: Dict[str, Any]) -> Message:
        "把数据库行转换为消息对象。"
        return Message(
            id=row.get("id"),
            session_scope=row.get("session_scope"),
            role=row.get("role"),
            content=row.get("content"),
            name=row.get("name"),
            created_at=row.get("created_at"),
        )


    # ---------------- 维护 ----------------

    def reset(self) -> None:
        "删除并重建所有表。"
        with self._lock:
            try:
                METADATA.drop_all(self.engine)
                self._ensure_schema()
            except Exception as exc:
                raise StorageError("重置失败" + ": " + str(exc)) from exc

    def close(self) -> None:
        "释放引擎与连接池；重复调用安全。"
        with self._lock:
            if self.engine is not None:
                self.engine.dispose()
                self.engine = None

    def __enter__(self) -> "SQLDatabase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()



# 按配置装配数据库实例：按 backend 选择方言，参数取自对应方言分区。
# 方言实现按需惰性导入，避免与本模块形成导入环。


def build_database(settings: "DBSettings", logger: Optional["LoggerAdapter"] = None) -> BaseDB:
    """按 DBSettings.backend 构造对应方言的数据库实例。"""
    backend = settings.backend
    if backend == "sqlite":
        from rag_data.storage.db.sqlite_db import SQLiteDB

        path = settings.sqlite.path
        if not path:
            raise ConfigError("未配置 SQLite 库文件路径：请设置 db.sqlite.path 或环境变量 RAG_DB__SQLITE__PATH")
        return SQLiteDB(db_path=path, logger=logger)
    if backend == "mysql":
        from rag_data.storage.db.mysql_db import MySQLDB

        mysql = settings.mysql
        return MySQLDB(
            host=mysql.host,
            port=mysql.port,
            user=mysql.user,
            password=mysql.password,
            database=mysql.database,
            charset=mysql.charset,
            logger=logger,
        )
    from rag_data.storage.db.pg_db import PGDB

    pg = settings.pg
    return PGDB(
        connection_string=pg.connection_string or None,
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=pg.database,
        logger=logger,
    )

