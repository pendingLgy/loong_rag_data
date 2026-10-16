# 数据库表结构：用 SQLAlchemy Core 定义，方言差异由引擎自动适配。
#
# 时间列以 ISO 字符串存储，字典序即时间序，省去时区与类型转换。

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, Index, MetaData, SmallInteger, String, Table, Text

# 元数据与两张表；建表 DDL 由 SQLAlchemy 按所用方言生成。
METADATA = MetaData()

HISTORY_TABLE = Table(
    "history",
    METADATA,
    Column("id", BigInteger, primary_key=True),
    Column("memory_id", BigInteger, nullable=True),
    Column("old_memory", Text, nullable=True),
    Column("new_memory", Text, nullable=True),
    Column("event", String(32), nullable=True),
    Column("created_at", String(64), nullable=True),
    Column("updated_at", String(64), nullable=True),
    Column("is_deleted", SmallInteger, nullable=False, default=0),
    Column("actor_id", String(64), nullable=True),
    Column("role", String(32), nullable=True),
    Index("ix_history_memory_id", "memory_id"),
)

MESSAGES_TABLE = Table(
    "messages",
    METADATA,
    Column("id", BigInteger, primary_key=True),
    Column("session_scope", String(191), nullable=True),
    Column("role", String(32), nullable=True),
    Column("content", Text, nullable=True),
    Column("name", String(191), nullable=True),
    Column("created_at", String(64), nullable=True),
    Index("ix_messages_scope_created", "session_scope", "created_at"),
)

# 列名清单，供外部核对。
HISTORY_COLUMNS: List[str] = [column.name for column in HISTORY_TABLE.columns]
MESSAGE_COLUMNS: List[str] = [column.name for column in MESSAGES_TABLE.columns]


# ---------------- 建表 SQL（按方言书写，供人工建表或核对） ----------------

# 运行期由 METADATA.create_all 建表；下列 SQL 与其等价，可直接交给 DBA 执行。
# 多语句以分号分隔，MySQL 版把索引内联进建表语句。

SQLITE_CREATE_SQL: str = """CREATE TABLE IF NOT EXISTS history (
    id BIGINT NOT NULL PRIMARY KEY,
    memory_id BIGINT,
    old_memory TEXT,
    new_memory TEXT,
    event VARCHAR(32),
    created_at VARCHAR(64),
    updated_at VARCHAR(64),
    is_deleted SMALLINT NOT NULL DEFAULT 0,
    actor_id VARCHAR(64),
    role VARCHAR(32)
);

CREATE INDEX IF NOT EXISTS ix_history_memory_id ON history (memory_id);

CREATE TABLE IF NOT EXISTS messages (
    id BIGINT NOT NULL PRIMARY KEY,
    session_scope VARCHAR(191),
    role VARCHAR(32),
    content TEXT,
    name VARCHAR(191),
    created_at VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS ix_messages_scope_created ON messages (session_scope, created_at);"""

MYSQL_CREATE_SQL: str = """CREATE TABLE IF NOT EXISTS history (
    id BIGINT NOT NULL PRIMARY KEY,
    memory_id BIGINT,
    old_memory TEXT,
    new_memory TEXT,
    event VARCHAR(32),
    created_at VARCHAR(64),
    updated_at VARCHAR(64),
    is_deleted SMALLINT NOT NULL DEFAULT 0,
    actor_id VARCHAR(64),
    role VARCHAR(32),
    INDEX ix_history_memory_id (memory_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS messages (
    id BIGINT NOT NULL PRIMARY KEY,
    session_scope VARCHAR(191),
    role VARCHAR(32),
    content TEXT,
    name VARCHAR(191),
    created_at VARCHAR(64),
    INDEX ix_messages_scope_created (session_scope, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

PG_CREATE_SQL: str = """CREATE TABLE IF NOT EXISTS history (
    id BIGINT NOT NULL PRIMARY KEY,
    memory_id BIGINT,
    old_memory TEXT,
    new_memory TEXT,
    event VARCHAR(32),
    created_at VARCHAR(64),
    updated_at VARCHAR(64),
    is_deleted SMALLINT NOT NULL DEFAULT 0,
    actor_id VARCHAR(64),
    role VARCHAR(32)
);

CREATE INDEX IF NOT EXISTS ix_history_memory_id ON history (memory_id);

CREATE TABLE IF NOT EXISTS messages (
    id BIGINT NOT NULL PRIMARY KEY,
    session_scope VARCHAR(191),
    role VARCHAR(32),
    content TEXT,
    name VARCHAR(191),
    created_at VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS ix_messages_scope_created ON messages (session_scope, created_at);"""

# 方言名到建表 SQL 的映射。
CREATE_SQL = {
    "sqlite": SQLITE_CREATE_SQL,
    "mysql": MYSQL_CREATE_SQL,
    "pg": PG_CREATE_SQL,
}


def create_table_sql(backend: str) -> str:
    """按方言返回建表 SQL；未登记的方言回落到 PostgreSQL。"""
    return CREATE_SQL.get(backend, PG_CREATE_SQL)


# ---------------- 数据对象 ----------------


class HistoryRecord(BaseModel):
    """一条历史记录：变更前后内容、事件类型与操作者。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    memory_id: Optional[int] = None
    old_memory: Optional[str] = None
    new_memory: Optional[str] = None
    event: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_deleted: bool = False
    actor_id: Optional[str] = None
    role: Optional[str] = None


class Message(BaseModel):
    """一条会话消息：角色、内容与所属会话。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    session_scope: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    created_at: Optional[str] = None

