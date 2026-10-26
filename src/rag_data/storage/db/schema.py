# 数据库表结构：用 SQLAlchemy Core 定义，方言差异由引擎自动适配。
#
# 时间列以 ISO 字符串存储，字典序即时间序，省去时区与类型转换。

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import BigInteger, Column, Index, MetaData, SmallInteger, String, Table, Text, TIMESTAMP

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
    Column("created_at_tz", TIMESTAMP(timezone=True), nullable=True),
    Column("updated_at_tz", TIMESTAMP(timezone=True), nullable=True),
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
    Column("created_at_tz", TIMESTAMP(timezone=True), nullable=True),
    Index("ix_messages_scope_created", "session_scope", "created_at_tz"),
)

# 列名清单，供外部核对。
HISTORY_COLUMNS: List[str] = [column.name for column in HISTORY_TABLE.columns]
MESSAGE_COLUMNS: List[str] = [column.name for column in MESSAGES_TABLE.columns]


# ---------------- 数据对象 ----------------

class HistoryRecord(BaseModel):
    """一条历史记录：变更前后内容、事件类型与操作者。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    memory_id: Optional[int] = None
    old_memory: Optional[str] = None
    new_memory: Optional[str] = None
    event: Optional[str] = None
    created_at_tz: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at_tz: Optional[datetime] = None
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
    created_at_tz: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
