# 关系型数据库存储：抽象接口、后端注册表与内置实现。

from rag_data.storage.db.base import BaseDB
from rag_data.storage.db.registry_db import (
    BUILTIN_DBS,
    available_dbs,
    is_db_registered,
    register_db,
    register_db_provider,
    resolve_db,
)
from rag_data.storage.db.db_engine import build_database
from rag_data.storage.db.schema import CREATE_SQL, HistoryRecord, Message, create_table_sql
from rag_data.storage.db.snowflake import SnowflakeGenerator, next_id

__all__ = [
    "BaseDB",
    "build_database",
    "HistoryRecord",
    "Message",
    "CREATE_SQL",
    "create_table_sql",
    "SnowflakeGenerator",
    "next_id",
    "register_db",
    "register_db_provider",
    "available_dbs",
    "is_db_registered",
    "resolve_db",
    "BUILTIN_DBS",
]

