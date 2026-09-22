# 标准库 logging 的格式化器：人类可读与 JSON 两种。

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from rag_data.logging.stdlib_adapter import RAG_FIELDS_ATTR


class HumanFormatter(logging.Formatter):
    """人类可读格式，按需追加业务字段。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields: Dict[str, Any] = getattr(record, RAG_FIELDS_ATTR, None) or {}
        if not fields:
            return base
        extra = " ".join(str(key) + "=" + str(value) for key, value in fields.items())
        return base + " | " + extra


class JsonFormatter(logging.Formatter):
    """单行 JSON 格式，便于日志采集。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields: Dict[str, Any] = getattr(record, RAG_FIELDS_ATTR, None) or {}
        payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
