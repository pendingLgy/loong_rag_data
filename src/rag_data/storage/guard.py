# 删表与删集合的前缀守卫：只放行集成测试表，避免误删生产数据。
#
# 前缀规则集中在此模块，db 与 vector_store 共用同一套判断，避免各实现层各写一份。

from __future__ import annotations

import re
from typing import Iterable, List

from rag_data.exceptions import StorageError

# 集成测试表前缀：只有以此开头的表或集合允许被删除。
TEST_TABLE_PREFIX = "autotest_"

# 前缀之外只允许 ASCII 字母、数字与下划线：删除语句按标识符直接拼接，须先约束字符集。
_TEST_NAME = re.compile(re.escape(TEST_TABLE_PREFIX) + r"[A-Za-z0-9_]+\Z")


def is_test_table(name: str) -> bool:
    """判断是否为允许删除的集成测试表：前缀匹配且满足标识符字符集。"""
    return isinstance(name, str) and _TEST_NAME.match(name) is not None


def ensure_test_table(name: str) -> str:
    """校验单个表名；不满足前缀或字符集约束时抛 StorageError。"""
    if not is_test_table(name):
        raise StorageError(
            "拒绝删除非集成测试表：" + str(name) + "；只有以 " + TEST_TABLE_PREFIX + " 开头的表允许删除"
        )
    return name


def ensure_test_tables(names: Iterable[str]) -> List[str]:
    """批量前置校验：任一表名不合法即整体拒绝，避免只删掉一部分。"""
    targets = [str(name) for name in names]
    for name in targets:
        ensure_test_table(name)
    return targets

