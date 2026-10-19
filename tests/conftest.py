# 测试公共夹具：确认 src 布局可导入，并提供常用夹具。
import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rag_data.config import RagSetting  # noqa: E402


@pytest.fixture()
def settings():
    return RagSetting()


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="运行真实调用外部服务的集成测试",
    )
