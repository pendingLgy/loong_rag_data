# 测试公共夹具：确保 src 布局可导入，并提供常用夹具。
import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rag_data.config import Settings  # noqa: E402
from rag_data.logging.factory import configure_logging  # noqa: E402


@pytest.fixture()
def settings():
    return Settings()


@pytest.fixture()
def logger(settings):
    return configure_logging(settings)

