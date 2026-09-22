# 日志适配层的单元测试。

import logging

from rag_data.logging.base import LogContext
from rag_data.logging.stdlib_adapter import RAG_FIELDS_ATTR, StdlibAdapter


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _make_adapter(name):
    std_logger = logging.getLogger(name)
    std_logger.setLevel(logging.DEBUG)
    for handler in list(std_logger.handlers):
        std_logger.removeHandler(handler)
    capture = _CaptureHandler()
    std_logger.addHandler(capture)
    std_logger.propagate = False
    return StdlibAdapter(std_logger), capture


def test_levels_are_forwarded():
    adapter, capture = _make_adapter("rag_data.test.levels")
    adapter.debug("d")
    adapter.info("i")
    adapter.warning("w")
    adapter.error("e")
    assert [record.levelname for record in capture.records] == ["DEBUG", "INFO", "WARNING", "ERROR"]


def test_bind_merges_fields_with_call_fields():
    adapter, capture = _make_adapter("rag_data.test.bind")
    adapter.bind(user_id="u1").info("hello", chunk=2)
    fields = getattr(capture.records[0], RAG_FIELDS_ATTR)
    assert fields["user_id"] == "u1"
    assert fields["chunk"] == 2


def test_bind_does_not_mutate_original():
    adapter, capture = _make_adapter("rag_data.test.immutable")
    adapter.bind(user_id="u1")
    adapter.info("plain")
    assert getattr(capture.records[0], RAG_FIELDS_ATTR) == {}


def test_context_binds_within_block_only():
    adapter, capture = _make_adapter("rag_data.test.context")
    with adapter.context(scope="ctx") as scoped:
        assert isinstance(scoped, StdlibAdapter)
        scoped.info("inside")
    adapter.info("outside")
    assert getattr(capture.records[0], RAG_FIELDS_ATTR)["scope"] == "ctx"
    assert getattr(capture.records[1], RAG_FIELDS_ATTR) == {}


def test_context_returns_context_manager():
    adapter, _ = _make_adapter("rag_data.test.ctx_type")
    assert isinstance(adapter.context(), LogContext)


def test_configure_logging_returns_adapter(logger):
    logger.info("smoke", stage="test")
