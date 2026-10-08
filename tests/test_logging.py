# 日志适配层单元测试：工厂装配、时区格式化与调用点（完整路径）。

import logging
import os
import time
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from rag_data.config import Settings
from rag_data.facade import build_logger
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import LoggerFactory, _load_pytest_log_format_from_toml
from rag_data.logging.stdlib_adapter import StdlibLogAdapter, StdlibTimezoneFormatter

EPOCH = 1704164645.0  # 2024-01-02 03:04:05 UTC
PLUS8 = timezone(timedelta(hours=8))


class _FakeLogger:
    # 记录 log 调用，隔离真实输出。
    def __init__(self):
        self.calls = []

    def log(self, level, msg, exc_info=False, stacklevel=None):
        self.calls.append((level, msg, exc_info, stacklevel))


def _record(message='hi', lineno=42, func='run'):
    return logging.LogRecord(
        name='rag_data.test', level=logging.INFO, pathname=__file__,
        lineno=lineno, msg=message, args=(), exc_info=None, func=func,
    )


# ---------------- 配置 ----------------


def test_default_logging_settings():
    logging_settings = Settings.load(use_env=False).logging
    assert logging_settings.backend == 'stdliblog'
    assert logging_settings.level == 'INFO'
    assert logging_settings.format == ''
    assert logging_settings.timezone == ''


def test_logging_settings_are_configurable():
    logging_settings = Settings.load(use_env=False, logging={'backend': 'loguru', 'timezone': 'UTC'}).logging
    assert logging_settings.backend == 'loguru'
    assert logging_settings.timezone == 'UTC'


# ---------------- StdlibTimezoneFormatter ----------------


def test_formatter_honours_timezone():
    formatter = StdlibTimezoneFormatter(datefmt='%Y-%m-%dT%H:%M:%S%z', tz=PLUS8)
    record = _record()
    record.created = EPOCH
    assert formatter.formatTime(record, formatter.datefmt) == '2024-01-02T11:04:05+0800'


def test_formatter_default_datefmt_is_iso():
    formatter = StdlibTimezoneFormatter(tz=timezone.utc)
    record = _record()
    record.created = EPOCH
    assert formatter.formatTime(record).startswith('2024-01-02T03:04:05')


def test_formatter_defaults_to_system_timezone():
    formatter = StdlibTimezoneFormatter()
    assert formatter.tz is None
    record = _record()
    record.created = EPOCH
    local = timezone(timedelta(seconds=-(time.timezone)))
    assert formatter.formatTime(record).startswith('2024-01-02')


# ---------------- StdlibLogAdapter ----------------


def test_adapter_logs_with_business_callsite():
    fake = _FakeLogger()
    StdlibLogAdapter(fake).info('hi')
    level, msg, exc_info, stacklevel = fake.calls[0]
    assert level == logging.INFO
    assert msg == 'hi'
    assert exc_info is False
    assert stacklevel == 3


def test_adapter_appends_fields():
    fake = _FakeLogger()
    StdlibLogAdapter(fake).info('hi', user_id='u1')
    assert fake.calls[0][1] == 'hi  user_id=u1'


def test_adapter_bind_merges_fields():
    fake = _FakeLogger()
    StdlibLogAdapter(fake).bind(session='s1').info('hi', user_id='u1')
    msg = fake.calls[0][1]
    assert 'session=s1' in msg
    assert 'user_id=u1' in msg


def test_adapter_exception_sets_exc_info():
    fake = _FakeLogger()
    StdlibLogAdapter(fake).exception('boom')
    assert fake.calls[0][0] == logging.ERROR
    assert fake.calls[0][2] is True


def test_adapter_bind_returns_new_instance():
    adapter = StdlibLogAdapter(_FakeLogger())
    assert isinstance(adapter.bind(a=1), StdlibLogAdapter)
    assert isinstance(adapter, LoggerAdapter)


# ---------------- LoggerFactory ----------------


def test_setup_and_get_logger_stdliblog():
    LoggerFactory.setup(backend='stdliblog', level='INFO')
    assert isinstance(LoggerFactory.get_logger(), StdlibLogAdapter)


def test_get_logger_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(LoggerFactory, '_backend', 'bogus')
    with pytest.raises(ValueError):
        LoggerFactory.get_logger()


def test_default_format_comes_from_pyproject():
    # 工厂未收到 fmt 时读取 pyproject 的 log_cli_format。
    fmt = _load_pytest_log_format_from_toml()
    assert fmt is not None
    assert '%(pathname)s' in fmt


def test_setup_applies_timezone():
    LoggerFactory.setup(backend='stdliblog', timezone_name='UTC')
    assert LoggerFactory._timezone == ZoneInfo('UTC')


def test_stdliblog_writes_full_path(capsys):
    fmt = '%(levelname)s|%(pathname)s:%(lineno)d|%(funcName)s|%(message)s'
    LoggerFactory.setup(backend='stdliblog', timezone_name='UTC', fmt=fmt)
    LoggerFactory.get_logger().info('hi')
    out = capsys.readouterr().out
    assert __file__ in out
    assert 'test_stdliblog_writes_full_path' in out


def test_loguru_writes_full_path(capsys):
    pytest.importorskip('loguru')
    LoggerFactory.setup(backend='loguru', timezone_name='UTC')
    LoggerFactory.get_logger().info('hi')
    out = capsys.readouterr().out
    assert __file__ in out
    assert 'test_loguru_writes_full_path' in out


# ---------------- 门面 ----------------


def test_build_logger_uses_settings():
    settings = Settings.load(use_env=False, logging={'backend': 'stdliblog', 'timezone': 'UTC'})
    assert isinstance(build_logger(settings), LoggerAdapter)


def test_structlog_writes_full_path(capsys):
    pytest.importorskip("structlog")
    LoggerFactory.setup(backend="structlog", timezone_name="UTC")
    LoggerFactory.get_logger().info("hi")

    out = capsys.readouterr().out
    assert out

