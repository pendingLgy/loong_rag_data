# 配置装载单元测试：三类来源（JSON 对象、环境变量、config.yml）。
#
# 依赖最小：仅 pydantic 与 pydantic-settings；YAML 用例在 PyYAML 缺失时自动跳过。

import pytest
from pydantic import ValidationError

from rag_data.config import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    EmbeddingSettings,
    ParsingSettings,
    Settings,
)
from rag_data.exceptions import ConfigError


# ---------------- 工具与夹具 ----------------


def _dump(lines):
    # 用换行拼接 YAML 文本，避免在源码里书写转义字符。
    return chr(10).join(lines)


@pytest.fixture()
def mapping():
    return {'embedding': {'provider': 'qwen', 'dim': 8}}


@pytest.fixture()
def environ():
    return {
        'RAG_EMBEDDING__PROVIDER': 'qwen',
        'RAG_EMBEDDING__DIM': '8',
        'RAG_LOGGING__LEVEL': 'DEBUG',
        'OTHER_KEY': 'ignored',
    }


# ---------------- 来源一：JSON 对象 ----------------


def test_load_mapping_applies_values(mapping):
    settings = Settings.load_mapping(mapping)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8


def test_load_mapping_returns_settings_type():
    assert isinstance(Settings.load_mapping(), Settings)


def test_load_mapping_fills_defaults_ignoring_env(monkeypatch):
    # 不读取环境变量：环境里设了值也应保持字段默认。
    monkeypatch.setenv(ENV_PREFIX + 'EMBEDDING' + ENV_NESTED_DELIMITER + 'PROVIDER', 'qwen')
    settings = Settings.load_mapping()
    assert settings.embedding.provider == EmbeddingSettings().provider
    assert settings.parsing.max_chars == ParsingSettings().max_chars


def test_load_mapping_overrides_beat_mapping(mapping):
    settings = Settings.load_mapping(mapping, embedding={'provider': 'openai'})
    assert settings.embedding.provider == 'openai'
    assert settings.embedding.dim == 8


def test_load_mapping_rejects_unknown_section():
    with pytest.raises(ConfigError):
        Settings.load_mapping({'bogus': {'x': 1}})


def test_load_mapping_rejects_non_mapping():
    with pytest.raises(ConfigError):
        Settings.load_mapping('config.yml')


def test_load_mapping_rejects_unknown_field():
    # 分区模型 extra=forbid，未知字段应校验失败。
    with pytest.raises(ValidationError):
        Settings.load_mapping({'embedding': {'unknown': 1}})


# ---------------- 来源二：环境变量 ----------------


def test_load_env_reads_nested_keys(environ):
    settings = Settings.load_env(environ)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8
    assert settings.logging.level == 'DEBUG'


def test_load_env_ignores_non_prefixed_and_unknown_sections():
    settings = Settings.load_env({'OTHER': '1', 'RAG_BOGUS__X': '2'})
    assert settings.embedding.provider == EmbeddingSettings().provider
    assert not hasattr(settings, 'bogus')


def test_load_env_is_isolated_from_real_environment(monkeypatch):
    # 显式传入 environ 时不应受真实环境影响。
    monkeypatch.setenv(ENV_PREFIX + 'PARSING' + ENV_NESTED_DELIMITER + 'MAX_CHARS', '12345')
    settings = Settings.load_env({})
    assert settings.parsing.max_chars == ParsingSettings().max_chars


def test_load_env_overrides_beat_environ(environ):
    settings = Settings.load_env(environ, logging={'level': 'ERROR'})
    assert settings.logging.level == 'ERROR'


def test_load_env_rejects_unknown_section():
    with pytest.raises(ConfigError):
        Settings.load_env({}, bogus={'x': 1})


# ---------------- 来源三：config.yml ----------------


def test_load_yaml_reads_sections(tmp_path):
    pytest.importorskip('yaml')
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['embedding:', '  provider: qwen', '  dim: 8', 'logging:', '  level: DEBUG']), encoding='utf-8')
    settings = Settings.load_yaml(str(path))
    assert isinstance(settings, Settings)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8
    assert settings.logging.level == 'DEBUG'


def test_load_yaml_empty_file_uses_defaults(tmp_path):
    pytest.importorskip('yaml')
    path = tmp_path / 'empty.yml'
    path.write_text('', encoding='utf-8')
    settings = Settings.load_yaml(str(path))
    assert settings.embedding == EmbeddingSettings()


def test_load_yaml_overrides_beat_file(tmp_path):
    pytest.importorskip('yaml')
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['embedding:', '  provider: qwen']), encoding='utf-8')
    settings = Settings.load_yaml(str(path), embedding={'provider': 'openai'})
    assert settings.embedding.provider == 'openai'


def test_load_yaml_missing_file_raises(tmp_path):
    pytest.importorskip('yaml')
    with pytest.raises(ConfigError):
        Settings.load_yaml(str(tmp_path / 'missing.yml'))


def test_load_yaml_non_mapping_raises(tmp_path):
    pytest.importorskip('yaml')
    path = tmp_path / 'list.yml'
    path.write_text(_dump(['- a', '- b']), encoding='utf-8')
    with pytest.raises(ConfigError):
        Settings.load_yaml(str(path))


def test_load_yaml_rejects_unknown_section(tmp_path):
    pytest.importorskip('yaml')
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['bogus:', '  x: 1']), encoding='utf-8')
    with pytest.raises(ConfigError):
        Settings.load_yaml(str(path))
