# 配置装载单元测试：三类来源（JSON 对象、环境变量、config.yml）与各分区（含 db）的默认值、覆盖与校验。
#
# 依赖最小：仅 pydantic 与 pydantic-settings；YAML 用例在 PyYAML 缺失时跳过（带原因）。

import pytest

from pydantic import ValidationError

from rag_data.config import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    MEMORY_ENV_PREFIX,
    SECTION_EMBEDDING,
    SECTION_LOGGING,
    SECTION_MODEL_CLASSES,
    SECTION_PARSING,
    MODULE_SECTIONS,
    EmbeddingSettings,
    MemorySettings,
    ParsingSettings,
    RagSetting,
)
from rag_data.exceptions import ConfigError


def _require_yaml():
    """确认 PyYAML 可用；缺失时带原因跳过（importorskip 的 reason 需 pytest 8.2+）。"""
    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        pytest.skip("缺少可选依赖 PyYAML（模块 yaml）：" + str(exc) + "；安装方式：pip install PyYAML")


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
    settings = RagSetting.load_mapping(mapping)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8


def test_load_mapping_returns_settings_type():
    assert isinstance(RagSetting.load_mapping(), RagSetting)


def test_load_mapping_fills_defaults_ignoring_env(monkeypatch):
    # 不读取环境变量：环境里设了值也应保持字段默认。
    monkeypatch.setenv(ENV_PREFIX + 'EMBEDDING' + ENV_NESTED_DELIMITER + 'PROVIDER', 'qwen')
    settings = RagSetting.load_mapping()
    assert settings.embedding.provider == EmbeddingSettings().provider
    assert settings.parsing.max_chars == ParsingSettings().max_chars


def test_load_mapping_overrides_beat_mapping(mapping):
    settings = RagSetting.load_mapping(mapping, embedding={'provider': 'openai'})
    assert settings.embedding.provider == 'openai'
    assert settings.embedding.dim == 8


def test_load_mapping_rejects_unknown_section():
    with pytest.raises(ConfigError):
        RagSetting.load_mapping({'bogus': {'x': 1}})


def test_load_mapping_rejects_non_mapping():
    with pytest.raises(ConfigError):
        RagSetting.load_mapping('config.yml')


def test_load_mapping_rejects_unknown_field():
    # 分区模型 extra=forbid，未知字段应校验失败。
    with pytest.raises(ValidationError):
        RagSetting.load_mapping({'embedding': {'unknown': 1}})


# ---------------- 来源二：环境变量 ----------------


def test_load_env_reads_nested_keys(environ):
    settings = RagSetting.load_env(environ)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8
    assert settings.logging.level == 'DEBUG'


def test_load_env_ignores_non_prefixed_and_unknown_sections():
    settings = RagSetting.load_env({'OTHER': '1', 'RAG_BOGUS__X': '2'})
    assert settings.embedding.provider == EmbeddingSettings().provider
    assert not hasattr(settings, 'bogus')


def test_load_env_is_isolated_from_real_environment(monkeypatch):
    # 显式传入 environ 时不应受真实环境影响。
    monkeypatch.setenv(ENV_PREFIX + 'PARSING' + ENV_NESTED_DELIMITER + 'MAX_CHARS', '12345')
    settings = RagSetting.load_env({})
    assert settings.parsing.max_chars == ParsingSettings().max_chars


def test_load_env_overrides_beat_environ(environ):
    settings = RagSetting.load_env(environ, logging={'level': 'ERROR'})
    assert settings.logging.level == 'ERROR'


def test_load_env_rejects_unknown_section():
    with pytest.raises(ConfigError):
        RagSetting.load_env({}, bogus={'x': 1})


# ---------------- 来源三：config.yml ----------------


def test_load_yaml_reads_sections(tmp_path):
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['embedding:', '  provider: qwen', '  dim: 8', 'logging:', '  level: DEBUG']), encoding='utf-8')
    settings = RagSetting.load_yaml(str(path))
    assert isinstance(settings, RagSetting)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8
    assert settings.logging.level == 'DEBUG'


def test_load_yaml_empty_file_uses_defaults(tmp_path):
    _require_yaml()
    path = tmp_path / 'empty.yml'
    path.write_text('', encoding='utf-8')
    settings = RagSetting.load_yaml(str(path))
    assert settings.embedding == EmbeddingSettings()


def test_load_yaml_overrides_beat_file(tmp_path):
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['embedding:', '  provider: qwen']), encoding='utf-8')
    settings = RagSetting.load_yaml(str(path), embedding={'provider': 'openai'})
    assert settings.embedding.provider == 'openai'


def test_load_yaml_missing_file_raises(tmp_path):
    _require_yaml()
    with pytest.raises(ConfigError):
        RagSetting.load_yaml(str(tmp_path / 'missing.yml'))


def test_load_yaml_non_mapping_raises(tmp_path):
    _require_yaml()
    path = tmp_path / 'list.yml'
    path.write_text(_dump(['- a', '- b']), encoding='utf-8')
    with pytest.raises(ConfigError):
        RagSetting.load_yaml(str(path))


def test_load_yaml_rejects_unknown_section(tmp_path):
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['bogus:', '  x: 1']), encoding='utf-8')
    with pytest.raises(ConfigError):
        RagSetting.load_yaml(str(path))
# ---------------- memory 分区（独立于 RagSetting） ----------------


def test_memory_defaults():
    m = MemorySettings()
    assert m.enable is False
    assert m.llm.provider == 'openai'
    assert m.db.backend == 'sqlite'
    assert m.vector.backend == 'milvus'


def test_memory_not_in_rag_sections():
    assert 'memory' not in MODULE_SECTIONS
    assert set(SECTION_MODEL_CLASSES) == {SECTION_PARSING, SECTION_EMBEDDING, SECTION_LOGGING}


def test_rag_rejects_memory_section():
    with pytest.raises(ConfigError):
        RagSetting.load_mapping({'memory': {'enable': True}})


def test_rag_setting_has_no_memory_attribute():
    settings = RagSetting.load_mapping()
    for name in ('memory', 'llm', 'db', 'vector'):
        assert not hasattr(settings, name)


def test_memory_env_prefix_is_isolated():
    # 记忆分区自带 RAG_MEMORY_ 前缀，与 RagSetting 的 RAG_ 分区键不重叠。
    assert MEMORY_ENV_PREFIX == 'RAG_MEMORY_'
    m = MemorySettings(enable=True, llm={'provider': 'qwen'})
    assert m.enable is True
    assert m.llm.provider == 'qwen'


def test_memory_rejects_nlp_field():
    # nlp 已从 MemorySettings 移除：句柄由解析侧 parsing 分区统一加载。
    with pytest.raises(ValidationError):
        MemorySettings(nlp='zh_core_web_md')


def test_memory_rejects_unknown_field():
    with pytest.raises(ValidationError):
        MemorySettings(bogus=1)


# ---------------- MemorySettings 三类来源 ----------------


def test_memory_load_mapping_applies_values():
    m = MemorySettings.load_mapping({'db': {'backend': 'pg'}})
    assert m.db.backend == 'pg'
    assert m.enable is False


def test_memory_load_mapping_returns_settings_type():
    assert isinstance(MemorySettings.load_mapping(), MemorySettings)


def test_memory_load_mapping_ignores_env(monkeypatch):
    monkeypatch.setenv('RAG_MEMORY__DB__BACKEND', 'pg')
    assert MemorySettings.load_mapping().db.backend == 'sqlite'


def test_memory_load_mapping_overrides_beat_mapping():
    m = MemorySettings.load_mapping({'db': {'backend': 'pg'}}, db={'backend': 'mysql'})
    assert m.db.backend == 'mysql'


def test_memory_load_mapping_rejects_non_mapping():
    with pytest.raises(ConfigError):
        MemorySettings.load_mapping('config.yml')


def test_memory_load_mapping_rejects_unknown_field():
    with pytest.raises(ValidationError):
        MemorySettings.load_mapping({'bogus': 1})


def test_memory_load_env_reads_nested_keys():
    m = MemorySettings.load_env({
        'RAG_MEMORY__ENABLE': 'true',
        'RAG_MEMORY__DB__BACKEND': 'mysql',
        'RAG_MEMORY__DB__MYSQL__HOST': 'db.internal',
        'RAG_MEMORY__VECTOR__VECTOR_SIZE': '768',
    })
    assert m.enable is True
    assert m.db.backend == 'mysql'
    assert m.db.mysql.host == 'db.internal'
    assert m.vector.vector_size == 768


def test_memory_load_env_ignores_other_prefixes():
    m = MemorySettings.load_env({'RAG_EMBEDDING__DIM': '8', 'OTHER': '1'})
    assert m.db.backend == 'sqlite'


def test_memory_load_env_overrides_beat_environ():
    m = MemorySettings.load_env({'RAG_MEMORY__DB__BACKEND': 'pg'}, db={'backend': 'mysql'})
    assert m.db.backend == 'mysql'

def test_memory_load_yaml_reads_fields(tmp_path):
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['db:', '  backend: pg']), encoding='utf-8')
    m = MemorySettings.load_yaml(str(path))
    assert m.db.backend == 'pg'


def test_memory_load_yaml_empty_file_uses_defaults(tmp_path):
    _require_yaml()
    path = tmp_path / 'empty.yml'
    path.write_text('', encoding='utf-8')
    m = MemorySettings.load_yaml(str(path))
    assert m.enable is False


def test_memory_load_yaml_missing_file_raises(tmp_path):
    _require_yaml()
    with pytest.raises(ConfigError):
        MemorySettings.load_yaml(str(tmp_path / 'missing.yml'))


def test_memory_load_yaml_rejects_non_mapping(tmp_path):
    _require_yaml()
    path = tmp_path / 'list.yml'
    path.write_text(_dump(['- a', '- b']), encoding='utf-8')
    with pytest.raises(ConfigError):
        MemorySettings.load_yaml(str(path))

# ---------------- memory.db ----------------


def test_db_defaults():
    db = MemorySettings().db
    assert db.backend == 'sqlite'
    assert db.sqlite.path == ''
    assert db.mysql.host == 'localhost'
    assert db.mysql.port == 3306
    assert db.pg.host == 'localhost'
    assert db.pg.port == 5432
    assert db.pg.connection_string == ''


def test_db_nested_in_memory_section():
    from rag_data.config import DBSettings
    assert isinstance(MemorySettings().db, DBSettings)


def test_db_mapping_override_backend():
    assert MemorySettings(**{'db': {'backend': 'pg'}}).db.backend == 'pg'


def test_db_mapping_preserves_other_nested_defaults():
    db = MemorySettings(**{'db': {'mysql': {'host': '10.0.0.5', 'port': 3307}}}).db
    assert db.mysql.host == '10.0.0.5'
    assert db.mysql.port == 3307
    assert db.mysql.user == 'root'
    assert db.pg.port == 5432


def test_db_env_nested_keys(monkeypatch):
    monkeypatch.setenv('RAG_MEMORY__DB__BACKEND', 'mysql')
    monkeypatch.setenv('RAG_MEMORY__DB__MYSQL__HOST', 'db.internal')
    monkeypatch.setenv('RAG_MEMORY__DB__SQLITE__PATH', 'data.db')
    db = MemorySettings().db
    assert db.backend == 'mysql'
    assert db.mysql.host == 'db.internal'
    assert db.sqlite.path == 'data.db'


def test_db_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        MemorySettings(**{'db': {'backend': 'oracle'}})


def test_db_rejects_unknown_field():
    with pytest.raises(ValidationError):
        MemorySettings(**{'db': {'sqlite': {'unknown': 1}}})


def test_db_rejects_bad_port():
    with pytest.raises(ValidationError):
        MemorySettings(**{'db': {'mysql': {'port': 0}}})


# ---------------- memory.vector ----------------


def test_vector_defaults():
    v = MemorySettings().vector
    assert v.backend == 'milvus'
    assert v.collection == 'rag_data'
    assert v.vector_size == 1024
    assert v.top_k == 5
    assert v.metric_type == 'COSINE'
    assert v.milvus.uri == ''
    assert v.pg.port == 5432
    assert v.pg.use_hnsw is True


def test_vector_nested_in_memory_section():
    from rag_data.config import VectorSettings
    assert isinstance(MemorySettings().vector, VectorSettings)


def test_vector_mapping_overrides():
    v = MemorySettings(**{'vector': {
        'backend': 'pg',
        'collection': 'docs',
        'vector_size': 768,
        'top_k': 9,
        'pg': {'host': 'vec.internal', 'use_hnsw': False},
    }}).vector
    assert v.backend == 'pg'
    assert v.collection == 'docs'
    assert v.vector_size == 768
    assert v.top_k == 9
    assert v.pg.use_hnsw is False
    assert v.pg.port == 5432


def test_vector_env_nested_keys(monkeypatch):
    monkeypatch.setenv('RAG_MEMORY__VECTOR__BACKEND', 'pg')
    monkeypatch.setenv('RAG_MEMORY__VECTOR__VECTOR_SIZE', '768')
    monkeypatch.setenv('RAG_MEMORY__VECTOR__MILVUS__URI', 'milvus.internal:19530')
    v = MemorySettings().vector
    assert v.backend == 'pg'
    assert v.vector_size == 768
    assert v.milvus.uri == 'milvus.internal:19530'


def test_vector_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        MemorySettings(**{'vector': {'backend': 'oracle'}})


def test_vector_rejects_unknown_field():
    with pytest.raises(ValidationError):
        MemorySettings(**{'vector': {'bogus': 1}})


def test_vector_rejects_bad_top_k():
    with pytest.raises(ValidationError):
        MemorySettings(**{'vector': {'top_k': 0}})



# ---------------- enable 开关 ----------------


def test_sections_default_disabled():
    settings = RagSetting.load_mapping()
    for name in ('parsing', 'embedding'):
        assert getattr(settings, name).enable is False
    assert settings.logging.enable is True
    m = MemorySettings()
    for name in ('llm', 'db', 'vector'):
        assert getattr(m, name).enable is False


def test_memory_enable_from_env(monkeypatch):
    monkeypatch.setenv('RAG_MEMORY__ENABLE', 'true')
    monkeypatch.setenv('RAG_MEMORY__DB__ENABLE', '1')
    m = MemorySettings()
    assert m.enable is True
    assert m.db.enable is True
    assert m.vector.enable is False

