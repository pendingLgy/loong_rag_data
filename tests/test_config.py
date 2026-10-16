# 配置装载单元测试：三类来源（JSON 对象、环境变量、config.yml）与各分区（含 db）的默认值、覆盖与校验。
#
# 依赖最小：仅 pydantic 与 pydantic-settings；YAML 用例在 PyYAML 缺失时跳过（带原因）。

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
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['embedding:', '  provider: qwen', '  dim: 8', 'logging:', '  level: DEBUG']), encoding='utf-8')
    settings = Settings.load_yaml(str(path))
    assert isinstance(settings, Settings)
    assert settings.embedding.provider == 'qwen'
    assert settings.embedding.dim == 8
    assert settings.logging.level == 'DEBUG'


def test_load_yaml_empty_file_uses_defaults(tmp_path):
    _require_yaml()
    path = tmp_path / 'empty.yml'
    path.write_text('', encoding='utf-8')
    settings = Settings.load_yaml(str(path))
    assert settings.embedding == EmbeddingSettings()


def test_load_yaml_overrides_beat_file(tmp_path):
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['embedding:', '  provider: qwen']), encoding='utf-8')
    settings = Settings.load_yaml(str(path), embedding={'provider': 'openai'})
    assert settings.embedding.provider == 'openai'


def test_load_yaml_missing_file_raises(tmp_path):
    _require_yaml()
    with pytest.raises(ConfigError):
        Settings.load_yaml(str(tmp_path / 'missing.yml'))


def test_load_yaml_non_mapping_raises(tmp_path):
    _require_yaml()
    path = tmp_path / 'list.yml'
    path.write_text(_dump(['- a', '- b']), encoding='utf-8')
    with pytest.raises(ConfigError):
        Settings.load_yaml(str(path))


def test_load_yaml_rejects_unknown_section(tmp_path):
    _require_yaml()
    path = tmp_path / 'config.yml'
    path.write_text(_dump(['bogus:', '  x: 1']), encoding='utf-8')
    with pytest.raises(ConfigError):
        Settings.load_yaml(str(path))

# ---------------- db 分区 ----------------


def test_db_defaults():
    settings = Settings.load_mapping()
    assert settings.db.backend == 'sqlite'
    assert settings.db.sqlite.path == ''
    assert settings.db.mysql.host == 'localhost'
    assert settings.db.mysql.port == 3306
    assert settings.db.mysql.user == 'root'
    assert settings.db.mysql.charset == 'utf8mb4'
    assert settings.db.pg.host == 'localhost'
    assert settings.db.pg.port == 5432
    assert settings.db.pg.user == 'postgres'
    assert settings.db.pg.connection_string == ''


def test_db_section_registered():
    from rag_data.config import MODULE_SECTIONS, SECTION_DB, SECTION_MODEL_CLASSES, DBSettings
    assert SECTION_DB == 'db'
    assert SECTION_DB in MODULE_SECTIONS
    assert SECTION_MODEL_CLASSES[SECTION_DB] is DBSettings


def test_db_mapping_override_backend():
    settings = Settings.load_mapping(db={'backend': 'pg'})
    assert settings.db.backend == 'pg'


def test_db_mapping_preserves_other_nested_defaults():
    settings = Settings.load_mapping(db={'mysql': {'host': '10.0.0.5', 'port': 3307}})
    assert settings.db.mysql.host == '10.0.0.5'
    assert settings.db.mysql.port == 3307
    assert settings.db.mysql.user == 'root'
    assert settings.db.pg.port == 5432



def test_db_mapping_overrides_beat_defaults():
    settings = Settings.load_mapping(db={'sqlite': {'path': 'store.db'}, 'pg': {'connection_string': 'postgresql:store'}})
    assert settings.db.sqlite.path == 'store.db'
    assert settings.db.pg.connection_string == 'postgresql:store'
    assert settings.db.pg.host == 'localhost'


def test_db_env_nested_keys():
    settings = Settings.load_env({
        'RAG_DB__BACKEND': 'mysql',
        'RAG_DB__MYSQL__HOST': 'db.internal',
        'RAG_DB__MYSQL__PORT': '3307',
        'RAG_DB__SQLITE__PATH': 'data.db',
    })
    assert settings.db.backend == 'mysql'
    assert settings.db.mysql.host == 'db.internal'
    assert settings.db.mysql.port == 3307
    assert settings.db.sqlite.path == 'data.db'
    assert settings.db.pg.port == 5432


def test_db_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        Settings.load_mapping(db={'backend': 'oracle'})


def test_db_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Settings.load_mapping(db={'sqlite': {'unknown': 1}})


def test_db_rejects_unknown_nested_section():
    with pytest.raises(ValidationError):
        Settings.load_mapping(db={'redis': {}})


def test_db_rejects_bad_port():
    with pytest.raises(ValidationError):
        Settings.load_mapping(db={'mysql': {'port': 0}})



# ---------------- vector 分区 ----------------


def test_vector_defaults():
    settings = Settings.load_mapping()
    v = settings.vector
    assert v.backend == 'milvus'
    assert v.collection == 'rag_data'
    assert v.vector_size == 1024
    assert v.top_k == 5
    assert v.metric_type == 'COSINE'
    assert v.milvus.uri == ''
    assert v.milvus.token == ''
    assert v.milvus.db_name == ''
    assert v.pg.host == 'localhost'
    assert v.pg.port == 5432
    assert v.pg.user == 'postgres'
    assert v.pg.database == 'rag_data'
    assert v.pg.use_hnsw is True
    assert v.pg.connection_string == ''


def test_vector_section_registered():
    from rag_data.config import MODULE_SECTIONS, SECTION_MODEL_CLASSES, SECTION_VECTOR, VectorSettings
    assert SECTION_VECTOR == 'vector'
    assert SECTION_VECTOR in MODULE_SECTIONS
    assert SECTION_MODEL_CLASSES[SECTION_VECTOR] is VectorSettings


def test_vector_mapping_overrides():
    settings = Settings.load_mapping(vector={
        'backend': 'pg',
        'collection': 'docs',
        'vector_size': 768,
        'top_k': 9,
        'pg': {'host': 'vec.internal', 'use_hnsw': False},
    })
    v = settings.vector
    assert v.backend == 'pg'
    assert v.collection == 'docs'
    assert v.vector_size == 768
    assert v.top_k == 9
    assert v.pg.host == 'vec.internal'
    assert v.pg.use_hnsw is False
    # 未覆盖的字段保留默认值。
    assert v.pg.port == 5432
    assert v.milvus.uri == ''


def test_vector_env_nested_keys():
    settings = Settings.load_env({
        'RAG_VECTOR__BACKEND': 'pg',
        'RAG_VECTOR__VECTOR_SIZE': '768',
        'RAG_VECTOR__PG__DATABASE': 'vec',
        'RAG_VECTOR__MILVUS__URI': 'milvus.internal:19530',
    })
    v = settings.vector
    assert v.backend == 'pg'
    assert v.vector_size == 768
    assert v.pg.database == 'vec'
    assert v.milvus.uri == 'milvus.internal:19530'
    assert v.top_k == 5


def test_vector_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        Settings.load_mapping(vector={'backend': 'oracle'})


def test_vector_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Settings.load_mapping(vector={'bogus': 1})


def test_vector_rejects_bad_top_k():
    with pytest.raises(ValidationError):
        Settings.load_mapping(vector={'top_k': 0})




# ---------------- enable 开关 ----------------


def test_sections_default_disabled():
    settings = Settings.load_mapping()
    for name in ['parsing', 'embedding', 'llm', 'db', 'vector']:
        assert getattr(settings, name).enable is False
    for name in ['logging']:
        assert getattr(settings, name).enable is True

def test_section_enable_overridable_by_mapping():
    settings = Settings.load_mapping(logging={'enable': False})
    assert settings.logging.enable is False
    assert settings.parsing.enable is False


def test_section_enable_overridable_by_env():
    settings = Settings.load_env({'RAG_DB__ENABLE': 'true', 'RAG_VECTOR__ENABLE': '1'})
    assert settings.db.enable is True
    assert settings.vector.enable is True
    assert settings.embedding.enable is False


