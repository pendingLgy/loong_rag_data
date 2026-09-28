# 配置的单元测试：覆盖默认值、JSON 配置源、环境变量与代码硬编码。

import pytest

from rag_data.config import (
    ChunkingSettings,
    EmbeddingSettings,
    LoggingSettings,
    Settings,
    StorageSettings,
    load_env_overrides,
)
from rag_data import RagData, build_settings
from rag_data.exceptions import ConfigError

_HAS_REAL_PYDANTIC_SETTINGS = False
try:
    import pydantic_settings

    _HAS_REAL_PYDANTIC_SETTINGS = getattr(pydantic_settings, "__file__", None) is not None
except ImportError:  # pragma: no cover
    pass

requires_pydantic_settings = pytest.mark.skipif(
    not _HAS_REAL_PYDANTIC_SETTINGS,
    reason="需要真实安装 pydantic-settings",
)


def test_module_defaults():
    config = Settings()
    assert config.storage.vector_dim == 1024
    assert config.storage.metric == "COSINE"
    assert config.storage.index_type == "HNSW"
    assert config.storage.backend == "memory"
    assert config.chunking.max_chars == 250
    assert config.chunking.overlap_sents == 1
    assert config.embedding.batch_size == 128
    assert config.embedding.model == ""
    assert config.nlp.spacy_model == "zh_core_web_sm"
    assert config.logging.backend == "auto"
    assert config.logging.level == "INFO"
    assert config.logging.json_output is False


def test_sections_accept_dict_overrides():
    config = Settings(chunking={"max_chars": 180}, embedding={"batch_size": 4})
    assert config.chunking.max_chars == 180
    assert config.embedding.batch_size == 4


def test_sections_accept_model_overrides():
    config = Settings(logging=LoggingSettings(backend="stdlib"))
    assert config.logging.backend == "stdlib"


def test_logging_json_alias_is_accepted():
    settings = LoggingSettings(**{"json": True})
    assert settings.json_output is True


def test_section_models_exist():
    assert StorageSettings().backend == "memory"
    assert ChunkingSettings().max_chars == 250
    assert EmbeddingSettings().batch_size == 128
    assert LoggingSettings().level == "INFO"


def test_unknown_section_field_is_rejected():
    with pytest.raises(Exception):
        ChunkingSettings(unknown="x")


def test_invalid_value_is_rejected():
    with pytest.raises(Exception):
        ChunkingSettings(max_chars=10)



@requires_pydantic_settings
def test_env_nested_override(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE__VECTOR_DIM", "256")
    assert Settings().storage.vector_dim == 256


@requires_pydantic_settings
def test_env_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("RAG_CHUNKING__MAX_CHARS", "10")
    with pytest.raises(Exception):
        Settings()


# ---------- 环境变量 ----------


def test_load_env_overrides_parses_nested_keys():
    env = {
        "RAG_STORAGE__BACKEND": "milvus",
        "RAG_STORAGE__VECTOR_DIM": "512",
        "RAG_LOGGING__JSON": "true",
    }
    data = load_env_overrides(env)
    assert data["storage"] == {"backend": "milvus", "vector_dim": "512"}
    assert data["logging"] == {"json": "true"}


def test_load_env_overrides_skips_non_config_keys():
    env = {"RAG_": "", "OTHER": "x"}
    assert load_env_overrides(env) == {}


def test_load_env_overrides_parses_json_values():
    value = '[{"name": "user_id", "type": "VARCHAR"}]'
    env = {"RAG_MODELS__PROMOTED_FIELDS": value}
    data = load_env_overrides(env)
    assert data["models"]["promoted_fields"] == [{"name": "user_id", "type": "VARCHAR"}]


def test_load_reads_environment_by_default(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE__BACKEND", "milvus")
    monkeypatch.setenv("RAG_STORAGE__VECTOR_DIM", "256")
    monkeypatch.setenv("RAG_LOGGING__JSON", "true")
    config = Settings.load()
    assert config.storage.backend == "milvus"
    assert config.storage.vector_dim == 256
    assert config.logging.json_output is True


def test_load_env_fills_fields_missing_from_override(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE__MILVUS_URI", "127.0.0.1:19531")
    config = Settings.load(storage={"backend": "milvus"})
    assert config.storage.backend == "milvus"
    assert config.storage.milvus_uri == "127.0.0.1:19531"


def test_load_hardcode_wins_over_env(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE__VECTOR_DIM", "256")
    monkeypatch.setenv("RAG_STORAGE__BACKEND", "milvus")
    monkeypatch.setenv("RAG_LOGGING__LEVEL", "DEBUG")
    config = Settings.load(storage={"vector_dim": 512}, logging={"level": "WARNING"})
    # 同时存在时以硬编码为准（硬编码 > 环境变量）
    assert config.storage.vector_dim == 512
    assert config.logging.level == "WARNING"
    # 未硬编码的字段仍由环境变量填充
    assert config.storage.backend == "milvus"
    assert config.logging.json_output is False


def test_load_env_is_lower_than_code_override(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE__VECTOR_DIM", "256")
    assert Settings.load(storage={"vector_dim": 512}).storage.vector_dim == 512


def test_load_use_env_false_ignores_environment(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE__BACKEND", "milvus")
    monkeypatch.setenv("RAG_STORAGE__VECTOR_DIM", "256")
    monkeypatch.setenv("RAG_LOGGING__LEVEL", "DEBUG")

    # 对照：默认读取环境变量，证明环境确实生效
    from_env = Settings.load()
    assert from_env.storage.backend == "milvus"
    assert from_env.storage.vector_dim == 256
    assert from_env.logging.level == "DEBUG"

    # use_env=False 时忽略环境变量，字段回落到默认值
    config = Settings.load(use_env=False)
    assert config.storage.backend == "memory"
    assert config.storage.vector_dim == 1024
    assert config.logging.level == "INFO"

    # 关闭环境变量只影响环境来源，显式给出的配置照常生效
    hardcoded = Settings.load(use_env=False, storage={"vector_dim": 512})
    assert hardcoded.storage.vector_dim == 512
    assert hardcoded.storage.backend == "memory"


def test_load_ignores_unrelated_env_prefix(monkeypatch):
    monkeypatch.setenv("RAG_UNRELATED", "1")
    assert Settings.load().storage.backend == "memory"


def test_load_invalid_env_value_raises(monkeypatch):
    monkeypatch.setenv("RAG_CHUNKING__MAX_CHARS", "10")
    with pytest.raises(Exception):
        Settings.load()


# ---------- 代码硬编码 ----------


def test_load_accepts_section_overrides():
    config = Settings.load(storage={"backend": "milvus", "vector_dim": 512})
    assert config.storage.backend == "milvus"
    assert config.storage.vector_dim == 512
    assert config.chunking.max_chars == 250


def test_load_accepts_mapping_source():
    config = Settings.load({"storage": {"backend": "milvus"}}, chunking={"max_chars": 180})
    assert config.storage.backend == "milvus"
    assert config.chunking.max_chars == 180


def test_load_unknown_section_raises():
    with pytest.raises(ConfigError):
        Settings.load(unknown={"a": 1})


def test_load_unknown_section_in_source_raises():
    with pytest.raises(ConfigError):
        Settings.load({"storage": {"vector_dim": 8}, "unknown": {}})


def test_load_unknown_field_in_section_raises():
    with pytest.raises(Exception):
        Settings.load(chunking={"unknown": 1})


def test_load_rejects_file_path():
    # 配置文件读取已移除，传路径应给出明确错误
    with pytest.raises(ConfigError):
        Settings.load("config.json")


def test_build_settings_accepts_overrides():
    assert build_settings(storage={"backend": "milvus"}).storage.backend == "milvus"


def test_ragdata_create_accepts_overrides(logger):
    app = RagData.create(overrides={"storage": {"backend": "memory"}}, logger=logger, nlp=None)
    assert app.settings.storage.backend == "memory"

