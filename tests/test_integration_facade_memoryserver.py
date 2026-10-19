# 门面集成测试：MemoryServer 用真实嵌入器、大模型、关系库与向量库跑通四条记忆流程。
#
# 外部依赖：嵌入与事实抽取需要 API Key；向量库需要 Milvus 或 PG；关系库默认用临时 SQLite。
# 任一能力不可用时按服务名与目标地址跳过，可用 pytest -rs 查看跳过清单。
#
# 配置来源为 conftest 的 *_JSON 与各分区设置对象。

from __future__ import annotations

import sys
import uuid

import pytest

from conftest import CASES_CONFIG, VECTOR_CONFIG, EMBEDDING_SETTINGS, LLM_SETTINGS, MILVUS_SETTINGS, PG_VECTOR_SETTINGS, SQLITE_SETTINGS, VECTOR_SETTINGS
from rag_data import MemoryServer
from rag_data.config import MemorySettings
from rag_data.embedding.embedder_handler import EmbedderHandler
from rag_data.exceptions import ConfigError, DataError
from rag_data.llm.llm_handler import LLMHandler
from rag_data.memory.handler import MemoryHandler
from rag_data.memory.scope import session_scope
from rag_data.storage.db import BaseDB
from rag_data.storage.vector_store.vector_engine import VectorEngine

# 属于 Settings 的参数取设置对象属性，用例专属值仍按点分路径读取。
API_KEY = EMBEDDING_SETTINGS.api_key
DIM = EMBEDDING_SETTINGS.dim
VECTOR_BACKEND = VECTOR_SETTINGS.backend
COLLECTION = CASES_CONFIG["facade_memory"]["collection"]
KEEP = VECTOR_CONFIG["keep"]

MILVUS_URI = MILVUS_SETTINGS.uri
MILVUS_TOKEN = MILVUS_SETTINGS.token
MILVUS_DB = MILVUS_SETTINGS.db_name

PG_CONNECTION_STRING = PG_VECTOR_SETTINGS.connection_string
PG_HOST = PG_VECTOR_SETTINGS.host
PG_PORT = PG_VECTOR_SETTINGS.port
PG_USER = PG_VECTOR_SETTINGS.user
PG_PASSWORD = PG_VECTOR_SETTINGS.password
PG_DB = PG_VECTOR_SETTINGS.database

# 只允许清理该前缀的集合。
COLLECTION_PREFIX = VECTOR_CONFIG["prefix"]

# 两轮对话：包含姓名、城市与项目等可抽取事实，供后续检索与更新复用。
MESSAGES = [
    {"role": "user", "content": "我叫林轩，现在在南京做检索系统开发。"},
    {"role": "assistant", "content": "记住了：你在南京做检索系统开发。"},
    {"role": "user", "content": "我的项目用 Python 加 Milvus，周末喜欢去紫金山爬山。"},
]
REWRITTEN = "我叫林轩，现在在杭州做向量检索平台开发。"
QUERY = "我周末喜欢做什么"

pytestmark = pytest.mark.integration

if ["--integration","--skip"] not in sys.argv:
    pytest.skip(
        f"[SKIP INFO] File {__file__}，integration 集成测试默认跳过",
        allow_module_level=True,
    )


# ---------------- 辅助 ----------------


def _requires_api_key() -> None: 
    if not API_KEY: 
        pytest.skip("未配置 API Key，请在 conftest 的 embedding.api_key 中填入")


def _target() -> str: 
    # 跳过信息里给出可核对的连接目标，便于区分驱动缺失与服务未启动。
    if VECTOR_BACKEND == "milvus": 
        return "milvus uri=" + MILVUS_URI + " collection=" + COLLECTION
    return "pg host=" + PG_HOST + ":" + str(PG_PORT) + " db=" + PG_DB + " collection=" + COLLECTION


def _skip_service(service: str, exc: Exception) -> None: 
    pytest.skip("跳过：" + service + "（目标 " + _target() + "）：" + type(exc).__name__ + "，" + str(exc))


def _flush(server: MemoryServer) -> None: 
    # Milvus 为准实时写入，显式 flush 后才能稳定读到刚写入的数据；PG 无需处理。
    client = getattr(getattr(server.vector, "store", None), "client", None)
    if client is None or not hasattr(client, "flush"): 
        return
    try: 
        client.flush(collection_name=server.vector.collection_name)
    except Exception: 
        pass


# ---------------- 夹具 ----------------


@pytest.fixture(scope="module")
def sqlite_path(tmp_path_factory) -> str: 
    if SQLITE_SETTINGS.path: 
        return SQLITE_SETTINGS.path
    return str(tmp_path_factory.mktemp("memory_it") / "memory.db")


@pytest.fixture(scope="module")
def memory_settings(sqlite_path: str) -> MemorySettings: 
    _requires_api_key()
    return MemorySettings.load_mapping({
        "enable": True,
        "embedding": {
            "enable": True,
            "provider": EMBEDDING_SETTINGS.provider,
            "model": EMBEDDING_SETTINGS.model,
            "api_key": API_KEY,
            "dim": DIM,
        },
        "llm": {
            "enable": True,
            "provider": LLM_SETTINGS.provider,
            "model": LLM_SETTINGS.model,
            "api_key": API_KEY,
            "temperature": LLM_SETTINGS.temperature,
        },
        "db": {"enable": True, "backend": "sqlite", "sqlite": {"path": sqlite_path}},
        "vector": {
            "enable": True,
            "backend": VECTOR_BACKEND,
            "collection": COLLECTION,
            "vector_size": DIM,
            "top_k": 5,
            "metric_type": "COSINE",
            "milvus": MILVUS_SETTINGS.model_dump(),
            "pg": PG_VECTOR_SETTINGS.model_dump(),
        },
    })


@pytest.fixture(scope="module")
def server(memory_settings: MemorySettings): 
    try: 
        instance = MemoryServer(memory_settings)
        # 只有集合名带安全前缀时才清空重建，避免误删他人集合。
        if COLLECTION.startswith(COLLECTION_PREFIX): 
            instance.vector.reset()
    except Exception as exc: 
        _skip_service("MemoryServer 装配或向量库链接", exc)
    try: 
        yield instance
    finally: 
        _cleanup(instance)


def _cleanup(server: MemoryServer) -> None: 
    # 仅当集合名带安全前缀且未要求保留时才删除集合，避免误删他人数据。
    if not KEEP and COLLECTION.startswith(COLLECTION_PREFIX): 
        try: 
            server.vector.delete_col()
        except Exception: 
            pass
    for component in (server.vector, server.db): 
        closer = getattr(component, "close", None)
        if callable(closer): 
            try: 
                closer()
            except Exception: 
                pass


@pytest.fixture(scope="module")
def user_id() -> str: 
    return "it_user_" + uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def written(server: MemoryServer, user_id: str): 
    try: 
        result = server.add(MESSAGES, user_id=user_id, metadata={"source": "integration"})
    except Exception as exc: 
        _skip_service("事实抽取与写入", exc)
    _flush(server)
    if not result.results: 
        pytest.skip("大模型未抽取到事实，跳过依赖既有记忆的用例")
    return result.results


def test_memory_server_wires_real_components(server: MemoryServer, memory_settings: MemorySettings): 
    assert server.settings is memory_settings
    assert isinstance(server.embedder, EmbedderHandler)
    assert isinstance(server.llm, LLMHandler)
    assert isinstance(server.db, BaseDB)
    assert isinstance(server.vector, VectorEngine)
    assert isinstance(server.memory, MemoryHandler)
    assert server.memory.embedder is server.embedder
    assert server.memory.llm is server.llm
    assert server.memory.db is server.db
    assert server.memory.vector is server.vector
    assert server.vector.collection_name == COLLECTION
    assert server.vector.vector_size == DIM


def test_vector_collection_is_ready(server: MemoryServer): 
    assert COLLECTION in server.vector.list_cols()


def test_disabled_memory_raises_config_error(): 
    server = MemoryServer(MemorySettings.load_mapping())
    assert server.memory is None
    assert server.embedder is None and server.llm is None
    assert server.db is None and server.vector is None
    with pytest.raises(ConfigError): 
        server.add(MESSAGES, user_id="u1")


def test_add_persists_vector_and_history(server: MemoryServer, written): 
    for item in written: 
        assert item.id > 0
        assert item.memory.strip()
        stored = server.memory.store.get_memory(item.id)
        assert stored is not None
        assert (stored.payload or {}).get("data") == item.memory
        events = [record.event for record in server.db.get_history(item.id)]
        assert "ADD" in events


def test_messages_are_saved_per_session(server: MemoryServer, user_id: str): 
    session = session_scope({"user_id": user_id})
    messages = server.db.get_last_messages(session, 10)
    assert messages
    assert all(message.session_scope == session for message in messages)
    assert any(message.content == MESSAGES[0]["content"] for message in messages)


def test_add_requires_entity_scope(server: MemoryServer): 
    with pytest.raises(DataError): 
        server.add(MESSAGES)


def test_search_returns_written_memory(server: MemoryServer, written, user_id: str): 
    ids = {item.id for item in written}
    result = server.search(QUERY, user_id=user_id, top_k=5, threshold=0.0)
    assert result.results
    assert any(item.id in ids for item in result.results)
    assert all(item.score is not None for item in result.results)


def test_search_is_scoped_by_user(server: MemoryServer, written): 
    ids = {item.id for item in written}
    result = server.search(QUERY, user_id="it_user_other", top_k=5, threshold=0.0)
    assert all(item.id not in ids for item in result.results)


def test_search_rejects_blank_query(server: MemoryServer, user_id: str): 
    with pytest.raises(DataError): 
        server.search("   ", user_id=user_id)


def test_update_rewrites_text(server: MemoryServer, written): 
    target = written[0]
    updated = server.update(target.id, text=REWRITTEN, metadata={"stage": "updated"})
    _flush(server)
    assert updated.result is not None
    assert updated.result.id == target.id
    assert updated.result.memory == REWRITTEN
    stored = server.memory.store.get_memory(target.id)
    assert stored is not None
    payload = stored.payload or {}
    assert payload.get("data") == REWRITTEN
    assert payload.get("stage") == "updated"
    assert "UPDATE" in [record.event for record in server.db.get_history(target.id)]


def test_delete_removes_memory(server: MemoryServer, written): 
    target = written[-1]
    result = server.delete(target.id)
    _flush(server)
    assert result.id == target.id
    assert server.memory.store.get_memory(target.id) is None
    records = server.db.get_history(target.id)
    assert records[-1].event == "DELETE"
    assert records[-1].is_deleted is True
