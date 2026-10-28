# 测试公共夹具兼集成测试配置：确认 src 布局可导入，并提供常用夹具与配置解析结果。
#
# 集成测试配置的唯一来源是下方各分区 JSON 字符串，不读取环境变量。
#   1. 按本机环境修改 *_JSON 中的连接信息与密钥，api_key 切勿提交仓库；
#   2. samples 未配置时对应用例自动跳过，md 默认取仓库 README.md；
#   3. 删除动作仅限 vector.prefix 指定的前缀。

import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 获取项目根目录（假设 conftest.py 在 tests 目录下，往上找一层是根目录）
# 如果 conftest.py 直接在根目录，可以根据实际路径调整
BASE_DIR = Path(__file__).resolve().parent.parent
env_file = BASE_DIR / ".env_ai_key"

# 显式加载自定义文件名的环境变量
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    # 如果在某些特殊 CI 环境，也可以尝试不带路径加载默认的
    load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "src")
_TESTS = os.path.dirname(os.path.abspath(__file__))
for _path in (_SRC, _TESTS):
    if _path not in sys.path: sys.path.insert(0, _path)

from rag_data.core.config import (  # noqa: E402
    EmbeddingSettings,
    LLMSettings,
    ParsingSettings,
    RagSetting,
)

# ---------------- 各分区 JSON 文本 ----------------

MILVUS_JSON = r"""
{
  "uri": "http://127.0.0.1:19530",
  "token": "root:Milvus",
  "db_name": "autotest_rag_data"
}
"""

PG_JSON = r"""
{
  "connection_string": "",
  "host": "localhost",
  "port": 5432,
  "user": "root",
  "password": "123456",
  "database": "autotest_rag_data"
}
"""

MYSQL_JSON = r"""
{
  "host": "127.0.0.1",
  "port": 3306,
  "user": "root",
  "password": "123456",
  "database": "autotest_rag_data"
}
"""

SQLITE_JSON = r"""
{
  "path": "F:\\workspace\\code\\python\\rag-data\\"
}
"""

VECTOR_JSON = r"""
{
  "backend": "both",
  "dim": 8,
  "keep": false,
  "prefix": "autotest_"
}
"""

EMBEDDING_JSON = r"""
{
  "enable": true,
  "provider": "qwen",
  "model": "qwen3.7-text-embedding",
  "dim": 1024,
  "api_key": "",
  "batch_size": 10
}
"""

LLM_JSON = r"""
{
  "enable": true,
  "provider": "qwen",
  "model": "qwen3.7-flash",
  "temperature": 0.0
}
"""

PARSING_JSON = r"""
{
  "spacy_model": "zh_core_web_sm",
  "max_chars": 500,
  "safe_max_chars": 2000
}
"""

SAMPLES_JSON = r"""
{
  "epub": "C:\\Users\\loong\\Desktop\\经营未来 (李明博) (z-library.sk, 1lib.sk, z-lib.sk).epub",
  "txt": "C:\\Users\\loong\\Desktop\\test_chunk.txt",
  "md": "F:\\workspace\\code\\python\\rag-data\\README.md",
  "pdf": "C:\\Users\\loong\\Desktop\\图解HTTP+彩色版.pdf",
  "docx": "C:\\Users\\loong\\Desktop\\xxx.docx",
  "sample_chars": 3000
}
"""

CASES_JSON = r"""
{
  "vector_store": {
    "collection": "autotest_memories"
  },
  "qwen_epub": {
    "collection": "autotest_qwen_epub"
  },
  "facade_ragdata": {
    "max_chars": 120,
    "safe_max_chars": 600
  },
  "facade_memory": {
    "collection": "autotest_memory",
    "backend": "pg"
  },
  "rag_eval": {
    "collection": "autotest_rag_eval",
    "top_k": 5,
    "min_recall": 0.8,
    "min_mrr": 0.5
  }
}
"""

# ---------------- 各分区配置字典 ----------------

MILVUS_CONFIG = json.loads(MILVUS_JSON)
PG_CONFIG = json.loads(PG_JSON)
MYSQL_CONFIG = json.loads(MYSQL_JSON)
SQLITE_CONFIG = json.loads(SQLITE_JSON)
VECTOR_CONFIG = json.loads(VECTOR_JSON)
EMBEDDING_CONFIG = json.loads(EMBEDDING_JSON)
LLM_CONFIG = json.loads(LLM_JSON)
PARSING_CONFIG = json.loads(PARSING_JSON)
SAMPLES_CONFIG = json.loads(SAMPLES_JSON)
CASES_CONFIG = json.loads(CASES_JSON)

# ---------------- 各分区设置对象 ----------------
#
# JSON 中每个分区在此一次性实例化为配置模型，用例直接读属性，不再逐字段取值与转换；
# 非 Settings 的配置（samples、cases、vector 的 keep 与 prefix）直接从 对应 *_CONFIG 字典读取。
# 注意 PGSettings 与 PGVectorSettings 共用 pg 分区：前者供关系库，后者供向量库。

PARSING_SETTINGS = ParsingSettings(**PARSING_CONFIG)
EMBEDDING_SETTINGS = EmbeddingSettings(**EMBEDDING_CONFIG)
EMBEDDING_SETTINGS.api_key = os.getenv("embedding_api_key", "")

LLM_SETTINGS = LLMSettings(**LLM_CONFIG)


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
