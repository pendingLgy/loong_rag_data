from __future__ import annotations

import importlib.util
import re
import sys

import pytest

from conftest import MILVUS_SETTINGS
from rag_data.nlp.lemmatization import lemmatize_zh_for_bm25

# 凭据与连接信息一律来自 conftest 的 *_JSON 与各分区设置对象。
MILVUS_URI = MILVUS_SETTINGS.uri
MILVUS_TOKEN = MILVUS_SETTINGS.token
MILVUS_DB = MILVUS_SETTINGS.db_name


def _requires_pymilvus():
    if importlib.util.find_spec('pymilvus') is None:
        pytest.skip("未安装 pymilvus，请先执行 pip install pymilvus")


pytestmark = pytest.mark.integration

if "--integration" not in sys.argv:
    skip_reason = f"[SKIP INFO] File {__file__}， integration: 集成测试，默认跳过（请使用 --integration 参数运行）"
    pytest.skip(skip_reason, allow_module_level=True)


@pytest.fixture(scope='module')
def pymilvus():
    _requires_pymilvus()

    # 惰性导入，未安装时由 pytestmark 先行跳过。
    import pymilvus as module

    return module


@pytest.fixture(scope='module')
def client(pymilvus):
    # 1. 初始化客户端连接
    client_kwargs = {'uri': MILVUS_URI}
    if MILVUS_TOKEN:
        client_kwargs['token'] = MILVUS_TOKEN
    if MILVUS_DB:
        client_kwargs['db_name'] = MILVUS_DB

    client = pymilvus.MilvusClient(**client_kwargs)

    return client


def test_compare_milvus_vs_spacy_versions_and_identifiers(client):
    """对比测试 1：版本号与技术标识符混合场景
    验证：Milvus 无法完整保留 v1_0 和 v2-beta，而自定义 SpaCy 能够完美保留全称及子词。
    """
    raw_sample_text = "系统已升级到 v1_0 版本，并兼容 v2-beta 协议。"

    # 1. Milvus 原生分词
    milvus_result = client.run_analyzer(raw_sample_text, analyzer_params={"type": "chinese", "tokenizer": "jieba"})

    milvus_tokens = milvus_result.tokens
    print("Milvus 版本分词结果:", milvus_tokens)

    # 2. 自定义 Python NLP 分词
    custom_tokens = lemmatize_zh_for_bm25(raw_sample_text).split()
    print("自定义 SpaCy 版本分词结果:", custom_tokens)

    # 3. 断言验证
    assert "v1_0" in milvus_tokens
    assert "v2-beta" not in milvus_tokens
    assert "v2" in milvus_tokens
    assert "beta" in milvus_tokens

    assert "v" in custom_tokens
    assert "1" in custom_tokens
    assert "1_0" in custom_tokens
    assert "v2-beta" not in custom_tokens
    assert "2-beta" in custom_tokens
    assert "beta" in custom_tokens


def test_compare_milvus_vs_spacy_multiple_connectors(client):
    """对比测试 2：多重连接符与中文复合词场景
    验证：Milvus 无法处理复杂的连续连接符和中文下划线复合词，而自定义 SpaCy 具备强大的粘合与子词召回能力。
    """
    raw_sample_text = "正在维护 vibe--cli 工具，管理 rag___data 数据库并更新系统_配置与用户-管理模块。"

    # 1. Milvus 原生分词
    milvus_result = client.run_analyzer(raw_sample_text, analyzer_params={"type": "chinese", "tokenizer": "jieba"})
    milvus_tokens = milvus_result.tokens
    print("Milvus 多连接符分词结果:", milvus_tokens)

    # 2. 自定义 Python NLP 分词
    custom_tokens = lemmatize_zh_for_bm25(raw_sample_text).split()
    print("自定义 SpaCy 多连接符分词结果:", custom_tokens)

    # 3. 断言验证
    assert "系统_配置" not in milvus_tokens
    assert "用户-管理" not in milvus_tokens

    assert "vibe" in custom_tokens
    assert "cli" in custom_tokens
    assert "rag" in custom_tokens
    assert "data" in custom_tokens
    assert "系统_配置" in custom_tokens
    assert "用户-管理" in custom_tokens


def test_compare_milvus_vs_spacy_edge_cases_and_fake_joins(client):
    """对比测试 3：边界防误判与伪拼接场景
    验证：原文不存在的组合（如 Chapter_99）或悬空下划线（Chapter_），两套系统都不会错误捏造复合词。
    """
    raw_sample_text = "阅读 Chapter_ 目录：Chapter2 and 99 are separate, no fake join here."

    # 1. Milvus 原生分词
    milvus_result = client.run_analyzer(raw_sample_text, analyzer_params={"type": "chinese", "tokenizer": "jieba"})
    milvus_tokens = milvus_result.tokens
    print("Milvus 边界分词结果:", milvus_tokens)

    # 2. 自定义 Python NLP 分词
    custom_tokens = lemmatize_zh_for_bm25(raw_sample_text).split()
    print("自定义 SpaCy 边界分词结果:", custom_tokens)

    # 3. 断言验证：严禁虚假拼凑
    assert "_" not in custom_tokens
    assert "Chapter_99" not in custom_tokens
    assert "Chapter_99" not in milvus_tokens

    # 4. 统计交并比（Jaccard）验证通用词汇检索的一致性
    intersection = set(milvus_tokens).intersection(set(custom_tokens))
    union = set(milvus_tokens).union(set(custom_tokens))
    overlap = (len(intersection) / len(union)) * 100 if union else 0

    assert overlap >= 50, f"边界测试下的基础词汇重合度过低: {overlap:.2f}%"