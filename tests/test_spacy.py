from rag_data.nlp.lemmatization import lemmatize_for_bm25, lemmatize_zh_for_bm25
from rag_data.nlp.spacy import get_nlp_lemma


def test_get_nlp_lemma():
    nlp = get_nlp_lemma()
    doc = nlp("你好，我是谁")

    # 1. 获取所有切分出来的词语列表
    tokens = [token.text for token in doc]
    print("分词列表:", tokens)

    # 2. 如果需要获取词形还原（Lemmatization）的结果
    lemmas = [token.lemma_ for token in doc]
    print("词形还原:", lemmas)

    # 3. 打印详细的词语及其属性
    for token in doc:
        print(f"词语: {token.text}, 词性: {token.pos_}, 还原形: {token.lemma_}")


def test_lemmatize_for_bm25():
    str_val = "我是Long，我是IT研发"
    lemm = lemmatize_zh_for_bm25(str_val)
    print(lemm)


def test_lemmatize_for_bm25_basic_and_underscores():
    """测试用例 1：基础混合场景，验证中英文及带有下划线的技术标识符（Chapter_1 等）的子词与全称顺序。"""
    text = "目录Content Chapter_1 Chapter_2 Chapter_3"
    lemm = lemmatize_zh_for_bm25(text)
    # 根据新的拆分与组合规则，期望输出依次包含左词、右词以及完整复合词
    expected = "目录 Content Chapter 1 Chapter_1 Chapter 2 Chapter_2 Chapter 3 Chapter_3"
    assert lemm == expected


def test_lemmatize_for_bm25_number():
    """测试用例 2：数字与连字符场景。"""
    text = "目录Content Chapter1 Chapter2 Chapter3"
    lemm = lemmatize_zh_for_bm25(text)
    assert lemm == '目录 Content Chapter 1 Chapter 2 Chapter 3'


def test_lemmatize_for_bm25_english_ing_rule():
    """测试用例 3：验证英文 -ing 规则的保留以及标点/停用词的自动过滤。"""
    text = "The system is processing data and running tasks smoothly."
    lemm = lemmatize_zh_for_bm25(text)
    assert "processing" in lemm
    assert "running" in lemm
    assert "system" in lemm


def test_lemmatize_for_bm25_fake_underscore():
    """测试用例 4：验证“原文强校验（candidate in text）”机制。
    如果分词器异常把不相干的词通过下划线拼凑，但原文中并不存在，则不应该被错误合并。
    """
    text = "Chapter and 99 are separate."
    lemm = lemmatize_zh_for_bm25(text)
    assert "Chapter_99" not in lemm


def test_lemmatize_for_bm25_pure_chinese_and_tech():
    """测试用例 5：纯中文、代码下划线混合场景（如 vibe_cli 仓库路径）。"""
    text = "在本地路径 F:/workspace/code/python/vibe_cli 中运行异步事件。"
    lemm = lemmatize_zh_for_bm25(text)
    assert "vibe_cli" in lemm
    assert "异步" in lemm
    assert "事件" in lemm


def test_vibe_cli_decomposition_and_retention():
    """专用测试：验证 vibe_cli 能够同时解析并返回子词 vibe、cli 以及全称 vibe_cli。"""
    text = "vibe_cli"
    lemm = lemmatize_zh_for_bm25(text)
    tokens = lemm.split()
    assert "vibe" in tokens
    assert "cli" in tokens
    assert "vibe_cli" in tokens


def test_boundary_empty_and_pure_symbols():
    """边界测试 1：空字符串、纯空白、纯标点、纯连接符等极端的输入鲁棒性。"""
    assert lemmatize_zh_for_bm25("") == ""
    assert lemmatize_zh_for_bm25("   ") == ""
    assert lemmatize_zh_for_bm25("!@#$%^&*()") == ""
    assert lemmatize_zh_for_bm25("--- ___ —-") == ""


def test_boundary_single_chars_and_numbers():
    """边界测试 2：单字符、纯数字以及版本号复合标识符。"""
    text = "a b 1 2 v1_0 v2-beta"
    lemm = lemmatize_zh_for_bm25(text)
    tokens = lemm.split()
    assert "a" in tokens
    assert "b" in tokens
    assert "1" in tokens
    assert "2" in tokens
    assert "1_0" in tokens
    assert "2-beta" in tokens
    assert "beta" in tokens


def test_scenario_multiple_symbols_combination():
    """多种符号组合与多连接符测试：验证混合使用下划线、短横线、连续连接符的场景（如 vibe--cli, rag___data）。"""
    text = "项目 vibe--cli 与 rag___data，系统_配置-模块的组合测试。"
    lemm = lemmatize_zh_for_bm25(text)
    tokens = lemm.split()

    # 验证复合词及拆分子词均正确保留
    assert "vibe" in tokens
    assert "cli" in tokens
    assert "rag" in tokens
    assert "data" in tokens
    assert "系统_配置" in tokens
    assert "模块" in tokens
    assert "组合" in tokens
    assert "测试" in tokens


def test_scenario_dangling_and_invalid_connectors():
    """边界测试 3：悬空连字符或无效拼接场景。"""
    # 1. 结尾悬空下划线
    text1 = "Chapter_"
    tokens = lemmatize_zh_for_bm25(text1)
    assert "_" not in tokens

    # 2. 原文中拼凑不存在的组合
    text2 = "Chapter2 and 99 are separate."
    lemm2 = lemmatize_zh_for_bm25(text2)
    assert "Chapter_99" not in lemm2


def test_scenario_chinese_connectors():
    """测试：中文与下划线/连字符混合场景。"""
    text = "系统_配置与用户-管理模块正在运行。"
    lemm = lemmatize_zh_for_bm25(text)
    assert "系统_配置" in lemm
    assert "用户-管理" in lemm
    assert "模块" in lemm
    assert "运行" in lemm