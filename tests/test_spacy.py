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
    str = "我是Long，我是IT研发"
    lemm = lemmatize_zh_for_bm25(str)
    print(lemm)