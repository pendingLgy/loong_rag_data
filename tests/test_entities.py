# 实体抽取的单元测试。

from rag_data.ingestion.entities import extract_entities


class _Entity:
    def __init__(self, text):
        self.text = text


class _Doc:
    def __init__(self, ents):
        self.ents = ents


def _nlp(values):
    def _call(text):
        return _Doc([_Entity(value) for value in values])

    return _call


def test_empty_text_or_missing_nlp_returns_empty():
    assert extract_entities("", _nlp(["X"])) == []
    assert extract_entities("   ", _nlp(["X"])) == []
    assert extract_entities("内容", None) == []


def test_dedup_is_case_insensitive():
    assert extract_entities("文本", _nlp(["Alpha", "alpha", "Beta"])) == ["Alpha", "Beta"]


def test_blank_entities_are_skipped():
    assert extract_entities("文本", _nlp(["  ", "Real"])) == ["Real"]
