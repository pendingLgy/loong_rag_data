# 两阶段切块中句级切分逻辑的单元测试。

from rag_data.ingestion.chunking import build_semantic_chunks, _split_by_punctuation


class _FakeSent:
    def __init__(self, text):
        self.text = text


class _FakeDoc:
    def __init__(self, sents):
        self.sents = sents


def _fake_nlp(text):
    return _FakeDoc([_FakeSent("第一句。"), _FakeSent("第二句。")])


class _RecordingLogger:
    def __init__(self):
        self.events = []

    def warning(self, msg, **fields):
        self.events.append((msg, fields))


def test_empty_text_returns_no_chunks():
    assert build_semantic_chunks("") == []
    assert build_semantic_chunks("   ") == []


def test_single_sentence_becomes_single_chunk():
    assert build_semantic_chunks("只有一句话。") == ["只有一句话。"]


def test_split_by_punctuation():
    assert _split_by_punctuation("你好。世界！真的吗？好") == ["你好。", "世界！", "真的吗？", "好"]


def test_chunks_split_and_overlap():
    text = "甲甲乙乙。丙丙丁丁。戊戊己己。"
    no_overlap = build_semantic_chunks(text, max_chars=10, overlap_sents=0)
    assert no_overlap == ["甲甲乙乙。 丙丙丁丁。", "戊戊己己。"]
    with_overlap = build_semantic_chunks(text, max_chars=10, overlap_sents=1)
    assert with_overlap[0] == "甲甲乙乙。 丙丙丁丁。"
    assert with_overlap[1] == "丙丙丁丁。 戊戊己己。"


def test_overlap_larger_than_accumulated_is_safe():
    chunks = build_semantic_chunks("甲甲乙乙。丙丙丁丁。", max_chars=10, overlap_sents=5)
    assert chunks


def test_injected_nlp_is_used():
    assert build_semantic_chunks("任意文本", nlp=_fake_nlp) == ["第一句。 第二句。"]


def test_long_sentence_warning_logged():
    recorder = _RecordingLogger()
    long_sentence = "L" * 20
    text = "短。" + long_sentence
    chunks = build_semantic_chunks(text, max_chars=5, overlap_sents=0, logger=recorder)
    assert chunks == ["短。", long_sentence]
    assert recorder.events


def test_long_sentence_without_logger_is_silent():
    long_sentence = "L" * 20
    chunks = build_semantic_chunks("短。" + long_sentence, max_chars=5)
    assert len(chunks) == 2
