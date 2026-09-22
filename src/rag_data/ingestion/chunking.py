# 两阶段语义切块中的句级切分与重叠聚合。

from __future__ import annotations

from typing import Any, List, Optional

from rag_data.logging.base import LoggerAdapter


def build_semantic_chunks(
    text: str,
    max_chars: int = 250,
    overlap_sents: int = 1,
    nlp: Optional[Any] = None,
    logger: Optional[LoggerAdapter] = None,
) -> List[str]:
    """将文本按句聚合为不超过 max_chars 的块，块间保留 overlap_sents 句重叠。"""
    if not text or not text.strip():
        return []
    sentences = _split_sentences(text, nlp)
    if not sentences:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        if not current:
            current.append(sentence)
            current_len = len(sentence)
            continue
        if current_len + len(sentence) <= max_chars:
            current.append(sentence)
            current_len += len(sentence)
            continue
        chunks.append(" ".join(current))
        if logger is not None and len(sentence) > max_chars:
            logger.warning("检测到超长单句，单独成块", length=len(sentence), max_chars=max_chars)
        tail = current[-overlap_sents:] if overlap_sents > 0 else []
        current = list(tail)
        current.append(sentence)
        current_len = sum(len(item) for item in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_sentences(text: str, nlp: Optional[Any]) -> List[str]:
    """优先使用注入的 spaCy 句柄切句，否则回退到标点切分。"""
    if nlp is not None:
        doc = nlp(text)
        return [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    # TODO: 生产环境应注入加载了 zh_core_web_sm 并启用 sentencizer 的句柄，
    #       此处回退实现仅按中英文标点切分，精度有限。
    return _split_by_punctuation(text)


def _split_by_punctuation(text: str) -> List[str]:
    """按常见中英文标点切句的回退实现。"""
    terminators = set("。！？!?；;")
    sentences: List[str] = []
    for line in text.splitlines():
        buffer: List[str] = []
        for char in line:
            buffer.append(char)
            if char in terminators:
                sentence = "".join(buffer).strip()
                if sentence:
                    sentences.append(sentence)
                buffer = []
        tail = "".join(buffer).strip()
        if tail:
            sentences.append(tail)
    return sentences
