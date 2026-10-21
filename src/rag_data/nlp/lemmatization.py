"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging

from rag_data.nlp.spacy import get_nlp_lemma

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """

    nlp_lemma = get_nlp_lemma()
    if nlp_lemma is None:
        return text

    doc = nlp_lemma(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)

def lemmatize_zh_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """

    nlp_lemma = get_nlp_lemma()
    if nlp_lemma is None:
        return text

    doc = nlp_lemma(text)  # 中文不需要特意转 .lower()
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        # 优先使用 token.text（兼容中文和英文原形），如果模型有 lemma 也可以做兜底
        word = token.text
        if word.isalnum():
            tokens.append(word)

        # 英文的 -ing 规则保留，对中文不影响
        if token.text.endswith("ing") and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)
