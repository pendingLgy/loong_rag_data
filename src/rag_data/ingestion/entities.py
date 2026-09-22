# 基于 spaCy NER 的实体抽取。

from __future__ import annotations

from typing import Any, List, Optional


def extract_entities(text: str, nlp: Optional[Any] = None) -> List[str]:
    """抽取实体并去重，保持首次出现顺序。"""
    if not text or not text.strip() or nlp is None:
        return []
    # TODO: nlp 应由调用方注入加载了 zh_core_web_sm 的句柄并启用 ner；
    #       未注入时返回空列表，待人工补充默认加载与实体归一策略。
    doc = nlp(text)
    seen = set()
    result: List[str] = []
    for entity in doc.ents:
        value = entity.text.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
