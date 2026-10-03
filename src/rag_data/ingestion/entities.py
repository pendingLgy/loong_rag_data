# 实体抽取：使用调用方注入的 spaCy 句柄，未注入时不做抽取。

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set

# 实体两端常见的包裹标点与空白，归一化时剥除。
_TRIM_CHARS = " \t\r\n'“”‘’()（）[]【】《》「」『』"


def extract_entities(
    text: str,
    nlp: Any = None,
    *,
    labels: Optional[Iterable[str]] = None,
) -> List[str]:
    """抽取实体：归一化后去重，保持首次出现顺序；未注入句柄时返回空列表。"""
    # 句柄由调用方注入（通常来自 RagData.nlp 或 pipeline 的 nlp 参数）；
    # nlp 为 None 表示不做实体抽取。labels 非空时仅保留命中的实体类型。
    if not text or not text.strip() or nlp is None:
        return []
    allowed = {str(label).upper() for label in labels} if labels else None
    seen: Set[str] = set()
    result: List[str] = []
    for entity in _iter_entities(nlp, text):
        value = _normalize(getattr(entity, "text", ""))
        if not value or not _is_meaningful(value):
            continue
        if allowed is not None and _label(entity).upper() not in allowed:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _iter_entities(handle: Any, text: str) -> Iterable[Any]:
    """运行句柄并返回实体序列；句柄无 ents 时视为空。"""
    doc = handle(text)
    return getattr(doc, "ents", None) or []


def _normalize(value: Any) -> str:
    """归一化实体文本：去首尾空白，并剥除包裹的引号或括号。"""
    return str(value).strip().strip(_TRIM_CHARS).strip()


def _is_meaningful(value: str) -> bool:
    """过滤纯标点或纯空白等无信息量的实体。"""
    return any(char.isalnum() for char in value)


def _label(entity: Any) -> str:
    """实体类型标签；句柄未提供 label_ 时返回空串。"""
    return str(getattr(entity, "label_", "") or "")
