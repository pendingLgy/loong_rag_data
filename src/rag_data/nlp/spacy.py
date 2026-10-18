# spaCy 句柄加载：惰性导入、按模型名缓存、失败也缓存，避免逐次重试拖慢批量导入。
#
# 缓存策略：
#   成功 -> 句柄对象
#   失败 -> _FAILED 哨兵（与「尚未加载」区分），环境修复后调用 clear_cache() 重建。
#
# 文件名与第三方包同名：本模块位于 rag_data.nlp 下，Python 3 只做绝对导入，
# 因此函数内的 import spacy 仍解析到第三方包，不会自我引用。

from __future__ import annotations

from typing import Any, Dict, Optional

from rag_data.logging.base import LoggerAdapter

DEFAULT_MODEL = "zh_core_web_md"

# 失败哨兵：与「尚未加载」区分，避免每次调用都重复尝试加载。
_FAILED = object()
_PIPELINES: Dict[str, Any] = {}


def load_nlp(model_name: str = DEFAULT_MODEL, logger: Optional[LoggerAdapter] = None) -> Optional[Any]:
    """加载 spaCy 句柄：补 sentencizer 以支持句级切分；未装 spaCy 或模型缺失时返回 None。"""
    cached = _PIPELINES.get(model_name)
    if cached is _FAILED:
        return None
    if cached is not None:
        return cached
    try:
        import spacy
    except ImportError:
        _PIPELINES[model_name] = _FAILED
        if logger is not None:
            logger.warning("spaCy 未安装，分句与实体抽取将使用回退实现", model=model_name)
        return None
    try:
        nlp = spacy.load(model_name, disable=["tagger", "ner", "attribute_ruler"])
    except Exception as exc:  # noqa: BLE001 模型缺失或加载失败均降级
        _PIPELINES[model_name] = _FAILED
        if logger is not None:
            logger.warning("spaCy 模型加载失败，将使用回退实现", model=model_name, error=str(exc))
        return None
    ensure_sentence_boundaries(nlp)
    _PIPELINES[model_name] = nlp
    if logger is not None:
        logger.info("spaCy 模型加载完成", model=model_name, pipes=list(nlp.pipe_names))
    return nlp


def clear_cache() -> None:
    """清空句柄缓存（含失败记录），供测试或运行时重建句柄使用。"""
    _PIPELINES.clear()


def ensure_sentence_boundaries(nlp: Any) -> None:
    """句级切分依赖 parser 或 sentencizer；两者都缺时补一个 sentencizer。"""
    names = list(getattr(nlp, "pipe_names", None) or [])
    if "parser" not in names and "sentencizer" not in names:
        nlp.add_pipe("sentencizer")


