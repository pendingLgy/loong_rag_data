# nlp 子包：spaCy 句柄的加载、缓存与清空；解析器与门面共用。
#
#   from rag_data.nlp import load_nlp, clear_cache, DEFAULT_MODEL

from rag_data.nlp.spacy import DEFAULT_MODEL, clear_cache, ensure_sentence_boundaries, load_nlp

__all__ = [
    "DEFAULT_MODEL",
    "load_nlp",
    "clear_cache",
    "ensure_sentence_boundaries",
]
