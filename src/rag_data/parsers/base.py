# 文档解析器基类：一种格式一个子类，parse 与 chunk 均由子类实现。
#
# 句柄：未装 spaCy 或模型缺失时返回 None，由调用方降级（标点回退）；
# 成功与失败都缓存，避免逐次重试拖慢批量导入，环境修复后 clear_cache() 可重建。
#
# nlp 约定：由装配层（facade）加载后逐级注入；子类只消费注入的句柄，nlp 为 None 时按各子类规则处理。

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Union

from rag_data.logging.base import LoggerAdapter

DEFAULT_MODEL = "zh_core_web_sm"

# 失败哨兵：与「尚未加载」区分，避免每次调用都重复尝试加载。
_FAILED = object()
_PIPELINES: Dict[str, Any] = {}


class DocumentParser(ABC):
    """文档解析器：子类声明负责的后缀并实现 parse、chunk；句柄加载由基类提供。"""

    # 该解析器负责的扩展名（小写、含点）；子类必须覆盖，且不得与其它解析器重复。
    SUFFIXES: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, max_chars: int = 500, safe_max_chars = 2000):
        self.max_chars = max_chars
        self.safe_max_chars = safe_max_chars

    @abstractmethod
    def parse(self, path: str, logger: Optional[LoggerAdapter] = None) -> str:
        """解析单个文件并返回纯文本。"""

    @abstractmethod
    def chunk(
        self,
        text: str,
        nlp: Any = None,
        logger: Optional[LoggerAdapter] = None,
    ) -> List[str]:
        """把纯文本切分为块；nlp 为注入的句柄，None 时各子类按自身规则处理。"""

    # ---------------- 文件读取 ----------------

    @staticmethod
    def _read_text(path: str) -> str:
        """读取 UTF-8 文本，供无结构格式共用。"""
        with open(path, "r", encoding="utf-8") as stream:
            return stream.read()

    # ---------------- 句柄加载 ----------------

    @classmethod
    def load_nlp(cls, model_name: str = DEFAULT_MODEL, logger: Optional[LoggerAdapter] = None) -> Optional[Any]:
        """加载 spaCy 句柄：补齐 sentencizer 以支持句级切分，保留 ner 供实体抽取。"""
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
            nlp = spacy.load(model_name)
        except Exception as exc:  # noqa: BLE001 模型缺失或加载失败均降级
            _PIPELINES[model_name] = _FAILED
            if logger is not None:
                logger.warning("spaCy 模型加载失败，将使用回退实现", model=model_name, error=str(exc))
            return None
        cls._ensure_sentence_boundaries(nlp)
        _PIPELINES[model_name] = nlp
        if logger is not None:
            logger.info("spaCy 模型加载完成", model=model_name, pipes=list(nlp.pipe_names))
        return nlp

    @classmethod
    def clear_cache(cls) -> None:
        """清空句柄缓存（含失败记录），供测试或运行时重建句柄使用。"""
        _PIPELINES.clear()

    @staticmethod
    def _ensure_sentence_boundaries(nlp: Any) -> None:
        """句级切分依赖 parser 或 sentencizer；两者都缺时补一个 sentencizer。"""
        names = list(getattr(nlp, "pipe_names", None) or [])
        if "parser" not in names and "sentencizer" not in names:
            nlp.add_pipe("sentencizer")

    def split_sentences(self, text: str, nlp: Any = None) -> List[str]:
        """使用 spaCy 进行中英文及复杂缩写的精确分句，并按换行符二次切分。"""
        if not text:
            return []

        doc = nlp(text)
        raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # 如果 spaCy 没有切出任何内容，退化为原文本
        if not raw_sentences:
            raw_sentences = [text]

        # 按换行符再次切分，并清洗空字符串
        final_sentences = []
        for sent in raw_sentences:
            # 按 '\n' 拆分，去除两端空白，过滤掉拆分后的空行
            # keepends=True 会保留行尾的 \n
            sub_sents = [line for line in sent.splitlines(keepends=True) if line.strip()]
            final_sentences.extend(sub_sents)

        return final_sentences

    def _extract_overlap_prefix(self, chunk_sentences: List[str]) -> str:
        """
        从当前段提取累计长度达到 1/4 (25%) 的完整句子/行作为前缀。
        针对包含 \\n 的单条大文本，会自动先按换行符二次切分为多行，确保按行精准提取。
        """
        if not chunk_sentences:
            return ""

        # 1. 如果传入的数据包含换行符，先按 \n 展开拆分为独立的行/句
        flat_sentences = []
        for item in chunk_sentences:
            if "\n" in item:
                # 按换行符切分，并过滤掉纯空白行
                lines = [line for line in item.split("\n") if line.strip()]
                flat_sentences.extend(lines)
            elif item.strip():
                flat_sentences.append(item.strip())

        if not flat_sentences:
            return ""

        # 2. 计算展平后的总字符数与 1/4 门槛
        total_len = sum(len(s) for s in flat_sentences)
        target_overlap_len = total_len / 4.0

        overlap_sents = []
        accumulated_len = 0

        # 3. 从后往前倒序累加完整的单行/单句
        for sent in reversed(flat_sentences):
            overlap_sents.insert(0, sent)
            accumulated_len += len(sent)

            # 核心规则：达到或超过 25% 字符数时停止，保证只多不少且不砸断单行
            if accumulated_len >= target_overlap_len:
                break

        # 4. 用 \n 拼接还原为带换行格式的前缀文本
        return "\n".join(overlap_sents)

    def _process_text_recursive(
            self,
            text: str,
            overlap_prefix: str = "",
            nlp=Any
    ) -> List[str]:
        """
        纯 str 入参递归切分：
        1. 当 text 文本长度 <= max_chars 时，不切分句子，直接拼上前缀返回；
        2. 当 text 文本长度 > max_chars 时，才切分句子并按不超过 max_chars 组装；
        3. 重叠前缀（overlap_prefix）不占用 max_chars 的长度限制；
        4. 截取未处理的剩余 text（str）递归调用自身。
        """
        if not text or not text.strip():
            return []

        # =========================================================================
        # 1. 递归基线条件：只有当 text 整体长度超过 max_chars 时才切分
        # =========================================================================
        if len(text) <= self.max_chars:
            final_chunk = overlap_prefix + ("\n" if overlap_prefix and len(overlap_prefix) > 0 else "") + text
            return [final_chunk]

        # =========================================================================
        # 2. 文本大于 max_chars 时，才进行句子切分
        # =========================================================================
        sentences = self.split_sentences(text, nlp)

        current_chunk_sents = []
        current_len = 0
        consumed_char_count = 0  # 记录当前段在原始 text 中消耗的字符偏移

        for sent in sentences:
            sent_len = len(sent)

            # 【修改点 1】：如果第一句就超过了 max_chars，进行强行硬切分处理
            if not current_chunk_sents and sent_len > self.max_chars:
                # 截取前 max_chars 字符作为当前句
                hard_cut_sent = sent[:self.max_chars]
                current_chunk_sents.append(hard_cut_sent)
                current_len = len(hard_cut_sent)
                consumed_char_count = text.find(hard_cut_sent, consumed_char_count) + len(hard_cut_sent)
                break

            # 累加句子本体（不计入 overlap_prefix），超过 max_chars 则停止
            if current_chunk_sents and (current_len + sent_len > self.max_chars):
                break

            current_chunk_sents.append(sent)
            current_len += sent_len

            # 计算在原字符串中的消耗偏移
            match_idx = text.find(sent, consumed_char_count)
            if match_idx != -1:
                consumed_char_count = match_idx + len(sent)
            else:
                consumed_char_count += len(sent)

        # 拼接当前段落（加上上一段的 1/4 重叠前缀）
        current_chunk_text = overlap_prefix + ("\n" if overlap_prefix and len(overlap_prefix) > 0 else "") + "".join(current_chunk_sents)

        # 3. 提取当前段句子本体末尾 1/4 作为下一个段落的前缀
        next_overlap_prefix = self._extract_overlap_prefix(current_chunk_sents)

        # 4. 截取未处理的剩余文本 str，递归处理
        # 【修改点 2】：安全兜底，避免 offset 未推进导致死循环
        if consumed_char_count == 0:
            consumed_char_count = self.max_chars

        remaining_text = text[consumed_char_count:].lstrip()
        rest_chunks = self._process_text_recursive(
            text=remaining_text,
            overlap_prefix=next_overlap_prefix,
            nlp=nlp
        )

        return [current_chunk_text] + rest_chunks

# 模块级入口：实体抽取、流程门面等非解析器模块直接复用句柄加载。
load_nlp = DocumentParser.load_nlp
clear_cache = DocumentParser.clear_cache
