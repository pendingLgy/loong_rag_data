# 流程门面（Facade）：把各流程的调用方法集中封装于一处。
#
# 调用方只需要引入本模块或包根公共入口，无需再逐个导入 config、
# embedding、parsers、logging 等子模块：
#
#   from rag_data import RagData
#
#   app = RagData.create()            # 装配配置、日志、句柄与向量化
#   app.parse("a.md")               # 流程：解析文件为文本
#   app.chunk("a.md", text)          # 流程：按格式切块
#   app.vectorize(["一段文本"])        # 流程：向量化字符串
#
# 也可以只调用单个流程的装配方法，例如仅需要日志与向量化：
#
#   settings = build_settings()
#   logger = build_logger(settings)
#   embedder = build_embedder(settings, logger)

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import LoggerFactory
from rag_data.parsers import chunk_document, parse_document
from rag_data.parsers.base import load_nlp

ConfigSource = Optional[Mapping[str, Any]]


# ---------------- 分流程的装配方法 ----------------


def build_settings(source: ConfigSource = None, **overrides: Any) -> Settings:
    """流程一：加载配置，优先级为代码硬编码 > 环境变量 > 字段默认值。"""
    # source 为整体配置字典，overrides 以分区为单位，二者都是代码硬编码。
    # 未硬编码的字段由环境变量填充，同时存在时以硬编码为准。
    return Settings.load(source, **overrides)


def build_logger(settings: Settings) -> LoggerAdapter:
    """流程二：装配日志适配层，后端、级别、格式与时区均取自 logging 分区。"""
    logging_settings = settings.logging
    LoggerFactory.setup(
        backend=logging_settings.backend,
        timezone_name=logging_settings.timezone or None,
        level=logging_settings.level,
        fmt=logging_settings.format or None,
    )
    return LoggerFactory.get_logger()


def build_nlp(settings: Settings, logger: LoggerAdapter) -> Optional[Any]:
    """流程三：加载 spaCy 句柄；不可用时返回 None，由解析器按自身规则处理。"""
    # 加载与缓存见 rag_data.parsers.base；模型名来自 parsing 分区。
    return load_nlp(settings.parsing.spacy_model, logger)


def build_embedder(settings: Settings, logger: LoggerAdapter, model: Optional[Any] = None) -> Embedder:
    """流程四：装配向量化组件，provider 由 embedding.provider 决定。"""
    # 未注入 model 时按配置装配 provider，注入时直接使用注入对象。
    return Embedder(settings, logger, model=model)


# ---------------- 一站式门面 ----------------


class RagData:
    """按流程提供调用方法的一站式门面：一次装配，逐流程调用。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embedder: Optional[Embedder] = None,
        logger: Optional[LoggerAdapter] = None,
        nlp: Any = None,
        model: Optional[Any] = None,
    ) -> None:
        self.settings = settings if settings is not None else build_settings()
        self.logger = logger if logger is not None else build_logger(self.settings)
        self.nlp = nlp if nlp is not None else build_nlp(self.settings, self.logger)
        self.embedder = (
            embedder if embedder is not None else build_embedder(self.settings, self.logger, model=model)
        )

    @classmethod
    def create(
        cls,
        source: ConfigSource = None,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "RagData":
        """按配置源创建门面实例；overrides 可在代码中硬编码分区配置。"""
        # kwargs 传给构造器（embedder、logger、nlp 等），overrides 归入配置。
        return cls(build_settings(source, **(overrides or {})), **kwargs)

    # ---------------- 流程方法 ----------------

    def parse(self, path: str) -> str:
        """流程：解析单个文件为纯文本。"""
        return parse_document(path, logger=self.logger)

    def chunk(self, path: str, text: str) -> List[str]:
        """流程：按文件扩展名切块，规则由对应解析器自持，长度取自 parsing 分区。"""
        parsing = self.settings.parsing
        return chunk_document(
            path, text, nlp=self.nlp, logger=self.logger,
            max_chars=parsing.max_chars, safe_max_chars=parsing.safe_max_chars,
        )

    def vectorize(self, texts: Sequence[str]) -> List[List[float]]:
        """流程：批量向量化字符串，返回与输入同序的稠密向量。"""
        return self.embedder.encode(list(texts))

    def vectorize_text(self, text: str) -> List[float]:
        """流程：向量化单个字符串，返回其稠密向量。"""
        return self.embedder.encode([text])[0]


# ---------------- 一键调用入口 ----------------


def vectorize(
    texts: Sequence[str],
    source: ConfigSource = None,
    **kwargs: Any,
) -> List[List[float]]:
    """一键向量化字符串：内部完成全部流程装配，返回与输入同序的稠密向量。"""
    return RagData.create(source, **kwargs).vectorize(texts)


__all__ = [
    "RagData",
    "build_settings",
    "build_logger",
    "build_nlp",
    "build_embedder",
    "vectorize",
]

