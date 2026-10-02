# 流程门面（Facade）：把各流程的调用方法集中封装于一处。
#
# 调用方只需要引入本模块或包根公共入口，无需再逐个导入 config、
# embedding、ingestion、logging 等子模块：
#
#   from rag_data import RagData
#
#   app = RagData.create()          # 装配全部流程
#   app.ingest([docs.md])           # 流程：向量化文件
#   app.vectorize(["一段文本"])      # 流程：向量化字符串
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
from rag_data.ingestion.pipeline import IngestionPipeline
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import configure_logging
from rag_data.models import DocumentChunk

ConfigSource = Optional[Mapping[str, Any]]
_UNSET: Any = object()


# ---------------- 分流程的装配方法 ----------------


def build_settings(source: ConfigSource = None, **overrides: Any) -> Settings:
    """流程一：加载配置，优先级为代码硬编码 > 环境变量 > 字段默认值。"""
    # source 为整体配置字典，overrides 以分区为单位，二者都是代码硬编码。
    # 未硬编码的字段由环境变量填充，同时存在时以硬编码为准。
    return Settings.load(source, **overrides)


def build_logger(settings: Settings) -> LoggerAdapter:
    """流程二：装配日志适配层，按配置选择 stdlib、loguru 或 structlog。"""
    return configure_logging(settings)


def build_nlp(settings: Settings, logger: LoggerAdapter) -> Optional[Any]:
    """流程三：加载 spaCy 句柄；不可用时返回 None，由回退实现接管。"""
    model_name = settings.nlp.spacy_model
    try:
        import spacy
    except ImportError:
        logger.warning("spaCy 未安装，分句与实体抽取将使用回退实现", model=model_name)
        return None
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:  # noqa: BLE001 模型缺失或加载失败均降级
        logger.warning("spaCy 模型加载失败，将使用回退实现", model=model_name, error=str(exc))
        return None
    # TODO: 按流程细化管道配置——切块需要句子边界，实体抽取需要 ner；
    #       当前仅在缺少句法分析器时补一个 sentencizer 兜底。
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    logger.info("spaCy 模型加载完成", model=model_name, pipes=list(nlp.pipe_names))
    return nlp


def build_embedder(settings: Settings, logger: LoggerAdapter, model: Optional[Any] = None) -> Embedder:
    """流程四：装配向量化组件，provider 由 embedding.provider 决定。"""
    # 未注入 model 时按配置装配 provider，注入时直接使用注入对象。
    return Embedder(settings, logger, model=model)


def build_pipeline(
    settings: Settings,
    embedder: Embedder,
    logger: LoggerAdapter,
    nlp: Optional[Any] = None,
) -> IngestionPipeline:
    """流程五：装配离线导入管道。"""
    return IngestionPipeline(embedder, settings, logger, nlp=nlp)


# ---------------- 一站式门面 ----------------


class RagData:
    """按流程提供调用方法的一站式门面：一次装配，逐流程调用。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embedder: Optional[Embedder] = None,
        logger: Optional[LoggerAdapter] = None,
        nlp: Any = _UNSET,
        model: Optional[Any] = None,
    ) -> None:
        self.settings = settings if settings is not None else build_settings()
        self.logger = logger if logger is not None else build_logger(self.settings)
        self.nlp = build_nlp(self.settings, self.logger) if nlp is _UNSET else nlp
        self.embedder = (
            embedder if embedder is not None else build_embedder(self.settings, self.logger, model=model)
        )
        self.pipeline = build_pipeline(self.settings, self.embedder, self.logger, nlp=self.nlp)

    @classmethod
    def create(
        cls,
        source: ConfigSource = None,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "RagData":
        """按配置源创建门面实例；overrides 可在代码中硬编码分区配置。"""
        # kwargs 传给构造器（embedder、logger 等），overrides 归入配置。
        return cls(build_settings(source, **(overrides or {})), **kwargs)

    # ---------------- 流程方法 ----------------

    def ingest(self, paths: Sequence[str], user_id: Optional[str] = None) -> List[DocumentChunk]:
        """流程：批量向量化文件，返回向量化切块。"""
        return self.pipeline.run(list(paths), user_id=user_id)

    def ingest_file(self, path: str, user_id: Optional[str] = None) -> List[DocumentChunk]:
        """流程：向量化单个文件，返回向量化切块。"""
        return self.pipeline.ingest_file(path, user_id=user_id)

    def vectorize(self, texts: Sequence[str]) -> List[List[float]]:
        """流程：批量向量化字符串，返回与输入同序的稠密向量。"""
        return self.embedder.encode(list(texts))

    def vectorize_text(self, text: str) -> List[float]:
        """流程：向量化单个字符串，返回其稠密向量。"""
        return self.embedder.encode([text])[0]


# ---------------- 一键调用入口 ----------------


def ingest(
    paths: Sequence[str],
    source: ConfigSource = None,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> List[DocumentChunk]:
    """一键向量化：内部完成全部流程装配，返回向量化切块。"""
    return RagData.create(source, **kwargs).ingest(paths, user_id=user_id)


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
    "build_pipeline",
    "ingest",
    "vectorize",
]
