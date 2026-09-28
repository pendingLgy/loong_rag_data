# 流程门面（Facade）：把各流程的调用方法集中封装于一处。
#
# 调用方只需要引入本模块或包根公共入口，无需再逐个导入 config、storage、
# embedding、ingestion、logging 等子模块：
#
#   from rag_data import RagData
#
#   app = RagData.create()          # 装配全部流程
#   app.init_collection()           # 流程：建集合
#   app.ingest([docs.md])           # 流程：批量导入
#   app.query(vector, top_n=5)      # 流程：按向量检索
#
# 也可以只调用单个流程的装配方法，例如仅需要日志与存储：
#
#   settings = build_settings()
#   logger = build_logger(settings)
#   store = build_store(settings, logger)

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

from rag_data.config import Settings
from rag_data.embedding.embedder import Embedder
from rag_data.ingestion.pipeline import IngestionPipeline
from rag_data.logging.base import LoggerAdapter
from rag_data.logging.factory import configure_logging
from rag_data.models import MemoryRecord, QueryHit, resolve_record_class
from rag_data.storage.base import BaseVectorStore
from rag_data.storage.registry import create_store
from rag_data.storage.memory_store import InMemoryVectorStore

DEFAULT_TOP_N = 5
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


def build_record_class(settings: Settings) -> Type[MemoryRecord]:
    """流程：解析配置中的记录类路径；默认 MilvusRecord，可按需继承扩展字段。"""
    return resolve_record_class(settings.models.record_class)


def build_store(
    settings: Settings,
    logger: LoggerAdapter,
    record_class: Optional[Type[MemoryRecord]] = None,
) -> BaseVectorStore:
    """流程三：按 settings.storage.backend 从注册表装配向量库。"""

    # 后端实现由注册表解析：继承 BaseVectorStore 即自动注册，
    # 因此新增存储方式只需实现类并在配置里写后端名，无需改动本函数。
    cls = record_class if record_class is not None else build_record_class(settings)
    # 表结构由记录类声明，装配时无需再传入额外的列。
    return create_store(
        settings.storage.backend,
        settings,
        logger,
        record_class=cls,
    )


def build_nlp(settings: Settings, logger: LoggerAdapter) -> Optional[Any]:
    """流程四：加载 spaCy 句柄；不可用时返回 None，由回退实现接管。"""
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
    """流程五：装配向量化组件，provider 由 embedding.provider 决定。"""
    # 未注入 model 时按配置装配 provider，注入时直接使用注入对象。
    return Embedder(settings, logger, model=model)


def build_pipeline(
    settings: Settings,
    store: BaseVectorStore,
    embedder: Embedder,
    logger: LoggerAdapter,
    nlp: Optional[Any] = None,
    record_class: Optional[Type[MemoryRecord]] = None,
) -> IngestionPipeline:
    """流程六：装配离线导入管道。"""
    cls = record_class if record_class is not None else build_record_class(settings)
    return IngestionPipeline(store, embedder, settings, logger, nlp=nlp, record_class=cls)


# ---------------- 一站式门面 ----------------


class RagData:
    """按流程提供调用方法的一站式门面：一次装配，逐流程调用。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        store: Optional[BaseVectorStore] = None,
        embedder: Optional[Embedder] = None,
        logger: Optional[LoggerAdapter] = None,
        nlp: Any = _UNSET,
        model: Optional[Any] = None,
        record_class: Optional[Type[MemoryRecord]] = None,
) -> None:
        self.settings = settings if settings is not None else build_settings()
        self.logger = logger if logger is not None else build_logger(self.settings)
        # 记录类由配置 models.record_class 决定，写入与读回保持一致。
        self.record_class = (
            record_class if record_class is not None else build_record_class(self.settings)
        )
        self.store = (
            store
            if store is not None
            else build_store(self.settings, self.logger, record_class=self.record_class)
        )
        self.nlp = build_nlp(self.settings, self.logger) if nlp is _UNSET else nlp
        self.embedder = (
            embedder if embedder is not None else build_embedder(self.settings, self.logger, model=model)
        )
        self.pipeline = build_pipeline(
            self.settings,
            self.store,
            self.embedder,
            self.logger,
            nlp=self.nlp,
            record_class=self.record_class,
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
        # kwargs 传给构造器（store、embedder、logger 等），overrides 归入配置。
        return cls(build_settings(source, **(overrides or {})), **kwargs)

    # ---------------- 流程方法 ----------------

    def init_collection(self) -> None:
        """流程：初始化集合与索引。"""
        self.store.ensure_collection()
        self.logger.info(
            "集合初始化完成",
            backend=self.settings.storage.backend,
            collection=self.settings.storage.collection_name,
        )

    def ingest(self, paths: Sequence[str], user_id: Optional[str] = None) -> int:
        """流程：批量导入文件，返回写入总条数。"""
        return self.pipeline.run(list(paths), user_id=user_id)

    def ingest_file(self, path: str, user_id: Optional[str] = None) -> int:
        """流程：导入单个文件，返回写入条数。"""
        return self.pipeline.ingest_file(path, user_id=user_id)

    def query(
        self,
        vector: List[float],
        top_n: Optional[int] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryHit]:
        """流程：按向量检索候选，按相似度降序返回。"""
        # filters 支持基类字段与扩展字段；user_id 为常用项，等价于 filters 中的同名字段。
        merged: Dict[str, Any] = dict(filters) if filters else {}
        if user_id is not None:
            merged["user_id"] = user_id
        limit = DEFAULT_TOP_N if top_n is None else top_n
        return self.store.query(vector, top_n=limit, filters=merged or None)


    def close(self) -> None:
        """流程：释放底层资源。"""
        self.store.close()

    def __enter__(self) -> "RagData":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# ---------------- 一键调用入口 ----------------


def init_collection(source: ConfigSource = None, **kwargs: Any) -> None:
    """一键建表：内部完成全部流程装配。"""
    with RagData.create(source, **kwargs) as app:
        app.init_collection()


def ingest(
    paths: Sequence[str],
    source: ConfigSource = None,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> int:
    """一键导入：内部完成全部流程装配，返回写入总条数。"""
    with RagData.create(source, **kwargs) as app:
        return app.ingest(paths, user_id=user_id)


__all__ = [
    "RagData",
    "build_settings",
    "build_logger",
    "build_record_class",
    "build_store",
    "build_nlp",
    "build_embedder",
    "build_pipeline",
    "init_collection",
    "ingest",
    "DEFAULT_TOP_N",
]
