# EPUB 解析与切块（全内存安全防护完整版）

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from rag_data.exceptions import ParseError, ParserDependencyError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DEFAULT_MODEL, DocumentParser, load_nlp

PACKAGES = "ebooklib 与 beautifulsoup4"
INSTALL_HINT = "pip install -e '.[epub]'"
CHUNK_INSTALL_HINT = "pip install -e '.[parsers]'"


class EpubParser(DocumentParser):
    """EPUB：按段落收集并依据字符大小邻近聚合，支持段落级与句级重叠切块（含全套内存安全优化）。"""

    SUFFIXES = frozenset({".epub"})

    def __init__(self, max_chars: int = 1000, safe_max_chars: int = 2000):
        super().__init__(max_chars, safe_max_chars)

    def parse(self, path: str, logger: Optional[LoggerAdapter] = None) -> str:
        """解析 EPUB：按 reading order 提取各章节段落并序列化返回（优化内存回收）。"""
        epub, bs_class = _import_deps()
        book = None
        try:
            book = epub.read_epub(path)
            chapter_paragraphs = _extract_chapter_paragraphs(book, bs_class)
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"EPUB 解析失败：{path}：{exc}") from exc
        finally:
            # 及时释放 ebooklib 的 Book 对象，避免大文件常驻内存
            if book is not None:
                del book

        all_paras = []
        for title, paras in chapter_paragraphs:
            if title:
                all_paras.append(f"# {title}")
            all_paras.extend(paras)

        full_text = "\n".join(all_paras)
        if logger is not None:
            logger.info(
                "EPUB 解析完成",
                source_path=path,
                chapters=len(chapter_paragraphs),
                chars=len(full_text),
            )
        return full_text

    def chunk(
            self,
            text: str,
            nlp: Any = None,
            logger: Optional[LoggerAdapter] = None,
    ) -> List[str]:
        """切块规则：基于 raw_chapters 仅通过一次单层循环完成切分与组装，支持超长章节 1/4 重叠二次切分。"""
        if not text or not text.strip():
            return []

        handle = nlp
        if handle is None:
            raise ParserDependencyError(
                f"文本切块需要 spacy 及模型，请执行 {CHUNK_INSTALL_HINT}"
            )

        # 直接获取原始章节列表
        raw_chapters = text.split("#")

        all_chunks: List[str] = []

        for idx, chap in enumerate(raw_chapters):
            overlap_prefix = all_chunks[-1] if len(all_chunks) else ""

            if not chap.strip():
                continue
            current_overlap_prefix = self._extract_overlap_prefix([overlap_prefix])

            min_chunks = self._process_text_recursive(text=chap, overlap_prefix=current_overlap_prefix, nlp=handle)
            all_chunks.extend(min_chunks)

        return all_chunks


def _import_deps() -> Tuple[Any, Any]:
    """导入 ebooklib 与 beautifulsoup4；未安装时给出安装指引。"""
    try:
        from bs4 import BeautifulSoup
        from ebooklib import epub
    except ImportError as exc:
        raise ParserDependencyError(f"EPUB 解析需要 {PACKAGES}，请执行 {INSTALL_HINT}") from exc
    return epub, BeautifulSoup


def _load_spacy(model_name: str, logger: Optional[LoggerAdapter] = None) -> Any:
    """加载 spaCy 句柄；未安装或模型缺失时给出安装指引。"""
    handle = load_nlp(model_name, logger)
    if handle is None:
        raise ParserDependencyError(
            f"文本切块需要 spacy 及模型 '{model_name}'，请执行 {CHUNK_INSTALL_HINT}"
        )
    return handle


def _extract_chapter_paragraphs(book: Any, bs_class: Any) -> List[Tuple[str, List[str]]]:
    """按 spine 提取各章节的标题与段落列表。"""
    chapters: List[Tuple[str, List[str]]] = []
    for spine_entry in book.spine:
        spine_id = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = book.get_item_with_id(spine_id)
        if item is not None and _is_chapter(item):
            title, paragraphs = _parse_chapter_html(item.get_content(), bs_class)
            if paragraphs:
                chapters.append((title, paragraphs))
    return chapters


def _is_chapter(item: Any) -> bool:
    is_chapter = getattr(item, "is_chapter", None)
    return bool(is_chapter()) if callable(is_chapter) else False


def _parse_chapter_html(content: bytes, bs_class: Any) -> Tuple[str, List[str]]:
    """解析单章 HTML：提取首个标题并按标签精准收集段落。"""
    soup = bs_class(content, "html.parser")

    title = ""
    heading_tag = soup.find(["h1", "h2", "h3"])
    if heading_tag:
        title = heading_tag.get_text().strip()
        heading_tag.decompose()

    paragraphs: List[str] = []
    for tag in soup.find_all(["p", "h4", "h5", "h6", "li"]):
        text = tag.get_text().strip()
        cleaned = " ".join(text.split())
        if cleaned:
            paragraphs.append(cleaned)

    if not paragraphs:
        raw_text = soup.get_text()
        paragraphs = [line.strip() for line in raw_text.splitlines() if line.strip()]

    return title, paragraphs
