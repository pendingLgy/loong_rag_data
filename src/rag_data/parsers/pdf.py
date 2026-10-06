# PDF 解析与切块（使用 pdfplumber 精准获取中文物理坐标，并智能过滤页眉和页脚）

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_data.exceptions import ParseError, ParserDependencyError
from rag_data.logging.base import LoggerAdapter
from rag_data.parsers.base import DocumentParser

PACKAGES = "pdfplumber"
INSTALL_HINT = "pip install pdfplumber"
CHUNK_INSTALL_HINT = "pip install -e '.[parsers]'"

PARAGRAPH_END_PUNCT = ("。", "！", "？", "…", ".", "!", "?", "”", "’")


class PdfParser(DocumentParser):
    """PDF：使用 pdfplumber 提取正文，支持过滤页眉页脚，基于物理坐标与自然排版逻辑智能重组段落。"""

    SUFFIXES = frozenset({".pdf"})

    def __init__(
            self,
            max_chars: int = 1000,
            safe_max_chars: int = 2000,
            header_ratio: float = 0.07,  # 顶部页眉过滤比例（如 7%）
            footer_ratio: float = 0.07,  # 底部页脚过滤比例（如 7%，以 842pt 高度计算约 59pt，可完美覆盖 785~801pt 的页码）
    ):
        super().__init__(max_chars, safe_max_chars)
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio

    def parse(self, path: str, logger: Optional[LoggerAdapter] = None) -> str:
        pdfplumber = _import_deps()
        try:
            all_paragraphs: List[str] = []
            current_para_buffer: str = ""

            # 记录上一行的坐标信息
            prev_x1: Optional[float] = None
            prev_top: Optional[float] = None

            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_width = float(page.width)
                    page_height = float(page.height)

                    # 动态计算当前页的页眉页脚边界阈值
                    # 比如 page_height = 842pt，footer_limit = 842 * (1 - 0.07) = 783.06pt
                    # 你的页码 top 为 785.90pt > 783.06pt，能够被完美精准过滤！
                    header_limit = page_height * self.header_ratio
                    footer_limit = page_height * (1.0 - self.footer_ratio)

                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True
                    )
                    if not words:
                        continue

                    for word in words:
                        text = word["text"].strip()
                        if not text:
                            continue

                        curr_x0 = word["x0"]
                        curr_x1 = word["x1"]
                        curr_top = word["top"]
                        curr_bottom = word["bottom"]

                        # -------------------------------------------------------------
                        # 页眉与页脚过滤逻辑：
                        # 只要文本块触发了顶部或底部阈值范围，直接过滤 skip
                        # -------------------------------------------------------------
                        if curr_top < header_limit or curr_bottom > footer_limit:
                            continue

                        # 缓冲区为空时直接初始化
                        if not current_para_buffer:
                            current_para_buffer = text
                            prev_x1 = curr_x1
                            prev_top = curr_top
                            continue

                        buffer_trimmed = current_para_buffer.rstrip()

                        # -------------------------------------------------------------
                        # 1. 位置与状态计算
                        # -------------------------------------------------------------
                        y_diff = abs(curr_top - prev_top) if prev_top is not None else 0.0
                        is_same_y = y_diff <= 3.0  # 同一行容差 3.0pt

                        # 判断上一行结尾是否未铺满页面右侧（以页面宽度的 85% 为临界值）
                        prev_line_not_full = (prev_x1 or 0.0) < (page_width * 0.85)

                        # 判断上一行结尾是否有句末标点
                        ends_with_punct = buffer_trimmed.endswith(PARAGRAPH_END_PUNCT)

                        has_newline = not is_same_y

                        # -------------------------------------------------------------
                        # 2. 核心段落判定逻辑：
                        # -------------------------------------------------------------
                        is_new_paragraph = has_newline and (prev_line_not_full or ends_with_punct)

                        # -------------------------------------------------------------
                        # 3. 拼接与重组逻辑
                        # -------------------------------------------------------------
                        if is_new_paragraph:
                            all_paragraphs.append(current_para_buffer)
                            current_para_buffer = text
                        elif is_same_y:
                            last_char = buffer_trimmed[-1] if buffer_trimmed else ""
                            first_char = text[0]
                            is_chinese = ('\u4e00' <= last_char <= '\u9fa5') or ('\u4e00' <= first_char <= '\u9fa5')
                            if is_chinese or last_char in "：:" or first_char in "：:":
                                current_para_buffer += text
                            else:
                                current_para_buffer += " " + text
                        else:
                            last_char = buffer_trimmed[-1] if buffer_trimmed else ""
                            first_char = text[0]
                            is_chinese = ('\u4e00' <= last_char <= '\u9fa5') or ('\u4e00' <= first_char <= '\u9fa5')
                            if is_chinese:
                                current_para_buffer += text
                            else:
                                current_para_buffer += " " + text

                        prev_x1 = curr_x1
                        prev_top = curr_top

            if current_para_buffer:
                all_paragraphs.append(current_para_buffer)

        except Exception as exc:
            raise ParseError(f"PDF 解析失败：{path}：{str(exc)}") from exc

        full_text = "\n".join(all_paragraphs)
        if logger is not None:
            logger.info("PDF 解析完成", source_path=path, chars=len(full_text))

        return full_text

    def chunk(
            self,
            text: str,
            nlp: Any = None,
            logger: Optional[LoggerAdapter] = None,
    ) -> List[str]:
        if not text or not text.strip():
            return []
        handle = nlp
        if handle is None:
            raise ParserDependencyError(f"文本切块需要 spacy 及模型，请执行 {CHUNK_INSTALL_HINT}")
        return self._process_text_recursive(text=text, overlap_prefix="", nlp=handle)


def _import_deps() -> Any:
    try:
        import pdfplumber
        return pdfplumber
    except ImportError as exc:
        raise ParserDependencyError(f"PDF 解析需要 {PACKAGES}，请执行 {INSTALL_HINT}") from exc