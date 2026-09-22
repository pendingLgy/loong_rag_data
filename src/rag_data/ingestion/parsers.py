# 文档解析：将 PDF、Word、Markdown 等转为纯文本。

from __future__ import annotations

import os
from typing import Optional

from rag_data.exceptions import ParseError
from rag_data.logging.base import LoggerAdapter

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
BINARY_SUFFIXES = {".pdf", ".docx", ".doc"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | BINARY_SUFFIXES


def parse_document(path: str, logger: Optional[LoggerAdapter] = None) -> str:
    """解析单个文档并返回纯文本。"""
    if not os.path.exists(path):
        raise ParseError("文件不存在：" + path)
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ParseError("暂不支持的文档格式：" + suffix)
    if suffix in TEXT_SUFFIXES:
        return _read_text(path)
    # TODO: 使用 unstructured 或 MinerU 解析 PDF 与 Word，输出带 Markdown 层级的文本，
    #       并清理页眉页脚与换行断词；依赖缺失时抛 ParserDependencyError。
    if logger is not None:
        logger.warning("二进制文档解析尚未实现", source_path=path, suffix=suffix)
    raise NotImplementedError("TODO: 实现 PDF 与 Word 的解析")


def _read_text(path: str) -> str:
    """读取纯文本类文档。"""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()
