# OpenAI 嵌入实现；Qwen 等 OpenAI 兼容服务复用同一套请求格式。
#
# 走标准库 HTTP，不引入额外依赖；网络细节集中在本模块，便于替换与测试。

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from rag_data.embedding.base import BaseEmbeddingProvider
from rag_data.exceptions import EmbeddingError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """调用 OpenAI 的 embeddings 接口。"""

    backend = "openai"
    default_model = "text-embedding-3-small"
    default_dim = 1536
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    def encode(self, texts: List[str]) -> List[List[float]]:
        """请求一次 embeddings，返回与输入同序的向量列表。"""
        data = self._post_json(self._endpoint(), self._payload(texts))
        return self._parse(data)

    # ---------------- 请求构造（子类可覆盖以适配不同厂商） ----------------

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/embeddings"

    def _payload(self, texts: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "input": list(texts)}
        # 仅在显式配置了维度、且与默认维度不同时才传 dimensions，兼容不支持该参数的模型。
        configured = self._settings.embedding.dim
        if configured and configured != self.default_dim:
            payload["dimensions"] = configured
        return payload

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key}

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送 JSON 请求；子类或测试可覆盖本方法以隔离网络。"""
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            timeout = self._settings.embedding.timeout
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise EmbeddingError("嵌入请求失败（HTTP " + str(exc.code) + "）：" + detail) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError("嵌入请求无法连接：" + str(exc.reason)) from exc

    def _parse(self, data: Dict[str, Any]) -> List[List[float]]:
        """按 index 排序还原响应顺序，保证与输入一一对应。"""
        items = data.get("data") or []
        if not items:
            raise EmbeddingError("嵌入响应缺少 data 字段：" + str(list(data.keys())))
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [[float(value) for value in item["embedding"]] for item in ordered]
