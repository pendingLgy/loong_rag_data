# 大模型调用封装：provider 由配置选择，模型句柄可注入以便测试。

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_data.config import LLMSettings
from rag_data.llm.registry import create_llm_provider
from rag_data.logging.base import LoggerAdapter


class LLM:
    """大模型调用封装：按配置装配 provider，注入的 model 优先且绕过配置。"""

    def __init__(self, settings: LLMSettings, logger: LoggerAdapter, model: Optional[Any] = None) -> None:
        self._settings = settings
        self._logger = logger
        # 注入的 model 只需实现 chat(messages, **overrides)，便于测试与自定义。
        self._model = model
        self._provider: Optional[Any] = None

    @property
    def provider(self) -> Any:
        """实际使用的调用对象：注入的 model 优先，否则按配置装配 provider。"""
        return self._target()

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> str:
        """单轮调用大模型，返回回复文本。"""
        return self._target().chat(messages, **overrides)

    def complete(self, prompt: str, system: str = "", **overrides: Any) -> str:
        """以单条用户消息完成一次生成，可选系统提示。"""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)

    # ---------------- 内部实现 ----------------

    def _target(self) -> Any:
        if self._model is not None:
            return self._model
        if self._provider is None:
            name = self._settings.provider
            self._provider = create_llm_provider(name, settings=self._settings, logger=self._logger)
            self._logger.info(
                "LLM provider 已装配",
                provider=name,
                model=getattr(self._provider, "model", ""),
            )
        return self._provider

