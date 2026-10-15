# 大模型模块单元测试：注册表、内置 provider 与 LLM 装配封装。
#
# 不触网络：provider 仅校验类属性与配置解析，调用路径用注入的假模型隔离。

import pytest

from rag_data.config import LLMSettings, Settings
from rag_data.exceptions import ConfigError, LLMError
from rag_data.llm import (
    BUILTIN_PROVIDERS,
    LLM,
    BaseLLMProvider,
    DeepSeekLLMProvider,
    OpenAILLMProvider,
    QwenLLMProvider,
    available_llm_providers,
    create_llm_provider,
    is_llm_registered,
    register_llm,
    register_llm_provider,
    resolve_llm_provider,
)


# ---------------- 测试替身 ----------------


class _FakeModel:
    # 受控模型：记录每次 chat 的入参，返回固定文本。
    def __init__(self, reply='reply'):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **overrides):
        self.calls.append((list(messages), dict(overrides)))
        return self.reply


# ---------------- 注册表 ----------------


def test_builtin_providers_declared():
    assert set(BUILTIN_PROVIDERS) == {'openai', 'qwen', 'deepseek'}


def test_available_providers_sorted():
    providers = available_llm_providers()
    assert providers == sorted(providers)
    for name in ('openai', 'qwen', 'deepseek'):
        assert name in providers
        assert is_llm_registered(name)


def test_resolve_returns_provider_class():
    assert resolve_llm_provider('qwen') is QwenLLMProvider
    assert resolve_llm_provider('deepseek') is DeepSeekLLMProvider


def test_create_llm_provider_builds_instance(logger):
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    assert isinstance(provider, BaseLLMProvider)
    assert provider.model == OpenAILLMProvider.default_model


def test_resolve_unknown_provider_raises():
    with pytest.raises(ConfigError):
        resolve_llm_provider('nope')


def test_register_llm_by_decorator(logger):
    name = 'dummy-llm'

    @register_llm_provider(name)
    class _Dummy(BaseLLMProvider):
        def chat(self, messages, **overrides):
            return 'ok'

    assert is_llm_registered(name)
    assert resolve_llm_provider(name) is _Dummy
    assert name in available_llm_providers()


def test_register_llm_by_class_object(logger):
    class _Local(BaseLLMProvider):
        def chat(self, messages, **overrides):
            return 'ok'

    register_llm('local-llm', _Local)
    assert resolve_llm_provider('local-llm') is _Local



# ---------------- 内置 provider ----------------


def test_qwen_overrides_openai_defaults():
    assert issubclass(QwenLLMProvider, OpenAILLMProvider)
    assert QwenLLMProvider.default_model == 'qwen-plus'
    assert QwenLLMProvider.api_key_env == 'DASHSCOPE_API_KEY'
    assert 'compatible-mode' in QwenLLMProvider.default_base_url


def test_deepseek_overrides_openai_defaults():
    assert issubclass(DeepSeekLLMProvider, OpenAILLMProvider)
    assert DeepSeekLLMProvider.default_model == 'deepseek-chat'
    assert DeepSeekLLMProvider.api_key_env == 'DEEPSEEK_API_KEY'
    assert DeepSeekLLMProvider.default_base_url == 'https://api.deepseek.com/v1'


def test_openai_defaults():
    assert OpenAILLMProvider.default_model == 'gpt-4o-mini'
    assert OpenAILLMProvider.api_key_env == 'OPENAI_API_KEY'


def test_model_falls_back_to_default(logger):
    provider = create_llm_provider('qwen', settings=LLMSettings(), logger=logger)
    assert provider.model == 'qwen-plus'
    assert provider.base_url == QwenLLMProvider.default_base_url


def test_model_configured_wins(logger):
    settings = LLMSettings(provider='qwen', model='qwen-max', base_url='http://localhost:1234/v1')
    provider = create_llm_provider('qwen', settings=settings, logger=logger)
    assert provider.model == 'qwen-max'
    assert provider.base_url == 'http://localhost:1234/v1'


# ---------------- API Key 解析 ----------------


def test_api_key_from_config(logger, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    provider = create_llm_provider('openai', settings=LLMSettings(api_key='cfg-key'), logger=logger)
    assert provider.api_key == 'cfg-key'


def test_api_key_from_env(logger, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    assert provider.api_key == 'env-key'


def test_api_key_missing_raises(logger, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    with pytest.raises(LLMError):
        provider.api_key


# ---------------- 请求参数构造 ----------------


def test_request_kwargs_base_fields(logger):
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    kwargs = provider._build_request_kwargs([{'role': 'user', 'content': 'x'}], {})
    assert kwargs['model'] == OpenAILLMProvider.default_model
    assert kwargs['messages'] == [{'role': 'user', 'content': 'x'}]
    assert kwargs['temperature'] == LLMSettings().temperature
    assert 'max_tokens' not in kwargs


def test_request_kwargs_max_tokens_optional(logger):
    provider = create_llm_provider('openai', settings=LLMSettings(max_tokens=64), logger=logger)
    kwargs = provider._build_request_kwargs([], {})
    assert kwargs['max_tokens'] == 64


def test_request_kwargs_applies_overrides(logger):
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    kwargs = provider._build_request_kwargs([], {'top_p': 0.5})
    assert kwargs['top_p'] == 0.5


def test_parse_extracts_message_content(logger):
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    data = {'choices': [{'message': {'content': 'hello'}}]}
    assert provider._parse(data) == 'hello'


def test_parse_without_choices_raises(logger):
    provider = create_llm_provider('openai', settings=LLMSettings(), logger=logger)
    with pytest.raises(LLMError):
        provider._parse({})



# ---------------- LLM 装配封装 ----------------


def test_llm_uses_injected_model(logger):
    model = _FakeModel(reply='hi')
    client = LLM(LLMSettings(), logger, model=model)
    out = client.chat([{'role': 'user', 'content': 'q'}])
    assert out == 'hi'
    assert model.calls == [([{'role': 'user', 'content': 'q'}], {})]


def test_llm_provider_defaults_to_config(logger):
    client = LLM(LLMSettings(provider='deepseek'), logger)
    assert isinstance(client.provider, DeepSeekLLMProvider)


def test_llm_keeps_injected_model_over_provider(logger):
    model = _FakeModel()
    client = LLM(LLMSettings(provider='qwen'), logger, model=model)
    assert client.provider is model


def test_llm_complete_builds_messages(logger):
    model = _FakeModel(reply='ok')
    client = LLM(LLMSettings(), logger, model=model)
    assert client.complete('hi', system='sys') == 'ok'
    messages, overrides = model.calls[0]
    assert messages[0] == {'role': 'system', 'content': 'sys'}
    assert messages[1] == {'role': 'user', 'content': 'hi'}


def test_llm_complete_without_system(logger):
    model = _FakeModel()
    client = LLM(LLMSettings(), logger, model=model)
    client.complete('hi')
    messages, _ = model.calls[0]
    assert len(messages) == 1
    assert messages[0]['role'] == 'user'


def test_llm_chat_forwards_overrides(logger):
    model = _FakeModel()
    client = LLM(LLMSettings(), logger, model=model)
    client.chat([], top_p=0.3)
    assert model.calls[0][1] == {'top_p': 0.3}


# ---------------- 配置分区 ----------------


def test_llm_settings_defaults():
    settings = LLMSettings()
    assert settings.provider == 'openai'
    assert settings.model == ''
    assert settings.temperature == 0.7
    assert settings.max_tokens == 0
    assert settings.timeout == 60.0


def test_settings_exposes_llm_section():
    settings = Settings.load_mapping()
    assert isinstance(settings.llm, LLMSettings)


def test_settings_loads_llm_overrides():
    settings = Settings.load_mapping(llm={'provider': 'qwen', 'model': 'qwen-max', 'max_tokens': 128})
    assert settings.llm.provider == 'qwen'
    assert settings.llm.model == 'qwen-max'
    assert settings.llm.max_tokens == 128


def test_settings_rejects_unknown_llm_field():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings.load_mapping(llm={'bogus': 1})


