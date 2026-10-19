# 大模型模块单元测试：纯注册表（只登记实现类）、内置 provider 与 LLMHandler 处理器。
#
# 不触网络：provider 仅校验类属性与配置解析，调用路径用注入的假模型隔离；
# 注册表只登记实现类，实例化统一由 LLMHandler 在调用时完成。

import pytest

from rag_data.config import LLMSettings, RagSetting
from rag_data.exceptions import ConfigError, LLMError
from rag_data.llm import (
    BUILTIN_PROVIDERS,
    LLMHandler,
    BaseLLMProvider,
    DeepSeekLLMProvider,
    OpenAILLMProvider,
    QwenLLMProvider,
    available_llm_providers,
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


def test_provider_class_instantiation():
    # 注册表只登记类，实例化由调用方完成。
    provider_cls = resolve_llm_provider('openai')
    provider = provider_cls(LLMSettings())
    assert isinstance(provider, BaseLLMProvider)
    assert provider.model == OpenAILLMProvider.default_model


def test_resolve_unknown_provider_raises():
    with pytest.raises(ConfigError):
        resolve_llm_provider('nope')


def test_register_llm_by_decorator():
    name = 'dummy-llm'

    @register_llm_provider(name)
    class _Dummy(BaseLLMProvider):
        def chat(self, messages, **overrides):
            return 'ok'

    assert is_llm_registered(name)
    assert resolve_llm_provider(name) is _Dummy
    assert name in available_llm_providers()


def test_register_llm_by_class_object():
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


def test_model_falls_back_to_default():
    provider = QwenLLMProvider(LLMSettings())
    assert provider.model == 'qwen-plus'
    assert provider.base_url == QwenLLMProvider.default_base_url


def test_model_configured_wins():
    settings = LLMSettings(provider='qwen', model='qwen-max', base_url='http://localhost:1234/v1')
    provider = QwenLLMProvider(settings)
    assert provider.model == 'qwen-max'
    assert provider.base_url == 'http://localhost:1234/v1'


# ---------------- API Key 解析 ----------------


def test_api_key_from_config(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    provider = OpenAILLMProvider(LLMSettings(api_key='cfg-key'))
    assert provider.api_key == 'cfg-key'


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    provider = OpenAILLMProvider(LLMSettings())
    assert provider.api_key == 'env-key'


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    provider = OpenAILLMProvider(LLMSettings())
    with pytest.raises(LLMError):
        provider.api_key


# ---------------- 请求参数构造 ----------------


def test_request_kwargs_base_fields():
    provider = OpenAILLMProvider(LLMSettings())
    kwargs = provider._build_request_kwargs([{'role': 'user', 'content': 'x'}], {})
    assert kwargs['model'] == OpenAILLMProvider.default_model
    assert kwargs['messages'] == [{'role': 'user', 'content': 'x'}]
    assert kwargs['temperature'] == LLMSettings().temperature
    assert 'max_tokens' not in kwargs


def test_request_kwargs_max_tokens_optional():
    provider = OpenAILLMProvider(LLMSettings(max_tokens=64))
    kwargs = provider._build_request_kwargs([], {})
    assert kwargs['max_tokens'] == 64


def test_request_kwargs_applies_overrides():
    provider = OpenAILLMProvider(LLMSettings())
    kwargs = provider._build_request_kwargs([], {'top_p': 0.5})
    assert kwargs['top_p'] == 0.5


def test_parse_extracts_message_content():
    provider = OpenAILLMProvider(LLMSettings())
    data = {'choices': [{'message': {'content': 'hello'}}]}
    assert provider._parse(data) == 'hello'


def test_parse_without_choices_raises():
    provider = OpenAILLMProvider(LLMSettings())
    with pytest.raises(LLMError):
        provider._parse({})


# ---------------- LLMHandler 处理器 ----------------


def test_handler_uses_injected_model():
    model = _FakeModel(reply='hi')
    client = LLMHandler(LLMSettings(), model=model)
    out = client.chat([{'role': 'user', 'content': 'q'}])
    assert out == 'hi'
    assert model.calls == [([{'role': 'user', 'content': 'q'}], {})]


def test_handler_provider_defaults_to_config():
    client = LLMHandler(LLMSettings(provider='deepseek'))
    assert isinstance(client.provider, DeepSeekLLMProvider)


def test_handler_keeps_injected_model_over_provider():
    model = _FakeModel()
    client = LLMHandler(LLMSettings(provider='qwen'), model=model)
    assert client.provider is model


def test_handler_complete_builds_messages():
    model = _FakeModel(reply='ok')
    client = LLMHandler(LLMSettings(), model=model)
    assert client.complete('hi', system='sys') == 'ok'
    messages, overrides = model.calls[0]
    assert messages[0] == {'role': 'system', 'content': 'sys'}
    assert messages[1] == {'role': 'user', 'content': 'hi'}


def test_handler_complete_without_system():
    model = _FakeModel()
    client = LLMHandler(LLMSettings(), model=model)
    client.complete('hi')
    messages, _ = model.calls[0]
    assert len(messages) == 1
    assert messages[0]['role'] == 'user'


def test_handler_chat_forwards_overrides():
    model = _FakeModel()
    client = LLMHandler(LLMSettings(), model=model)
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


# ---------------- 注册表只注册、不实例化 ----------------


def test_registry_has_no_factory_api():
    import rag_data.llm.registry_llm as registry_module

    assert not hasattr(registry_module, 'create_llm_provider')
    assert not hasattr(registry_module, 'create')


def test_registry_resolves_user_class_object():
    class _UserProvider(BaseLLMProvider):
        backend = None  # 不自动注册，改由 register_llm 显式登记

        def chat(self, messages, **overrides):
            return 'user'

    register_llm('user-llm', _UserProvider)
    assert resolve_llm_provider('user-llm') is _UserProvider
    assert is_llm_registered('user-llm')


def test_handler_instantiates_user_registered_provider():
    class _UserProvider(BaseLLMProvider):
        backend = None

        def __init__(self, settings):
            super().__init__(settings)
            self.calls = []

        def chat(self, messages, **overrides):
            self.calls.append(list(messages))
            return 'custom-reply'

    register_llm('handler-llm', _UserProvider)
    handler = LLMHandler(LLMSettings(provider='handler-llm'))
    assert isinstance(handler.provider, _UserProvider)
    assert handler.complete('hi') == 'custom-reply'


def test_llm_package_exports_handler_only():
    # 处理器以 LLMHandler 之名导出，旧的 LLM 名称不再存在。
    import rag_data.llm as llm_package

    assert 'LLMHandler' in llm_package.__all__
    assert 'LLM' not in llm_package.__all__
    assert not hasattr(llm_package, 'LLM')


def test_modules_renamed():
    # 旧模块 client 与 registry 已更名为 llm_handler 与 registry_llm。
    import rag_data.llm.llm_handler as handler_module
    import rag_data.llm.registry_llm as registry_module

    assert handler_module.__name__.endswith('.llm_handler')
    assert registry_module.__name__.endswith('.registry_llm')


def test_handler_defers_instantiation():
    # 构造处理器只登记配置，不实例化 provider；首次调用才实例化。
    created = []

    class _Lazy(BaseLLMProvider):
        backend = None

        def __init__(self, settings):
            super().__init__(settings)
            created.append(self)

        def chat(self, messages, **overrides):
            return 'ok'

    register_llm('lazy-llm', _Lazy)
    handler = LLMHandler(LLMSettings(provider='lazy-llm'))
    assert created == []          # 构造阶段未实例化
    handler.complete('hi')
    assert len(created) == 1      # 调用后才实例化


def test_handler_propagates_unknown_provider():
    handler = LLMHandler(LLMSettings(provider='definitely-missing'))
    with pytest.raises(ConfigError):
        handler.complete('hi')


def test_handler_caches_provider():
    # 同一处理器重复取 provider 只实例化一次。
    created = []

    class _Counting(BaseLLMProvider):
        backend = None

        def __init__(self, settings):
            super().__init__(settings)
            created.append(self)

        def chat(self, messages, **overrides):
            return 'ok'

    register_llm('counting-llm', _Counting)
    handler = LLMHandler(LLMSettings(provider='counting-llm'))
    first = handler.provider
    second = handler.provider
    assert first is second
    assert len(created) == 1


def test_old_module_paths_are_gone():
    # 旧模块 client 与 registry 已删除，导入应失败，避免残留引用。
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module('rag_data.llm.client')
    with pytest.raises(ImportError):
        importlib.import_module('rag_data.llm.registry')

