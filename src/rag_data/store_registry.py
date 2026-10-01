# 存储后端注册表：配置只写后端名即可切换。
#
# 后端名到实现的映射：既可为点分路径（惰性导入），也可为类对象
# （函数内定义的本地类无法按路径导入）。内置 milvus；用户可在
# storage.store_modules 中登记自有模块来扩充后端。

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Type

from rag_data.exceptions import ConfigError
from rag_data.registry import Ref, Registry

if TYPE_CHECKING:  # pragma: no cover
    from rag_data.config import Settings
    from rag_data.logging.base import LoggerAdapter

# 内置后端名称到类路径的映射，按需惰性导入。
BUILTIN_BACKENDS: Dict[str, str] = {
    "milvus": "rag_data.storage.milvus_store.MilvusVectorStore",
}

_REGISTRY = Registry("存储后端", BUILTIN_BACKENDS)


def register_store(name: str, ref: Ref) -> None:
    """登记存储后端：ref 可为类对象，也可为点分路径字符串。"""
    _REGISTRY.register(name, ref)


def register_backend(name: str):
    """装饰器形式登记存储后端。"""
    return _REGISTRY.decorator(name)


def is_registered(name: str) -> bool:
    """判断后端是否已登记。"""
    return name in _REGISTRY


def available_backends() -> List[str]:
    """返回已登记的后端名，按字典序排列。"""
    return _REGISTRY.names()


def resolve_store(name: str) -> Type[Any]:
    """按后端名取得实现类，点分路径登记的在此时惰性导入。"""
    return _REGISTRY.resolve(name)


def load_store_modules(refs: Sequence[str]) -> None:
    """导入用户提供的存储后端模块，导入即完成登记；重复调用幂等。"""
    # 约定：模块在导入期调用 register_store 或 register_backend，
    # 因此只需被导入一次，登记即生效，装配代码无需改动。
    for ref in refs:
        _import_store_module(ref)


def _import_store_module(ref: str) -> None:
    """按取值形态导入：以 .py 结尾视为文件路径，其余视为点分模块路径。"""
    if ref.endswith(".py"):
        _import_store_file(ref)
    else:
        importlib.import_module(ref)


def _import_store_file(path: str) -> None:
    """以文件路径导入模块；同一路径只执行一次，避免重复登记。"""
    module_name = _plugin_module_name(path)
    if module_name in sys.modules:
        return
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigError("存储后端模块文件不存在：" + str(path))
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ConfigError("无法加载存储后端模块：" + str(path))
    module = importlib.util.module_from_spec(spec)
    # 先入 sys.modules 再执行，模块内自引用（如 dataclass）才能解析到自身。
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _plugin_module_name(path: str) -> str:
    """按绝对路径派生稳定的模块名，作为 sys.modules 的去重键。"""
    digest = hashlib.sha1(str(Path(path).expanduser().resolve()).encode("utf-8")).hexdigest()
    return "rag_data_store_plugin_" + digest[:12]


def create_store(
    name: str,
    settings: Settings,
    logger: LoggerAdapter,
    **kwargs: Any,
) -> Any:
    """按后端名创建实例：先按配置导入用户模块，再以 settings 与 logger 实例化。"""
    load_store_modules(settings.storage.store_modules)
    return _REGISTRY.create(name, settings=settings, logger=logger, **kwargs)