from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Type


def _module_alias(duoduoclip_root: str) -> str:
    digest = hashlib.sha1(duoduoclip_root.encode("utf-8")).hexdigest()[:10]
    return f"_better_duoduoclip_{digest}"


def _ensure_package(alias: str, src_root: Path) -> ModuleType:
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing

    init_file = src_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias,
        init_file,
        submodule_search_locations=[str(src_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for DuoduoCLIP package at {src_root}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _bridge_src_subpackage(alias: str, subpackage: str) -> None:
    src_pkg = sys.modules.get("src")
    if src_pkg is None:
        return

    module_name = f"src.{subpackage}"
    if module_name in sys.modules:
        return

    bridged = importlib.import_module(f"{alias}.{subpackage}")
    sys.modules[module_name] = bridged
    setattr(src_pkg, subpackage, bridged)


def load_duoduoclip_class(duoduoclip_root: str) -> Type:
    src_root = Path(duoduoclip_root).resolve() / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"DuoduoCLIP src directory not found: {src_root}")

    alias = _module_alias(str(src_root))
    _ensure_package(alias, src_root)
    _bridge_src_subpackage(alias, "loss")
    module = importlib.import_module(f"{alias}.model.duoduoclip")
    return module.DuoduoCLIP
