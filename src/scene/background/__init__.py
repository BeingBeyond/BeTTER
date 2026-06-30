from .registry import (
    BackgroundRegistry,
    LayoutVariant,
    MaterialVariant,
    SceneCore,
    SceneEntry,
    load_registry,
    resolve_scene_paths,
)
from .manager_v2 import BackgroundHandle, ResolvedBackgroundVariant, SceneManagerV2
from .runtime_scene import BackgroundScene
from .scene_manager import PreloadInitialPolicy, SceneManager, SceneRequest, SceneSwitchStrategy

__all__ = [
    "BackgroundRegistry",
    "SceneCore",
    "SceneEntry",
    "MaterialVariant",
    "LayoutVariant",
    "load_registry",
    "resolve_scene_paths",
    "SceneManagerV2",
    "ResolvedBackgroundVariant",
    "BackgroundHandle",
    "BackgroundScene",
    "SceneManager",
    "SceneSwitchStrategy",
    "PreloadInitialPolicy",
    "SceneRequest",
]
