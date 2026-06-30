from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .manager_v2 import ResolvedBackgroundVariant, SceneManagerV2
from .registry import BackgroundRegistry, load_registry
from .runtime_scene import BackgroundScene


class SceneSwitchStrategy(str, Enum):
    LOAD_UNLOAD = "load_unload"
    PRELOAD_ACTIVATE = "preload_activate"


class PreloadInitialPolicy(str, Enum):
    FIRST_ACTIVE = "first_active"
    ALL_INACTIVE = "all_inactive"


@dataclass(frozen=True)
class SceneRequest:
    scene_id: str
    material_id: Optional[str] = None
    layout_id: Optional[str] = None
    seed: Optional[int] = None


class SceneManager:
    """Thin orchestration layer around background scene resolution and runtime scene lifecycle."""

    def __init__(
        self,
        registry: BackgroundRegistry,
        switch_strategy: SceneSwitchStrategy = SceneSwitchStrategy.LOAD_UNLOAD,
        preload_initial_policy: PreloadInitialPolicy = PreloadInitialPolicy.FIRST_ACTIVE,
    ):
        self.registry = registry
        self._resolver = SceneManagerV2(registry)
        self.switch_strategy = SceneSwitchStrategy(switch_strategy)
        self.preload_initial_policy = PreloadInitialPolicy(preload_initial_policy)

        self._current_scene: Optional[BackgroundScene] = None
        self._scene_cache: Dict[Tuple[str, str, str], BackgroundScene] = {}
        self._active_scene_key: Optional[Tuple[str, str, str]] = None

    @classmethod
    def from_registry_file(
        cls,
        registry_path: str | Path,
        switch_strategy: SceneSwitchStrategy = SceneSwitchStrategy.LOAD_UNLOAD,
        preload_initial_policy: PreloadInitialPolicy = PreloadInitialPolicy.FIRST_ACTIVE,
    ) -> "SceneManager":
        return cls(
            load_registry(registry_path),
            switch_strategy=switch_strategy,
            preload_initial_policy=preload_initial_policy,
        )

    def configure_switching(
        self,
        strategy: SceneSwitchStrategy,
        preload_initial_policy: Optional[PreloadInitialPolicy] = None,
    ) -> None:
        self.switch_strategy = SceneSwitchStrategy(strategy)
        if preload_initial_policy is not None:
            self.preload_initial_policy = PreloadInitialPolicy(preload_initial_policy)

    def list_scenes(self, category: Optional[str] = None) -> List[str]:
        return self._resolver.list_scenes(category=category)

    def resolve_variant(
        self,
        scene_id: Optional[str] = None,
        material_id: Optional[str] = None,
        layout_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> ResolvedBackgroundVariant:
        return self._resolver.resolve_variant(
            scene_id=scene_id,
            material_id=material_id,
            layout_id=layout_id,
            seed=seed,
        )

    def create_scene(
        self,
        resolved_variant: ResolvedBackgroundVariant,
        prim_path: str = "/World/Background",
    ) -> BackgroundScene:
        return BackgroundScene(resolved_variant=resolved_variant, prim_path=prim_path)

    @staticmethod
    def _scene_key(resolved: ResolvedBackgroundVariant) -> Tuple[str, str, str]:
        return (resolved.scene_id, resolved.material_id, resolved.layout_id)

    def _deactivate_active_scene(self, stage) -> None:
        if self._active_scene_key is None:
            return
        active_scene = self._scene_cache.get(self._active_scene_key)
        if active_scene is not None:
            active_scene.deactivate(stage)
        self._active_scene_key = None

    def preload_scenes(
        self,
        stage,
        scene_requests: List[SceneRequest],
        capture_light_baseline: bool = True,
        prim_root: str = "/World/BackgroundPool",
    ) -> List[BackgroundScene]:
        loaded_scenes: List[BackgroundScene] = []
        for req in scene_requests:
            resolved = self.resolve_variant(
                scene_id=req.scene_id,
                material_id=req.material_id,
                layout_id=req.layout_id,
                seed=req.seed,
            )
            key = self._scene_key(resolved)
            if key in self._scene_cache:
                loaded_scenes.append(self._scene_cache[key])
                continue

            prim_path = f"{prim_root}/{resolved.scene_id}_{resolved.material_id}_{resolved.layout_id}"
            scene = self.create_scene(resolved_variant=resolved, prim_path=prim_path)
            scene.load_into_stage(stage, activate_on_load=False)
            if capture_light_baseline:
                scene.activate(stage)
                scene.capture_light_baseline(stage)
                scene.deactivate(stage)
            self._scene_cache[key] = scene
            loaded_scenes.append(scene)

        self._active_scene_key = None
        if not loaded_scenes:
            return loaded_scenes

        if self.preload_initial_policy == PreloadInitialPolicy.FIRST_ACTIVE:
            first_scene = loaded_scenes[0]
            first_key = self._scene_key(first_scene.resolved_variant)
            first_scene.activate(stage)
            self._active_scene_key = first_key
            self._current_scene = first_scene
        else:
            self._current_scene = None

        return loaded_scenes

    def activate_preloaded_scene(
        self,
        stage,
        scene_id: str,
        material_id: Optional[str] = None,
        layout_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> BackgroundScene:
        resolved = self.resolve_variant(
            scene_id=scene_id,
            material_id=material_id,
            layout_id=layout_id,
            seed=seed,
        )
        key = self._scene_key(resolved)
        if key not in self._scene_cache:
            raise KeyError(f"Scene not preloaded: {key}")

        if self._active_scene_key is not None and self._active_scene_key != key:
            self._deactivate_active_scene(stage)

        target = self._scene_cache[key]
        target.activate(stage)
        self._active_scene_key = key
        self._current_scene = target
        return target

    def load_scene(
        self,
        stage,
        scene_id: Optional[str] = None,
        material_id: Optional[str] = None,
        layout_id: Optional[str] = None,
        seed: Optional[int] = None,
        prim_path: str = "/World/Background",
        capture_light_baseline: bool = True,
    ) -> BackgroundScene:
        if self.switch_strategy == SceneSwitchStrategy.PRELOAD_ACTIVATE:
            if scene_id is None:
                scene_id = self.registry.default_scene_id
            return self.activate_preloaded_scene(
                stage=stage,
                scene_id=scene_id,
                material_id=material_id,
                layout_id=layout_id,
                seed=seed,
            )

        if self._current_scene is not None:
            self._current_scene.unload(stage)

        resolved = self.resolve_variant(
            scene_id=scene_id,
            material_id=material_id,
            layout_id=layout_id,
            seed=seed,
        )

        scene = self.create_scene(resolved_variant=resolved, prim_path=prim_path)
        scene.load_into_stage(stage, activate_on_load=True)
        if capture_light_baseline:
            scene.capture_light_baseline(stage)
        self._current_scene = scene
        return scene

    def switch_scene(
        self,
        stage,
        scene_id: Optional[str] = None,
        material_id: Optional[str] = None,
        layout_id: Optional[str] = None,
        seed: Optional[int] = None,
        prim_path: str = "/World/Background",
        capture_light_baseline: bool = True,
    ) -> BackgroundScene:
        return self.load_scene(
            stage=stage,
            scene_id=scene_id,
            material_id=material_id,
            layout_id=layout_id,
            seed=seed,
            prim_path=prim_path,
            capture_light_baseline=capture_light_baseline,
        )

    def get_current_scene(self) -> Optional[BackgroundScene]:
        return self._current_scene

    def unload_current_scene(self, stage) -> None:
        if self.switch_strategy == SceneSwitchStrategy.PRELOAD_ACTIVATE:
            self._deactivate_active_scene(stage)
            self._current_scene = None
            return

        if self._current_scene is None:
            return
        self._current_scene.unload(stage)
        self._current_scene = None
