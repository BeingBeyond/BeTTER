from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime_scene import BackgroundScene

from .registry import BackgroundRegistry, SceneEntry, load_registry, resolve_scene_paths


@dataclass(frozen=True)
class ResolvedBackgroundVariant:
    scene_id: str
    material_id: str
    layout_id: str
    root_layer_path: Path
    payload_usdc_path: Path
    solve_state_path: Optional[Path]
    orientation: Dict


@dataclass
class BackgroundHandle:
    prim_path: str
    resolved_variant: ResolvedBackgroundVariant
    loaded: bool = True


class SceneManagerV2:
    """Background-scene manager V2 (background-only, no robot/embodiment)."""

    def __init__(self, registry: BackgroundRegistry):
        self.registry = registry

    def create_runtime_scene(
        self,
        resolved_variant: ResolvedBackgroundVariant,
        prim_path: str = "/World/Background",
    ) -> "BackgroundScene":
        from .runtime_scene import BackgroundScene

        return BackgroundScene(resolved_variant=resolved_variant, prim_path=prim_path)

    @classmethod
    def from_registry_file(cls, registry_path: str | Path) -> "SceneManagerV2":
        return cls(load_registry(registry_path))

    def list_scenes(self, category: Optional[str] = None) -> List[str]:
        if category is None:
            return [s.scene_id for s in self.registry.scenes]
        return [s.scene_id for s in self.registry.scenes if s.category == category]

    def get_scene(self, scene_id: Optional[str] = None) -> SceneEntry:
        target_id = scene_id or self.registry.default_scene_id
        return self.registry.get_scene(target_id)

    def resolve_variant(
        self,
        scene_id: Optional[str] = None,
        material_id: Optional[str] = None,
        layout_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> ResolvedBackgroundVariant:
        scene = self.get_scene(scene_id)
        rng = random.Random(seed)

        material_map = {m.material_id: m for m in scene.materials}
        layout_map = {l.layout_id: l for l in scene.layouts}

        selected_material = material_id or scene.materials[0].material_id
        if selected_material not in material_map:
            raise KeyError(f"Material '{selected_material}' not found in scene '{scene.scene_id}'")

        if layout_id is None:
            weighted_layouts = []
            for l in scene.layouts:
                weighted_layouts.extend([l.layout_id] * max(1, int(round(l.weight * 10))))
            selected_layout = rng.choice(weighted_layouts)
        else:
            selected_layout = layout_id

        if selected_layout not in layout_map:
            raise KeyError(f"Layout '{selected_layout}' not found in scene '{scene.scene_id}'")

        resolved_paths = resolve_scene_paths(self.registry, scene)
        for key, p in resolved_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Resolved path for '{key}' does not exist: {p}")

        return ResolvedBackgroundVariant(
            scene_id=scene.scene_id,
            material_id=selected_material,
            layout_id=selected_layout,
            root_layer_path=resolved_paths["root_layer"],
            payload_usdc_path=resolved_paths["payload_usdc"],
            solve_state_path=resolved_paths.get("solve_state"),
            orientation=scene.core.orientation,
        )

    def load_background(
        self,
        resolved_variant: ResolvedBackgroundVariant,
        prim_path: str = "/World/Background",
    ) -> BackgroundHandle:
        return BackgroundHandle(prim_path=prim_path, resolved_variant=resolved_variant, loaded=True)

    def load_runtime_scene(
        self,
        resolved_variant: ResolvedBackgroundVariant,
        stage,
        prim_path: str = "/World/Background",
        capture_light_baseline: bool = False,
    ) -> "BackgroundScene":
        scene = self.create_runtime_scene(resolved_variant=resolved_variant, prim_path=prim_path)
        scene.load_into_stage(stage)
        if capture_light_baseline:
            scene.capture_light_baseline(stage)
        return scene

    def enforce_static_orientation(self, handle: BackgroundHandle) -> Dict:
        return dict(handle.resolved_variant.orientation)

    def apply_material(self, handle: BackgroundHandle, material_id: str) -> None:
        if not handle.loaded:
            raise RuntimeError("Cannot apply material on unloaded background handle")
        handle.resolved_variant = ResolvedBackgroundVariant(
            scene_id=handle.resolved_variant.scene_id,
            material_id=material_id,
            layout_id=handle.resolved_variant.layout_id,
            root_layer_path=handle.resolved_variant.root_layer_path,
            payload_usdc_path=handle.resolved_variant.payload_usdc_path,
            solve_state_path=handle.resolved_variant.solve_state_path,
            orientation=handle.resolved_variant.orientation,
        )

    def apply_layout_metadata(self, handle: BackgroundHandle, layout_id: str) -> None:
        if not handle.loaded:
            raise RuntimeError("Cannot apply layout on unloaded background handle")
        handle.resolved_variant = ResolvedBackgroundVariant(
            scene_id=handle.resolved_variant.scene_id,
            material_id=handle.resolved_variant.material_id,
            layout_id=layout_id,
            root_layer_path=handle.resolved_variant.root_layer_path,
            payload_usdc_path=handle.resolved_variant.payload_usdc_path,
            solve_state_path=handle.resolved_variant.solve_state_path,
            orientation=handle.resolved_variant.orientation,
        )

    def unload_background(self, handle: BackgroundHandle) -> None:
        handle.loaded = False
