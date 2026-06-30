from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SceneCore:
    root_layer: str
    payload_usdc: str
    solve_state: Optional[str]
    orientation: Dict[str, Any]


@dataclass(frozen=True)
class MaterialVariant:
    material_id: str
    label: str
    overrides_layer: str
    tags: List[str]


@dataclass(frozen=True)
class LayoutVariant:
    layout_id: str
    layout_file: str
    weight: float = 1.0


@dataclass(frozen=True)
class SceneEntry:
    scene_id: str
    category: str
    display_name: str
    core: SceneCore
    materials: List[MaterialVariant]
    layouts: List[LayoutVariant]
    variation_policy: Dict[str, Any]
    compat: Dict[str, Any]


@dataclass(frozen=True)
class BackgroundRegistry:
    version: str
    default_scene_id: str
    scenes: List[SceneEntry]
    registry_path: Path

    @property
    def root_dir(self) -> Path:
        return self.registry_path.parent

    def get_scene(self, scene_id: str) -> SceneEntry:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"Scene '{scene_id}' not found in registry")


REQUIRED_SCENE_KEYS = {
    "scene_id",
    "category",
    "display_name",
    "core",
    "materials",
    "layouts",
    "variation_policy",
    "compat",
}


def _require_keys(payload: Dict[str, Any], keys: set, where: str) -> None:
    missing = sorted(keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Missing keys in {where}: {missing}")


def _parse_scene(scene_payload: Dict[str, Any]) -> SceneEntry:
    _require_keys(scene_payload, REQUIRED_SCENE_KEYS, f"scene '{scene_payload.get('scene_id', '<unknown>')}'")

    core_payload = scene_payload["core"]
    _require_keys(core_payload, {"root_layer", "payload_usdc", "orientation"}, f"scene '{scene_payload['scene_id']}'.core")

    materials = [
        MaterialVariant(
            material_id=m["material_id"],
            label=m.get("label", m["material_id"]),
            overrides_layer=m["overrides_layer"],
            tags=list(m.get("tags", [])),
        )
        for m in scene_payload["materials"]
    ]
    if not materials:
        raise ValueError(f"Scene '{scene_payload['scene_id']}' must define at least one material")

    layouts = [
        LayoutVariant(
            layout_id=l["layout_id"],
            layout_file=l["layout_file"],
            weight=float(l.get("weight", 1.0)),
        )
        for l in scene_payload["layouts"]
    ]
    if not layouts:
        raise ValueError(f"Scene '{scene_payload['scene_id']}' must define at least one layout")

    core = SceneCore(
        root_layer=core_payload["root_layer"],
        payload_usdc=core_payload["payload_usdc"],
        solve_state=core_payload.get("solve_state"),
        orientation=dict(core_payload["orientation"]),
    )

    return SceneEntry(
        scene_id=scene_payload["scene_id"],
        category=scene_payload["category"],
        display_name=scene_payload["display_name"],
        core=core,
        materials=materials,
        layouts=layouts,
        variation_policy=dict(scene_payload["variation_policy"]),
        compat=dict(scene_payload["compat"]),
    )


def load_registry(registry_path: str | Path) -> BackgroundRegistry:
    path = Path(registry_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(payload, {"version", "default_scene_id", "scenes"}, "registry root")

    scenes = [_parse_scene(scene) for scene in payload["scenes"]]
    if not scenes:
        raise ValueError("Registry must contain at least one scene entry")

    scene_ids = {scene.scene_id for scene in scenes}
    default_scene_id = payload["default_scene_id"]
    if default_scene_id not in scene_ids:
        raise ValueError(f"default_scene_id '{default_scene_id}' not found in scenes")

    return BackgroundRegistry(
        version=str(payload["version"]),
        default_scene_id=default_scene_id,
        scenes=scenes,
        registry_path=path,
    )


def resolve_scene_paths(registry: BackgroundRegistry, scene: SceneEntry) -> Dict[str, Path]:
    root = registry.root_dir
    resolved = {
        "root_layer": (root / scene.core.root_layer).resolve(),
        "payload_usdc": (root / scene.core.payload_usdc).resolve(),
    }
    if scene.core.solve_state:
        resolved["solve_state"] = (root / scene.core.solve_state).resolve()
    return resolved
