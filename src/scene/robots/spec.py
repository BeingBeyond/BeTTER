from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EmbodimentSpec:
    """Configuration for a robot embodiment.

    The spec stores path metadata, frame/joint naming, and compatibility aliases
    so robot-specific hardcoding can be moved from code to data.
    """

    embodiment_id: str
    display_name: str
    robot_class: str
    asset_root: str
    asset_path: str
    prim_paths: Dict[str, str]
    joint_names: Dict[str, List[str]]
    transforms: Dict[str, List[float]]
    compat: Dict[str, Any]
    runtime_asset_path: Optional[str] = None
    planner_asset_path: Optional[str] = None
    runtime_control: Dict[str, Any] = field(default_factory=dict)
    frame_roles: Dict[str, Any] = field(default_factory=dict)

    def resolve_asset_root(self, registry_root: Path) -> Path:
        root = Path(self.asset_root)
        if not root.is_absolute():
            root = (registry_root / root).resolve()
        return root

    def resolve_asset_path(self, registry_root: Path) -> Path:
        asset = Path(self.asset_path)
        if asset.is_absolute():
            return asset
        return (self.resolve_asset_root(registry_root) / asset).resolve()

    def resolve_runtime_asset_path(self, registry_root: Path) -> Path:
        asset = Path(self.runtime_asset_path or self.asset_path)
        if asset.is_absolute():
            return asset
        return (self.resolve_asset_root(registry_root) / asset).resolve()

    def resolve_planner_asset_path(self, registry_root: Path) -> Path:
        asset = Path(self.planner_asset_path or self.asset_path)
        if asset.is_absolute():
            return asset
        return (self.resolve_asset_root(registry_root) / asset).resolve()

    def resolve_prim_path(self, key: str) -> str:
        if key in self.prim_paths:
            return self.prim_paths[key]

        aliases = self.compat.get("prim_path_aliases", {})
        canonical_key = aliases.get(key)
        if canonical_key and canonical_key in self.prim_paths:
            return self.prim_paths[canonical_key]

        raise KeyError(f"Prim path '{key}' not found for embodiment '{self.embodiment_id}'")

    def get_transform(self, key: str) -> List[float]:
        if key in self.transforms:
            return self.transforms[key]

        aliases = self.compat.get("transform_aliases", {})
        canonical_key = aliases.get(key)
        if canonical_key and canonical_key in self.transforms:
            return self.transforms[canonical_key]

        raise KeyError(f"Transform '{key}' not found for embodiment '{self.embodiment_id}'")
