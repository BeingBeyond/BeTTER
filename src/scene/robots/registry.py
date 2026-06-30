from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Type

from .base import RobotBase
from .franka_robotiq import FrankaRobotiqRobot
from .panda import PandaRobot
from .spec import EmbodimentSpec


@dataclass(frozen=True)
class EmbodimentRegistry:
    version: str
    default_embodiment_id: str
    embodiments: List[EmbodimentSpec]
    registry_path: Path

    @property
    def root_dir(self) -> Path:
        return self.registry_path.parent

    def get_embodiment(self, embodiment_id: str) -> EmbodimentSpec:
        for embodiment in self.embodiments:
            if embodiment.embodiment_id == embodiment_id:
                return embodiment
        raise KeyError(f"Embodiment '{embodiment_id}' not found in registry")

    def get_default_embodiment(self) -> EmbodimentSpec:
        return self.get_embodiment(self.default_embodiment_id)


REQUIRED_EMBODIMENT_KEYS = {
    "embodiment_id",
    "display_name",
    "robot_class",
    "asset_root",
    "asset_path",
    "prim_paths",
    "joint_names",
    "transforms",
    "compat",
}


def _require_keys(payload: Dict[str, Any], keys: set, where: str) -> None:
    missing = sorted(keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Missing keys in {where}: {missing}")


def _parse_embodiment(payload: Dict[str, Any]) -> EmbodimentSpec:
    emb_id = payload.get("embodiment_id", "<unknown>")
    _require_keys(payload, REQUIRED_EMBODIMENT_KEYS, f"embodiment '{emb_id}'")

    if not payload["prim_paths"]:
        raise ValueError(f"Embodiment '{emb_id}' must define non-empty prim_paths")

    return EmbodimentSpec(
        embodiment_id=str(payload["embodiment_id"]),
        display_name=str(payload["display_name"]),
        robot_class=str(payload["robot_class"]),
        asset_root=str(payload["asset_root"]),
        asset_path=str(payload["asset_path"]),
        prim_paths=dict(payload["prim_paths"]),
        joint_names={k: list(v) for k, v in dict(payload["joint_names"]).items()},
        transforms={k: list(v) for k, v in dict(payload["transforms"]).items()},
        compat=dict(payload["compat"]),
        runtime_asset_path=(
            str(payload["runtime_asset_path"])
            if payload.get("runtime_asset_path") is not None
            else None
        ),
        planner_asset_path=(
            str(payload["planner_asset_path"])
            if payload.get("planner_asset_path") is not None
            else None
        ),
        runtime_control=dict(payload.get("runtime_control", {})),
        frame_roles=dict(payload.get("frame_roles", {})),
    )


def load_registry(registry_path: str | Path) -> EmbodimentRegistry:
    path = Path(registry_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(payload, {"version", "default_embodiment_id", "embodiments"}, "registry root")

    embodiments = [_parse_embodiment(emb) for emb in payload["embodiments"]]
    if not embodiments:
        raise ValueError("Registry must contain at least one embodiment entry")

    embodiment_ids = {emb.embodiment_id for emb in embodiments}
    default_embodiment_id = str(payload["default_embodiment_id"])
    if default_embodiment_id not in embodiment_ids:
        raise ValueError(
            f"default_embodiment_id '{default_embodiment_id}' not found in embodiments"
        )

    return EmbodimentRegistry(
        version=str(payload["version"]),
        default_embodiment_id=default_embodiment_id,
        embodiments=embodiments,
        registry_path=path,
    )


def resolve_embodiment_paths(
    registry: EmbodimentRegistry, embodiment: EmbodimentSpec
) -> Dict[str, Path]:
    root = embodiment.resolve_asset_root(registry.root_dir)
    asset = embodiment.resolve_asset_path(registry.root_dir)
    return {
        "asset_root": root,
        "asset_path": asset,
        "runtime_asset_path": embodiment.resolve_runtime_asset_path(registry.root_dir),
        "planner_asset_path": embodiment.resolve_planner_asset_path(registry.root_dir),
    }


ROBOT_CLASS_REGISTRY: Dict[str, Type[RobotBase]] = {
    "PandaRobot": PandaRobot,
    "FrankaRobotiqRobot": FrankaRobotiqRobot,
}


def create_robot(
    instance_id: str,
    prim_path: str,
    embodiment: EmbodimentSpec,
) -> RobotBase:
    robot_cls = ROBOT_CLASS_REGISTRY.get(embodiment.robot_class)
    if robot_cls is None:
        raise KeyError(
            f"Robot class '{embodiment.robot_class}' is not registered for embodiment "
            f"'{embodiment.embodiment_id}'"
        )
    return robot_cls(instance_id=instance_id, prim_path=prim_path, spec=embodiment)
