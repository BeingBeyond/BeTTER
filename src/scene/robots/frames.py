from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .spec import EmbodimentSpec


@dataclass(frozen=True)
class FrameRoleSummary:
    legacy_eef_prim_path: str
    tool_base_prim_path: str
    tool_base_link_name: str
    task_tcp_link_name: str
    task_tcp_kind: str


class FrankaRobotiqFrameResolver:
    """Resolve Franka+Robotiq runtime tool-base and planner TCP semantics.

    Runtime USD exposes the Robotiq tool base as a prim, but the grasp-semantic
    TCP currently exists only as a planner URDF link. This resolver makes that
    distinction explicit and provides the fixed transform between the two.
    """

    def __init__(
        self,
        embodiment: EmbodimentSpec,
        registry_root: str | Path | None,
    ) -> None:
        if not embodiment.embodiment_id.startswith("franka_robotiq"):
            raise ValueError(
                "FrankaRobotiqFrameResolver only supports FrankaRobotiq embodiments, "
                f"got {embodiment.embodiment_id!r}."
            )
        self._embodiment = embodiment
        self._registry_root = (
            Path(registry_root).resolve() if registry_root is not None else None
        )
        self._tool_base_to_task_tcp_matrix: np.ndarray | None = None

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        embodiment_id: str = "franka_robotiq",
    ) -> "FrankaRobotiqFrameResolver":
        return cls(
            embodiment=registry.get_embodiment(embodiment_id),
            registry_root=registry.root_dir,
        )

    @property
    def embodiment(self) -> EmbodimentSpec:
        return self._embodiment

    @property
    def legacy_eef_prim_path(self) -> str:
        role = _mapping_role(self._embodiment.frame_roles, "legacy_eef_frame")
        prim_path_key = str(role.get("prim_path_key", "eef_frame"))
        return self._embodiment.resolve_prim_path(prim_path_key)

    @property
    def tool_base_prim_path(self) -> str:
        role = _mapping_role(self._embodiment.frame_roles, "tool_base_frame")
        prim_path_key = str(role.get("prim_path_key", "eef_base_frame"))
        return self._embodiment.resolve_prim_path(prim_path_key)

    @property
    def tool_base_link_name(self) -> str:
        role = _mapping_role(self._embodiment.frame_roles, "tool_base_frame")
        return str(role.get("planner_link_name", "robotiq_base_link"))

    @property
    def task_tcp_link_name(self) -> str:
        role = _mapping_role(self._embodiment.frame_roles, "task_tcp_frame")
        return str(role.get("link_name", role.get("planner_link_name", "robotiq_tcp")))

    @property
    def task_tcp_kind(self) -> str:
        role = _mapping_role(self._embodiment.frame_roles, "task_tcp_frame")
        return str(role.get("kind", "planner_link"))

    def summary(self) -> FrameRoleSummary:
        return FrameRoleSummary(
            legacy_eef_prim_path=self.legacy_eef_prim_path,
            tool_base_prim_path=self.tool_base_prim_path,
            tool_base_link_name=self.tool_base_link_name,
            task_tcp_link_name=self.task_tcp_link_name,
            task_tcp_kind=self.task_tcp_kind,
        )

    def tool_base_to_task_tcp_matrix(self) -> np.ndarray:
        if self._tool_base_to_task_tcp_matrix is None:
            self._tool_base_to_task_tcp_matrix = self._load_tool_base_to_task_tcp_matrix()
        return self._tool_base_to_task_tcp_matrix.copy()

    def task_tcp_to_tool_base_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.tool_base_to_task_tcp_matrix())

    def tool_base_pose_to_task_tcp_pose(
        self,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tool_base_pose = _pose_to_matrix(
            np.asarray(position, dtype=np.float64).reshape(3),
            np.asarray(orientation_wxyz, dtype=np.float64).reshape(4),
        )
        task_tcp_pose = tool_base_pose @ self.tool_base_to_task_tcp_matrix()
        return _matrix_to_pose(task_tcp_pose)

    def task_tcp_pose_to_tool_base_pose(
        self,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_tcp_pose = _pose_to_matrix(
            np.asarray(position, dtype=np.float64).reshape(3),
            np.asarray(orientation_wxyz, dtype=np.float64).reshape(4),
        )
        tool_base_pose = task_tcp_pose @ self.task_tcp_to_tool_base_matrix()
        return _matrix_to_pose(tool_base_pose)

    def _load_tool_base_to_task_tcp_matrix(self) -> np.ndarray:
        transform = self._embodiment.transforms.get("tool_base_to_task_tcp_pose_wxyz")
        if transform is not None:
            return _pose_wxyz_to_matrix(transform, "tool_base_to_task_tcp_pose_wxyz")

        return _read_fixed_link_transform(
            urdf_path=self._resolve_planner_urdf_path(),
            source_link=self.tool_base_link_name,
            target_link=self.task_tcp_link_name,
        )

    def _resolve_planner_urdf_path(self) -> Path:
        planner_asset = self._embodiment.planner_asset_path
        if planner_asset is not None and Path(planner_asset).is_absolute():
            return Path(planner_asset)
        if self._registry_root is None:
            raise ValueError(
                "A registry_root is required to resolve relative planner_asset_path "
                f"for embodiment {self._embodiment.embodiment_id!r}."
            )
        return self._embodiment.resolve_planner_asset_path(self._registry_root)


def _mapping_role(frame_roles: Mapping[str, Any], role_name: str) -> Mapping[str, Any]:
    value = frame_roles.get(role_name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Frame role {role_name!r} is missing or is not a mapping.")
    return value


def _pose_wxyz_to_matrix(values: list[float], key: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (7,):
        raise ValueError(
            f"{key!r} must contain [x, y, z, qw, qx, qy, qz], got {array.shape}."
        )
    return _pose_to_matrix(array[:3], array[3:])


def _read_fixed_link_transform(
    urdf_path: Path,
    source_link: str,
    target_link: str,
) -> np.ndarray:
    if not urdf_path.exists():
        raise FileNotFoundError(f"Planner URDF not found: {urdf_path}")

    root = ET.parse(urdf_path).getroot()
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        if parent_link is None or child_link is None:
            continue

        parent_to_child = _joint_origin_matrix(joint)
        adjacency.setdefault(parent_link, []).append((child_link, parent_to_child))
        adjacency.setdefault(child_link, []).append(
            (parent_link, np.linalg.inv(parent_to_child))
        )

    if source_link == target_link:
        return np.eye(4, dtype=np.float64)

    queue: list[tuple[str, np.ndarray]] = [(source_link, np.eye(4, dtype=np.float64))]
    visited = {source_link}
    while queue:
        link_name, source_to_link = queue.pop(0)
        for next_link, link_to_next in adjacency.get(link_name, []):
            if next_link in visited:
                continue
            source_to_next = source_to_link @ link_to_next
            if next_link == target_link:
                return source_to_next
            visited.add(next_link)
            queue.append((next_link, source_to_next))

    raise ValueError(
        f"No fixed-joint path from {source_link!r} to {target_link!r} in {urdf_path}."
    )


def _joint_origin_matrix(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = np.zeros(3, dtype=np.float64)
    rpy = np.zeros(3, dtype=np.float64)
    if origin is not None:
        xyz_text = origin.get("xyz")
        rpy_text = origin.get("rpy")
        if xyz_text:
            xyz = np.asarray([float(value) for value in xyz_text.split()], dtype=np.float64)
        if rpy_text:
            rpy = np.asarray([float(value) for value in rpy_text.split()], dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"Joint {joint.get('name')!r} has invalid origin xyz: {xyz}.")
    if rpy.shape != (3,):
        raise ValueError(f"Joint {joint.get('name')!r} has invalid origin rpy: {rpy}.")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = xyz
    matrix[:3, :3] = _rpy_to_matrix(rpy)
    return matrix


def _pose_to_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    matrix[:3, :3] = _quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    return matrix


def _matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    position = transform[:3, 3].copy()
    quat_wxyz = _matrix_to_quat_wxyz(transform[:3, :3])
    return position, quat_wxyz


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )

    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Rotation matrix produced a zero quaternion.")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in np.asarray(rpy, dtype=np.float64).reshape(3)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x
