from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import RobotBase
from .spec import EmbodimentSpec


class PandaRobot(RobotBase):
    """Franka Panda embodiment with legacy action compatibility."""

    def __init__(self, instance_id: str, prim_path: str, spec: EmbodimentSpec) -> None:
        super().__init__(instance_id=instance_id, prim_path=prim_path, spec=spec)
        self._world_position = np.zeros(3, dtype=np.float32)
        self._world_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._eef_pose = np.zeros(6, dtype=np.float32)
        self._gripper_closeness = np.zeros(1, dtype=np.float32)
        self._articulation = None

    def initialize(self) -> None:
        from omni.isaac.core.articulations import Articulation  # type: ignore[import]

        self._articulation = Articulation(self._prim_path)
        self._articulation.initialize()
        self._initialized = True

    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._world_position.copy(), self._world_orientation_wxyz.copy()

    def set_world_pose(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        if position is not None:
            pos = np.asarray(position, dtype=np.float32)
            if pos.shape != (3,):
                raise ValueError(f"position must have shape (3,), got {pos.shape}")
            self._world_position = pos.copy()

        if orientation is not None:
            quat = np.asarray(orientation, dtype=np.float32)
            if quat.shape != (4,):
                raise ValueError(f"orientation must have shape (4,), got {quat.shape}")
            self._world_orientation_wxyz = quat.copy()

    def get_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        position = self._eef_pose[:3].copy()
        orientation = self._eef_pose[3:].copy()
        return position, orientation

    def set_eef_target(self, eef_pose: np.ndarray) -> None:
        pose = np.asarray(eef_pose, dtype=np.float32)
        if pose.shape != (6,):
            raise ValueError(f"eef_pose must have shape (6,), got {pose.shape}")
        self._eef_pose = pose.copy()

    def set_gripper_closeness(self, closeness: np.ndarray) -> None:
        command = np.asarray(closeness, dtype=np.float32)
        if command.shape != (1,):
            raise ValueError(
                f"gripper_closeness must have shape (1,), got {command.shape}"
            )
        self._gripper_closeness = np.clip(command, 0.0, 1.0)

    def get_joint_positions(self) -> Dict[str, float]:
        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        if not dof_names:
            return {}

        positions = articulation.get_joint_positions()
        if positions is None:
            return {}

        positions_array = np.asarray(positions, dtype=float).reshape(-1)
        if positions_array.shape[0] != len(dof_names):
            raise RuntimeError(
                f"Joint position size mismatch for '{self._prim_path}': "
                f"{positions_array.shape[0]} values for {len(dof_names)} joints."
            )

        return {
            joint_name: float(position)
            for joint_name, position in zip(dof_names, positions_array)
        }

    def set_joint_positions(self, positions: Dict[str, float]) -> None:
        articulation = self._require_articulation()
        if not positions:
            return

        dof_names = self._get_dof_names(articulation)
        unknown_joints = sorted(set(positions.keys()) - set(dof_names))
        if unknown_joints:
            raise KeyError(
                f"Unknown joints for articulation '{self._prim_path}': {unknown_joints}. "
                f"Available joints: {dof_names}"
            )

        joint_indices = np.asarray(
            [articulation.get_dof_index(joint_name) for joint_name in positions.keys()],
            dtype=np.int64,
        )
        joint_targets = np.asarray(
            [float(positions[joint_name]) for joint_name in positions.keys()],
            dtype=np.float32,
        )
        articulation.set_joint_positions(
            positions=joint_targets,
            joint_indices=joint_indices,
        )

    def get_joint_velocities(self) -> Dict[str, float]:
        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        if not dof_names:
            return {}

        velocities = articulation.get_joint_velocities()
        if velocities is None:
            return {}

        velocities_array = np.asarray(velocities, dtype=float).reshape(-1)
        if velocities_array.shape[0] != len(dof_names):
            raise RuntimeError(
                f"Joint velocity size mismatch for '{self._prim_path}': "
                f"{velocities_array.shape[0]} values for {len(dof_names)} joints."
            )

        return {
            joint_name: float(velocity)
            for joint_name, velocity in zip(dof_names, velocities_array)
        }

    def set_joint_velocities(self, velocities: Dict[str, float]) -> None:
        articulation = self._require_articulation()
        if not velocities:
            return

        dof_names = self._get_dof_names(articulation)
        unknown_joints = sorted(set(velocities.keys()) - set(dof_names))
        if unknown_joints:
            raise KeyError(
                f"Unknown joints for articulation '{self._prim_path}': {unknown_joints}. "
                f"Available joints: {dof_names}"
            )

        joint_indices = np.asarray(
            [articulation.get_dof_index(joint_name) for joint_name in velocities.keys()],
            dtype=np.int64,
        )
        joint_velocities = np.asarray(
            [float(velocities[joint_name]) for joint_name in velocities.keys()],
            dtype=np.float32,
        )
        articulation.set_joint_velocities(
            velocities=joint_velocities,
            joint_indices=joint_indices,
        )

    def _require_articulation(self):
        if self._articulation is None:
            raise RuntimeError(f"PandaRobot '{self._prim_path}' is not initialized.")
        if (
            hasattr(self._articulation, "handles_initialized")
            and not self._articulation.handles_initialized
        ):
            self._articulation.initialize()
        return self._articulation

    @staticmethod
    def _get_dof_names(articulation) -> list[str]:
        dof_names = articulation.dof_names
        if dof_names is None:
            return []
        return [str(name) for name in dof_names]
